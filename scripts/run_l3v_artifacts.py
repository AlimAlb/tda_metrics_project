"""Wave L3V — PCA-2-визуализация LLM-контролей (§5.2–5.4): артефакты + фигуры.

Строит из embedding-кэша (Qwen2.5-3B, chat_knowledge_v1, mean_answer):
- results/raw/l3v/l3d_pca2_r0.parquet — length-matched L3d r0, слои 9/18/27/36;
- results/raw/l3v/l4a_mixture_pca2_r0.parquet — вложенные смеси r0, слои 9/18/36,
  alpha 0.25/0.5/1.0 (P 500 correct + Qa 500, метка исходного типа);
- results/raw/l3v/l3_direct_pca2_layers_r0.parquet — direct L3 r0, 4 слоя;
- results/raw/l3v/l3_barcodes_layer18_r0.parquet — интервалы H0/H1 трёх облаков
  (реплика wave-BC: сиды 404/405/406, n=500, PCA-16 слоя 18) + кросс-чек
  статистик с results/tables/llm_barcodes_stats.csv;
- manifest.json с происхождением (сид-схема wave3/wave_bc, PCA-правило, хэши).

Фигуры: l3d_length_matched_pca2_layers.png, l4a_mixture_pca2_layers.png,
l3_direct_pca2_layers.png, l3_barcodes_layer18.png, layerwise_llm_dynamics.png
(из results/raw/llm/results.parquet), layerwise_rtd_cka.png (из
results/raw/wave5/layerwise.parquet). PCA-обучения: fit на полном dev-correct
пуле слоя (16 компонент; pc1/pc2 — первые две компоненты рабочего PCA-16).
Существующие raw (llm/wave5/bc) не модифицируются.
"""
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np

RAW_L3V = 'results/raw/l3v'
FIGURES = 'results/figures'
TABLES = 'results/tables'
CACHE_DIR = 'embeddings/l3r_cache'
LOG_PATH = 'logs/l3v_run.log'

LAYERS_ALL = [9, 18, 27, 36]
LAYERS_L4A = [9, 18, 36]
ALPHAS_L4A = [0.25, 0.5, 1.0]
N_CLOUD = 500
COLORS = {'correct': 'tab:blue', 'hallucinated': 'tab:red'}

SECTION_STATUS = {}
_STARTED = time.perf_counter()


class Tee:
    def __init__(self, path):
        self._file = open(path, 'a', encoding='utf-8')

    def write(self, text):
        sys.__stdout__.write(text)
        self._file.write(text)

    def flush(self):
        sys.__stdout__.flush()
        self._file.flush()


def progress(text):
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(65536), b''):
            digest.update(chunk)
    return digest.hexdigest()


def section(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            progress(f'=== SECTION {name} START ===')
            started = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                SECTION_STATUS[name] = 'ok'
                progress(f'=== SECTION {name} OK ({time.perf_counter() - started:.1f}s) ===')
                return result
            except Exception:
                SECTION_STATUS[name] = 'failed: ' + traceback.format_exc(limit=3)
                progress(f'=== SECTION {name} FAIL ===')
                progress(traceback.format_exc(limit=3))
                return None
        return wrapper
    return decorator


@section('load')
def section_load(_):
    from tda_metrics.embedding_cache import load_cache
    from tda_metrics.llm_experiments import prompt_id_split, prompt_ids_hash
    meta, vectors, manifest = load_cache(CACHE_DIR, expect={
        'model': 'Qwen/Qwen2.5-3B-Instruct',
        'pooling': 'mean_answer',
        'input_format': 'chat_knowledge_v1',
    })
    split = prompt_id_split(meta, seed=0)
    progress(f'кэш: {len(meta)} строк, vectors {vectors.shape}, dev {len(split["dev"])}')
    return {
        'meta': meta, 'vectors': vectors, 'manifest': manifest, 'split': split,
        'dev_hash': prompt_ids_hash(split['dev']),
    }


def pools_for_layer(state, layer):
    from tda_metrics.llm_experiments import layer_clouds
    meta, vectors = state['meta'], state['vectors']
    dev_ids = state['split']['dev']
    allowed = set(dev_ids)
    pools = {}
    for kind in ('correct', 'hallucinated'):
        matrix = layer_clouds(meta, vectors, dev_ids, kind, layer)
        mask = ((meta['answer_kind'] == kind) & (meta['layer_index'] == layer)
                & meta['prompt_id'].isin(allowed))
        pool_meta = meta[mask].reset_index(drop=True)
        if len(pool_meta) != len(matrix):
            raise RuntimeError(f'пул {kind}@{layer}: meta {len(pool_meta)} != облако {len(matrix)}')
        pools[kind] = (matrix, pool_meta)
    return pools


def pca16_models(state):
    from tda_metrics.reduction import pca_fit
    models = {}
    sizes = {}
    for layer in LAYERS_ALL:
        pools = pools_for_layer(state, layer)
        ref = pools['correct'][0]
        models[layer] = pca_fit(ref, n_components=16)
        sizes[layer] = {'correct': int(len(ref)), 'hallucinated': int(len(pools['hallucinated'][0]))}
        progress(f'PCA-16 слой {layer}: fit на dev-correct {ref.shape}')
    return models, sizes


def _subsample_idx(pool_size, seed, n=N_CLOUD):
    return np.random.default_rng(seed).choice(pool_size, size=n, replace=False)


@section('artifacts')
def section_artifacts(state):
    import pandas as pd
    from tda_metrics.llm_experiments import length_matched_pair
    from tda_metrics.mixtures import make_alpha_replacement
    from tda_metrics.reduction import pca_transform
    from tda_metrics.llm_experiments import prompt_ids_hash

    os.makedirs(RAW_L3V, exist_ok=True)
    models, pool_sizes = state['models'], state['pool_sizes']
    meta, vectors, split = state['meta'], state['vectors'], state['split']

    rows_l3d = []
    rows_direct = []
    rows_l4a = []
    l3d_prompt_hashes = {}
    direct_prompt_hashes = {}
    for layer in LAYERS_ALL:
        pools = pools_for_layer(state, layer)
        model = models[layer]

        c_m, h_m, meta_c, meta_h = length_matched_pair(
            meta, vectors, split['dev'], layer, n_max=N_CLOUD)
        if len(c_m) != N_CLOUD or len(h_m) != N_CLOUD:
            raise RuntimeError(f'l3d@{layer}: {len(c_m)}/{len(h_m)} != {N_CLOUD}')
        proj_c = pca_transform(c_m, model=model)[0]
        proj_h = pca_transform(h_m, model=model)[0]
        for j in range(N_CLOUD):
            rows_l3d.append({
                'layer': layer, 'group': 'correct',
                'pc1': float(proj_c[j, 0]), 'pc2': float(proj_c[j, 1]),
                'prompt_id': meta_c['prompt_id'].iloc[j],
                'answer_length': float(meta_c['n_pooled'].iloc[j]),
                'repeat': 0,
            })
            rows_l3d.append({
                'layer': layer, 'group': 'hallucinated',
                'pc1': float(proj_h[j, 0]), 'pc2': float(proj_h[j, 1]),
                'prompt_id': meta_h['prompt_id'].iloc[j],
                'answer_length': float(meta_h['n_pooled'].iloc[j]),
                'repeat': 0,
            })
        l3d_prompt_hashes[layer] = {
            'correct': prompt_ids_hash(sorted(meta_c['prompt_id'])),
            'hallucinated': prompt_ids_hash(sorted(meta_h['prompt_id'])),
        }

        pool_c, pool_meta_c = pools['correct']
        pool_h, pool_meta_h = pools['hallucinated']
        idx = _subsample_idx(len(pool_c), 200)
        p_raw = pool_c[idx]
        h_raw = pool_h[idx]
        p_meta_sel = pool_meta_c.iloc[idx].reset_index(drop=True)
        h_meta_sel = pool_meta_h.iloc[idx].reset_index(drop=True)
        proj_p = pca_transform(p_raw, model=model)[0]
        proj_h_dir = pca_transform(h_raw, model=model)[0]
        for j in range(N_CLOUD):
            rows_direct.append({
                'layer': layer, 'group': 'correct',
                'pc1': float(proj_p[j, 0]), 'pc2': float(proj_p[j, 1]),
                'prompt_id': p_meta_sel['prompt_id'].iloc[j],
                'answer_length': float(p_meta_sel['n_pooled'].iloc[j]),
                'repeat': 0,
            })
            rows_direct.append({
                'layer': layer, 'group': 'hallucinated',
                'pc1': float(proj_h_dir[j, 0]), 'pc2': float(proj_h_dir[j, 1]),
                'prompt_id': h_meta_sel['prompt_id'].iloc[j],
                'answer_length': float(h_meta_sel['n_pooled'].iloc[j]),
                'repeat': 0,
            })
        direct_prompt_hashes[layer] = {
            'correct': prompt_ids_hash(sorted(p_meta_sel['prompt_id'])),
            'hallucinated': prompt_ids_hash(sorted(h_meta_sel['prompt_id'])),
        }

        if layer in LAYERS_L4A:
            permutation = np.random.default_rng(100).permutation(N_CLOUD)
            for alpha in ALPHAS_L4A:
                for j in range(N_CLOUD):
                    rows_l4a.append({
                        'layer': layer, 'alpha': alpha,
                        'pc1': float(proj_p[j, 0]), 'pc2': float(proj_p[j, 1]),
                        'group': 'correct', 'side': 'P',
                        'prompt_id': p_meta_sel['prompt_id'].iloc[j],
                        'answer_length': float(p_meta_sel['n_pooled'].iloc[j]),
                        'repeat': 0,
                    })
                q = make_alpha_replacement(p_raw, h_raw, alpha, permutation)
                replaced = np.zeros(N_CLOUD, dtype=bool)
                replaced[permutation[:int(round(alpha * N_CLOUD))]] = True
                proj_q = pca_transform(q, model=model)[0]
                for j in range(N_CLOUD):
                    rows_l4a.append({
                        'layer': layer, 'alpha': alpha,
                        'pc1': float(proj_q[j, 0]), 'pc2': float(proj_q[j, 1]),
                        'group': 'hallucinated' if replaced[j] else 'correct',
                        'side': 'Q',
                        'prompt_id': h_meta_sel['prompt_id'].iloc[j] if replaced[j]
                                     else p_meta_sel['prompt_id'].iloc[j],
                        'answer_length': float(
                            h_meta_sel['n_pooled'].iloc[j] if replaced[j]
                            else p_meta_sel['n_pooled'].iloc[j]),
                        'repeat': 0,
                    })

    pd.DataFrame(rows_l3d).to_parquet(
        os.path.join(RAW_L3V, 'l3d_pca2_r0.parquet'), index=False)
    pd.DataFrame(rows_l4a).to_parquet(
        os.path.join(RAW_L3V, 'l4a_mixture_pca2_r0.parquet'), index=False)
    pd.DataFrame(rows_direct).to_parquet(
        os.path.join(RAW_L3V, 'l3_direct_pca2_layers_r0.parquet'), index=False)
    progress(f'parquet: l3d {len(rows_l3d)}, l4a {len(rows_l4a)}, direct {len(rows_direct)} строк')
    return {
        'l3d_prompt_hashes': l3d_prompt_hashes,
        'direct_prompt_hashes': direct_prompt_hashes,
        'pool_sizes': pool_sizes,
    }


@section('barcodes')
def section_barcodes(state):
    import pandas as pd
    from scipy.spatial.distance import cdist
    from tda_metrics.barcodes import (
        bar_counts, longest_bar_length, persistence_diagrams, single_linkage_bars,
    )
    from tda_metrics.reduction import pca_transform

    layer = 18
    model = state['models'][layer]
    pools = pools_for_layer(state, layer)
    pool_c, pool_meta_c = pools['correct']
    pool_h, _ = pools['hallucinated']
    sources = {
        'correct': pool_c[_subsample_idx(len(pool_c), 404)],
        'hallucinated': pool_h[_subsample_idx(len(pool_h), 405)],
        'correct2': pool_c[_subsample_idx(len(pool_c), 406)],
    }
    rows = []
    stats = {}
    for name, matrix in sources.items():
        cloud = pca_transform(matrix, model=model)[0]
        h0 = single_linkage_bars(cloud)
        for bar in h0:
            death = float(bar[1])
            rows.append({
                'cloud': name, 'homology_dim': 0, 'birth': 0.0,
                'death': None if not math.isfinite(death) else death,
                'persistence': None if not math.isfinite(death) else death,
                'n_points': N_CLOUD, 'layer': layer, 'repeat': 0,
            })
        dgms = persistence_diagrams(cloud, maxdim=1)
        h1 = np.asarray(dgms[1]).reshape(-1, 2)
        for bar in h1:
            rows.append({
                'cloud': name, 'homology_dim': 1, 'birth': float(bar[0]),
                'death': float(bar[1]), 'persistence': float(bar[1] - bar[0]),
                'n_points': N_CLOUD, 'layer': layer, 'repeat': 0,
            })
        deaths = h0[:-1, 1]
        mean_pd = float(cdist(cloud, cloud).mean())
        stats[name] = {
            'h0_merge_mean': float(deaths.mean()),
            'h0_merge_sd': float(deaths.std(ddof=1)),
            'h0_merge_q25': float(np.quantile(deaths, 0.25)),
            'h0_merge_q75': float(np.quantile(deaths, 0.75)),
            'h1_longest': longest_bar_length(dgms[1]),
            'h1_count': bar_counts(dgms[1], mean_pd),
            'mean_pairwise_dist': mean_pd,
        }
        progress(f"баркоды[{name}]: merge {stats[name]['h0_merge_mean']:.4f}, "
                 f"H1 longest {stats[name]['h1_longest']:.4f}, "
                 f"H1 count {stats[name]['h1_count']}, mean_pd {mean_pd:.3f}")
    frame = pd.DataFrame(rows)
    frame.to_parquet(os.path.join(RAW_L3V, 'l3_barcodes_layer18_r0.parquet'), index=False)

    cross_check = {}
    stats_path = os.path.join(TABLES, 'llm_barcodes_stats.csv')
    if os.path.exists(stats_path):
        old = pd.read_csv(stats_path).set_index('cloud')
        for name, entry in stats.items():
            if name not in old.index:
                continue
            deltas = {}
            for key in ('h0_merge_mean', 'h1_longest', 'mean_pairwise_dist'):
                old_val = float(old.loc[name, key])
                new_val = entry[key]
                deltas[key] = {
                    'old': old_val, 'new': new_val,
                    'rel_diff': abs(new_val - old_val) / (abs(old_val) + 1e-12),
                }
            cross_check[name] = deltas
    return {'stats': stats, 'cross_check': cross_check, 'n_rows': len(rows)}


@section('figures_clouds')
def section_figures_clouds(_):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd

    os.makedirs(FIGURES, exist_ok=True)
    l3d = pd.read_parquet(os.path.join(RAW_L3V, 'l3d_pca2_r0.parquet'))
    direct = pd.read_parquet(os.path.join(RAW_L3V, 'l3_direct_pca2_layers_r0.parquet'))
    mixture = pd.read_parquet(os.path.join(RAW_L3V, 'l4a_mixture_pca2_r0.parquet'))

    def scatter_groups(ax, frame):
        for group in ('correct', 'hallucinated'):
            sub = frame[frame['group'] == group]
            ax.scatter(sub['pc1'], sub['pc2'], s=11, alpha=0.45,
                       color=COLORS[group], label=group, edgecolors='none')

    def panel_axes(ax, layer, note):
        ax.set_xlabel('PC1', fontsize=9)
        ax.set_ylabel('PC2', fontsize=9)
        ax.set_title(f'layer {layer}{note}', fontsize=10)
        ax.tick_params(labelsize=8)

    fig, axes = plt.subplots(1, len(LAYERS_ALL), figsize=(16.5, 4.4))
    for ax, layer in zip(axes, LAYERS_ALL):
        sub = l3d[l3d['layer'] == layer]
        scatter_groups(ax, sub)
        panel_axes(ax, layer, '')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=9, frameon=False)
    fig.suptitle('L3d length-matched (r0): PCA-2 скрытых состояний по слоям — '
                 'иллюстрация; количественные расчёты выполнены в PCA-16; n = 500 на группу',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(os.path.join(FIGURES, 'l3d_length_matched_pca2_layers.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, len(LAYERS_ALL), figsize=(16.5, 4.4))
    for ax, layer in zip(axes, LAYERS_ALL):
        sub = direct[direct['layer'] == layer]
        scatter_groups(ax, sub)
        panel_axes(ax, layer, '')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=9, frameon=False)
    fig.suptitle('L3 direct (r0): PCA-2 по слоям — fit отдельно на dev-correct каждого '
                 'слоя: сравнимо перекрытие внутри слоя, межслойные масштабы не '
                 'интерпретируются; расчёты в PCA-16; n = 500 на группу',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(os.path.join(FIGURES, 'l3_direct_pca2_layers.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(len(LAYERS_L4A), len(ALPHAS_L4A), figsize=(13.5, 11.5))
    for row, layer in enumerate(LAYERS_L4A):
        block = mixture[mixture['layer'] == layer]
        pooled = pd.concat([block[block['alpha'] == a][['pc1', 'pc2']]
                             for a in ALPHAS_L4A])
        xlim = (float(pooled['pc1'].min()), float(pooled['pc1'].max()))
        ylim = (float(pooled['pc2'].min()), float(pooled['pc2'].max()))
        for col, alpha in enumerate(ALPHAS_L4A):
            ax = axes[row, col]
            sub = block[(block['alpha'] == alpha) & (block['side'] == 'Q')]
            scatter_groups(ax, sub)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            panel_axes(ax, layer, f', alpha = {alpha:g}')
            if row == len(LAYERS_L4A) - 1:
                ax.set_xlabel('PC1', fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=9, frameon=False)
    fig.suptitle('L4a вложенная alpha-замена (r0): PCA-2-иллюстрация; расчёты §5.3 '
                 'выполнены в PCA-16; оси общие внутри строки, между строками '
                 'PCA независимы; n = 500 (P и Q_alpha)', fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(os.path.join(FIGURES, 'l4a_mixture_pca2_layers.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    progress('фигуры облаков: 3 шт записаны')


@section('figure_barcodes')
def section_figure_barcodes(_):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    from tda_metrics.barcodes import plot_barcode

    frame = pd.read_parquet(os.path.join(RAW_L3V, 'l3_barcodes_layer18_r0.parquet'))
    h0 = frame[frame['homology_dim'] == 0]
    clouds = ['correct', 'hallucinated', 'correct2']
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2), sharex=True)
    max_death = 0.0
    for _, row in h0.iterrows():
        if row['death'] is not None and not pd.isna(row['death']):
            max_death = max(max_death, float(row['death']))
    for ax, cloud in zip(axes, clouds):
        bars = h0[h0['cloud'] == cloud][['birth', 'death']].to_numpy(dtype=float)
        plot_barcode(ax, bars, level=0, max_bars=60, label=cloud)
        h1_rows = frame[(frame['cloud'] == cloud) & (frame['homology_dim'] == 1)]
        ax.set_title(f"{cloud}: H0-баркод (n=500)", fontsize=10)
        ax.set_xlim(0, max_death * 1.05)
        ax.set_xlabel('расстояние слияния (single-linkage)', fontsize=9)
        n_h1 = len(h1_rows)
        longest_h1 = float(h1_rows['persistence'].max()) if n_h1 else 0.0
        ax.text(0.98, 0.03, f'H1: {n_h1} интервалов, longest {longest_h1:.2f}',
                transform=ax.transAxes, ha='right', fontsize=8, color='gray')
    fig.suptitle('Баркоды layer 18 (r0, exploratory): более короткие H0-связи у '
                 'компактного hallucinated-облака относительно same-law '
                 'контролей — качественная диагностика, не тест гипотезы; '
                 'расчёт в PCA-16', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIGURES, 'l3_barcodes_layer18.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    progress('фигура баркодов записана')


@section('figures_layerwise')
def section_figures_layerwise(_):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd

    llm = pd.read_parquet('results/raw/llm/results.parquet')
    ok = llm[llm['status'] == 'ok']
    families = {'l3_direct': 'direct L3', 'l3d_length_matched': 'length-matched L3d',
                'l3e_shuffled_label': 'shuffled-label L3e'}
    colors = {'l3_direct': 'tab:blue', 'l3d_length_matched': 'tab:orange',
              'l3e_shuffled_label': 'tab:green'}
    metrics_top = ['ntd_PQ', 'ntd_QP']
    metrics_bottom = ['precision@3', 'recall@3']

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    failed_note = []
    for row, metrics in ((0, metrics_top), (1, metrics_bottom)):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            for family, label in families.items():
                block = ok[(ok['experiment_family'] == family)
                           & (ok['layer'].isin(LAYERS_ALL))]
                xs, means, sds = [], [], []
                for layer in LAYERS_ALL:
                    vals = block[(block['layer'] == layer)][metric].dropna()
                    if len(vals) == 0:
                        failed = llm[(llm['experiment_family'] == family)
                                     & (llm['layer'] == layer)
                                     & (llm['status'] == 'failed')]
                        if len(failed) and family not in failed_note:
                            failed_note.append(family)
                        continue
                    xs.append(layer)
                    means.append(float(vals.mean()))
                    sds.append(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
                if xs:
                    ax.errorbar(xs, means, yerr=sds, marker='o', ms=4,
                                color=family and colors[family], label=label, capsize=3)
            ax.set_title(metric, fontsize=10)
            ax.set_xlabel('слой', fontsize=9)
            ax.set_xticks(LAYERS_ALL)
            ax.tick_params(labelsize=8)
    axes[0, 0].legend(fontsize=8)
    note = ('точки — среднее по 5 повторам, усы — SD; '
            + ('пропуск L3e на слое 9: пять прогонов не завершились (не ноль); ' if failed_note else '')
            + 'четыре слоя — грубая сетка глубины, не поиск оптимального слоя')
    fig.suptitle('LLM: послойная динамика контролей L3/L3d/L3e (dev, PCA-16)\n' + note,
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(FIGURES, 'layerwise_llm_dynamics.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    progress('layerwise_llm_dynamics.png записана')

    by_layer = None
    try:
        lw = pd.read_parquet('results/raw/wave5/layerwise.parquet')
        cols = {c.lower(): c for c in lw.columns}
        comp_col = cols.get('comparison') or cols.get('mode')
        metric_rtd = cols.get('rtd') or cols.get('cka_distance')
        by_layer = {}
        for layer in LAYERS_ALL:
            for comp in ('cross', 'identity'):
                block = lw[(lw[cols['layer']] == layer) & (lw[comp_col] == comp)]
                by_layer.setdefault(str(layer), {}).setdefault(comp, {})
                if 'rtd' in lw.columns:
                    by_layer[str(layer)][comp]['rtd'] = {
                        'mean': float(block['rtd'].mean()),
                        'sd': float(block['rtd'].std(ddof=1)),
                    }
                if 'cka_distance' in lw.columns:
                    by_layer[str(layer)][comp]['cka_distance'] = {
                        'mean': float(block['cka_distance'].mean()),
                        'sd': float(block['cka_distance'].std(ddof=1)),
                    }
    except Exception:
        progress('layerwise.parquet не читается стандартно — фолбэк на summary.json')
        with open('results/raw/wave5/layerwise_summary.json', encoding='utf-8') as handle:
            summary = json.load(handle)
        by_layer = summary['by_layer']

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    series = {'cross': 'cross (correct vs hallucinated)', 'identity': 'identity / same-law'}
    palette = {'cross': 'tab:red', 'identity': 'tab:gray'}
    for ax, metric, label in ((axes[0], 'rtd', 'RTD'),
                              (axes[1], 'cka_distance', 'CKA-расстояние')):
        for comp, comp_label in series.items():
            xs, means, sds = [], [], []
            for layer in LAYERS_ALL:
                entry = by_layer[str(layer)][comp][metric]
                xs.append(layer)
                means.append(entry['mean'])
                sds.append(entry.get('sd', 0.0) or 0.0)
            ax.errorbar(xs, means, yerr=sds, marker='o', ms=4,
                        color=palette[comp], label=comp_label, capsize=3)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('слой', fontsize=9)
        ax.set_xticks(LAYERS_ALL)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=8)
    fig.suptitle('Послойные RTD и CKA (dev, n=1000, PCA-16): RTD на независимых '
                 'облаках — exploratory random-coupling; CKA — сходство '
                 'представлений между слоями; отдельные панели из-за разных масштабов',
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(FIGURES, 'layerwise_rtd_cka.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    progress('layerwise_rtd_cka.png записана')


@section('checks')
def section_checks(state):
    import pandas as pd
    from tda_metrics.llm_experiments import length_matched_pair, prompt_ids_hash

    problems = []
    l3d = pd.read_parquet(os.path.join(RAW_L3V, 'l3d_pca2_r0.parquet'))
    direct = pd.read_parquet(os.path.join(RAW_L3V, 'l3_direct_pca2_layers_r0.parquet'))
    mixture = pd.read_parquet(os.path.join(RAW_L3V, 'l4a_mixture_pca2_r0.parquet'))
    bars = pd.read_parquet(os.path.join(RAW_L3V, 'l3_barcodes_layer18_r0.parquet'))

    for frame, name in ((l3d, 'l3d'), (direct, 'direct')):
        if set(frame['layer']) != set(LAYERS_ALL):
            problems.append(f'{name}: слои {sorted(set(frame["layer"]))}')
        if set(frame['group']) != {'correct', 'hallucinated'}:
            problems.append(f'{name}: группы {sorted(set(frame["group"]))}')
        for layer in LAYERS_ALL:
            for group in ('correct', 'hallucinated'):
                count = len(frame[(frame['layer'] == layer) & (frame['group'] == group)])
                if count != N_CLOUD:
                    problems.append(f'{name}@{layer}/{group}: {count} != {N_CLOUD}')
    if set(mixture['layer']) != set(LAYERS_L4A):
        problems.append(f'l4a: слои {sorted(set(mixture["layer"]))}')
    if sorted(set(mixture['alpha'])) != sorted(ALPHAS_L4A):
        problems.append(f'l4a: alpha {sorted(set(mixture["alpha"]))}')
    for layer in LAYERS_L4A:
        kept_prev = None
        for alpha in ALPHAS_L4A:
            block = mixture[(mixture['layer'] == layer) & (mixture['alpha'] == alpha)]
            if len(block) != 2 * N_CLOUD:
                problems.append(f'l4a@{layer}/a{alpha}: {len(block)} != {2 * N_CLOUD}')
            p_rows = block[block['side'] == 'P']
            if len(p_rows) != N_CLOUD or set(p_rows['group']) != {'correct'}:
                problems.append(f'l4a@{layer}/a{alpha}: P-строки некорректны')
            q_kept = set(block[(block['side'] == 'Q')
                               & (block['group'] == 'correct')]['prompt_id'])
            expected_kept = N_CLOUD - int(round(alpha * N_CLOUD))
            if len(q_kept) != expected_kept:
                problems.append(f'l4a@{layer}/a{alpha}: Q-kept {len(q_kept)} != {expected_kept}')
            if kept_prev is not None and not q_kept <= kept_prev:
                problems.append(f'l4a@{layer}/a{alpha}: замены не вложены')
            kept_prev = q_kept

    for layer in LAYERS_ALL:
        c_m, h_m, meta_c, meta_h = length_matched_pair(
            state['meta'], state['vectors'], state['split']['dev'], layer, n_max=N_CLOUD)
        expected_c = prompt_ids_hash(sorted(meta_c['prompt_id']))
        expected_h = prompt_ids_hash(sorted(meta_h['prompt_id']))
        got = state['artifacts']['l3d_prompt_hashes'][layer]
        if got['correct'] != expected_c or got['hallucinated'] != expected_h:
            problems.append(f'l3d@{layer}: prompt-hash не совпадает с пересчётом')

    for cloud in ('correct', 'hallucinated', 'correct2'):
        block = bars[(bars['cloud'] == cloud) & (bars['homology_dim'] == 0)]
        if len(block) != N_CLOUD:
            problems.append(f'баркоды[{cloud}]: H0-баров {len(block)} != {N_CLOUD}')

    if problems:
        raise RuntimeError('CHECKS FAIL:\n' + '\n'.join(problems))
    progress('все программные проверки пройдены')
    return {'checks': 'ok'}


@section('manifest')
def section_manifest(state):
    artifacts = {}
    for name in ('l3d_pca2_r0.parquet', 'l4a_mixture_pca2_r0.parquet',
                 'l3_direct_pca2_layers_r0.parquet', 'l3_barcodes_layer18_r0.parquet'):
        path = os.path.join(RAW_L3V, name)
        artifacts[name] = _sha256_file(path)
    cache_manifest = state['manifest']
    manifest = {
        'created': datetime.now(timezone.utc).isoformat(),
        'sections': SECTION_STATUS,
        'embedding_cache': {
            'model': cache_manifest.get('model'),
            'input_format': cache_manifest.get('input_format'),
            'pooling': cache_manifest.get('pooling'),
            'layer_indices': cache_manifest.get('layer_indices'),
            'n_items': cache_manifest.get('n_items'),
            'dtype': cache_manifest.get('dtype'),
            'manifest_sha256': _sha256_file(os.path.join(CACHE_DIR, 'manifest.json')),
        },
        'split': {
            'rule': 'prompt_id_split(meta, seed=0, 0.6/0.2/0.2)',
            'dev_prompt_ids_hash': state['dev_hash'],
        },
        'pca_rule': ('PCA-16 fit на полном dev-correct пуле каждого слоя '
                     '(reduction.pca_fit); pc1/pc2 — первые две компоненты '
                     'рабочего PCA-16-пространства; hallucinated ответы в fit '
                     'не используются'),
        'seeds': {
            'l3d_r0': 'детерминированный length_matched_pair (wave3-семантика, '
                      'n_max=500, бины по n_pooled)',
            'l3_direct_r0': 'np.random.default_rng(200).choice, позиционные '
                            'индексы в обоих пулах (wave3 _subsample k=0)',
            'l4a_r0': 'облака l3-direct r0 + permutation '
                      'np.random.default_rng(100) (wave3 k=0), вложенная замена '
                      'make_alpha_replacement',
            'barcodes': 'np.random.default_rng(404/405/406).choice, n=500, '
                        'PCA-16 слоя 18 (реплика wave-BC)',
        },
        'l3d_prompt_hashes': state['artifacts']['l3d_prompt_hashes'],
        'direct_prompt_hashes': state['artifacts']['direct_prompt_hashes'],
        'pool_sizes': state['artifacts']['pool_sizes'],
        'barcodes_stats': state['barcodes']['stats'],
        'barcodes_cross_check_vs_llm_barcodes_stats_csv': state['barcodes']['cross_check'],
        'artifacts_sha256': artifacts,
        'runtime_seconds': time.perf_counter() - _STARTED,
    }
    with open(os.path.join(RAW_L3V, 'manifest.json'), 'w', encoding='utf-8') as handle:
        json.dump(json_safe(manifest), handle, ensure_ascii=False, indent=2)
    progress('manifest.json записан')
    return manifest


def main():
    progress('L3V start')
    state = section_load(None)
    if state is None:
        print('L3V DONE rc=1')
        return
    try:
        models, pool_sizes = pca16_models(state)
    except Exception:
        progress('PCA FAIL:\n' + traceback.format_exc(limit=3))
        print('L3V DONE rc=1')
        return
    state['models'] = models
    state['pool_sizes'] = pool_sizes
    artifacts = section_artifacts(state)
    state['artifacts'] = artifacts or {}
    barcodes = section_barcodes(state)
    state['barcodes'] = barcodes or {}
    section_figures_clouds(None)
    section_figure_barcodes(None)
    section_figures_layerwise(None)
    checks = section_checks(state)
    section_manifest(state)
    ok = (SECTION_STATUS.get('artifacts') == 'ok'
          and SECTION_STATUS.get('barcodes') == 'ok'
          and SECTION_STATUS.get('checks') == 'ok')
    print('L3V DONE rc=' + ('0' if ok else '1'))


if __name__ == '__main__':
    os.makedirs('logs', exist_ok=True)
    os.makedirs(RAW_L3V, exist_ok=True)
    sys.stdout = Tee(LOG_PATH)
    main()
