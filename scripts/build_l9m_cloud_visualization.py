"""L9M: визуализация динамики облаков при росте доли галлюцинаций (wave L9M).

Запускается на Colab VM ПОСЛЕ основного прогона run_l9m_chain.py (кэш
embeddings/l9m_cache и results/raw/l9m/rows.jsonl должны существовать).
Один dev-базис (repeat=0, сиды протокола) строится детерминированно той же
механикой: PCA-2 (fit на P) проекции length-matched облаков при
alpha in {0.0, 0.25, 0.5, 1.0}; P — серый (#4d4d4d), Q-correct —
таб:cyan, Q-hallucinated — таб:red. Заголовки панелей: метрики пакета
(recall@3, precision@3, ntd_PQ, mtd_PQ, q_len_mean) из rows.jsonl
основного прогона + кросс-чек пересчётом (метрики заголовков — из
рабочего пространства PCA-16, облака — только проекция). Компоновка 1x4.
Артефакты: results/figures/l9m_cloud_dynamics.png +
results/raw/l9m_clouds/{points.parquet, panel_metrics.csv, summary.json,
manifest.json}. SystemExit не поднимается.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone

RAW = 'results/raw/l9m'
CLOUDS = 'results/raw/l9m_clouds'
FIGURES = 'results/figures'
ROWS_PATH = os.path.join(RAW, 'rows.jsonl')
CACHE_DIR = 'embeddings/l9m_cache'

LAYER = 18
PCA_DIM = 16
N_CLOUD = 500
N_BINS = None  # берётся из manifest основного прогона (locked)
REPEAT = 0
ALPHAS = [0.0, 0.25, 0.5, 1.0]
SEEDS = (100000 + REPEAT, 200000 + REPEAT, 300000 + REPEAT)
METRIC_KEYS = ['recall@3', 'precision@3', 'ntd_PQ', 'mtd_PQ']

SECTION_STATUS = {}
_STARTED = time.perf_counter()


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


@section('env')
def section_env(_):
    if not os.path.exists(ROWS_PATH):
        raise RuntimeError(f'{ROWS_PATH} отсутствует — сначала основной прогон')
    lock = json.load(open(os.path.join(RAW, 'manifest.json')))['grids']
    globals()['N_BINS'] = int(lock['n_bins_locked'])
    globals()['N_CLOUD'] = int(lock['n_cloud'])
    progress(f'лок из manifest: n_bins={N_BINS}, n_cloud={N_CLOUD}')
    return {'lock': lock}


def load_rows_packets():
    import pandas as pd
    rows = []
    with open(ROWS_PATH, encoding='utf-8') as handle:
        for line in handle:
            rec = json.loads(line)
            if rec.get('kind') == 'packet' and rec.get('split') == 'dev' \
                    and rec.get('repeat') == REPEAT and rec.get('status') == 'ok':
                rows.append(rec)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError('dev-пакеты repeat=0 не найдены в rows.jsonl')
    return frame.set_index('alpha')


def build_panels(deps):
    import numpy as np
    from tda_metrics.embedding_cache import load_cache
    from tda_metrics.llm_experiments import prompt_id_split
    from tda_metrics.reduction import pca_fit, pca_transform
    from tda_metrics.l9m_pools import sample_length_matched_pools, build_length_matched_packet

    meta, vectors, manifest = load_cache(
        CACHE_DIR,
        expect={'model': 'Qwen/Qwen2.5-3B-Instruct',
                'input_format': 'chat_knowledge_v1',
                'pooling': 'mean_answer'})
    pos = manifest['layer_indices'].index(LAYER)
    meta_l = meta[meta['layer_index'] == LAYER].reset_index(drop=True)
    vectors_l = vectors[:, pos, :]
    splits = prompt_id_split(meta_l, seed=0)

    dev_mask = (meta_l['prompt_id'].isin(set(splits['dev']))
                & (meta_l['answer_kind'] == 'correct')).to_numpy()
    model16 = pca_fit(vectors_l[dev_mask], n_components=PCA_DIM)
    x16, _ = pca_transform(vectors_l, model=model16)
    x16 = np.asarray(x16, dtype=float)

    allowed = set(splits['dev'])
    prompt_np = meta_l['prompt_id'].to_numpy()
    kind_np = meta_l['answer_kind'].to_numpy()
    len_np = meta_l['n_pooled'].to_numpy()
    in_dev = np.isin(prompt_np, list(allowed))
    rows_c = np.flatnonzero(in_dev & (kind_np == 'correct'))
    rows_h = np.flatnonzero(in_dev & (kind_np == 'hallucinated'))
    pools = sample_length_matched_pools(
        x16[rows_c], rows_c, len_np[rows_c], prompt_np[rows_c],
        x16[rows_h], rows_h, len_np[rows_h], prompt_np[rows_h],
        n_cloud=N_CLOUD, seed_half=SEEDS[0], seed_sample=SEEDS[1],
        n_bins=N_BINS, bin_ladder=())
    if pools['n_cloud_actual'] != N_CLOUD:
        raise RuntimeError(
            f'n_cloud_actual {pools["n_cloud_actual"]} != {N_CLOUD} — сиды/лок разошлись с прогоном')

    pca2 = pca_fit(pools['P'], n_components=2)
    p2, _ = pca_transform(pools['P'], model=pca2)
    q2_correct, _ = pca_transform(pools['Q_correct'], model=pca2)
    h2, _ = pca_transform(pools['H'], model=pca2)

    import pandas as pd
    point_rows = []
    for index in range(N_CLOUD):
        point_rows.append({'alpha': None, 'x': float(p2[index, 0]),
                          'y': float(p2[index, 1]), 'side': 'P',
                          'n_pooled': int(pools['p_lengths'][index])})
    for alpha in ALPHAS:
        packet = build_length_matched_packet(pools, alpha, SEEDS[2])
        q2, _ = pca_transform(packet['Q'], model=pca2)
        for index in range(N_CLOUD):
            kind = packet['Q_items'][index][0]
            point_rows.append({
                'alpha': alpha, 'x': float(q2[index, 0]), 'y': float(q2[index, 1]),
                'side': 'Q_hallucinated' if kind == 'hallucinated' else 'Q_correct',
                'n_pooled': int(packet['q_lengths'][index])})
        deps.setdefault('packets', {})[alpha] = (packet, pools)
    points = pd.DataFrame(point_rows)
    progress(f'points: {len(points)} строк, панелей alpha: {len(ALPHAS)}')
    return {'points': points, 'pools': pools, 'meta_l': meta_l, 'x16': x16,
            'packets': deps.get('packets'), 'splits': splits}


@section('crosscheck')
def section_crosscheck(deps):
    import numpy as np
    from tda_metrics.metrics import TopologyMetrics
    packets = deps['panels']['packets']
    rows_frame = deps['rows']
    metrics_obj = TopologyMetrics(seed=0)
    rows = []
    for alpha in ALPHAS:
        packet, _ = packets[alpha]
        result = metrics_obj.compute_all(
            np.asarray(packet['P'], dtype=float),
            np.asarray(packet['Q'], dtype=float), skip=('rtd',))
        stored = rows_frame.loc[alpha]
        entry = {'alpha': alpha,
                 'n_hall': int(packet['n_hall']),
                 'q_len_mean': float(np.mean(packet['q_lengths']))}
        deltas = {}
        for key in METRIC_KEYS:
            recomputed = float(result[key])
            entry[key] = recomputed
            entry[f'{key}_stored'] = float(stored[key])
            deltas[key] = abs(recomputed - float(stored[key]))
        entry['max_delta'] = max(deltas.values())
        rows.append(entry)
        progress(f"alpha={alpha}: r@3={entry['recall@3']:.3f} "
                 f"p@3={entry['precision@3']:.3f} max|Δ|={entry['max_delta']:.4f}")
    import pandas as pd
    frame = pd.DataFrame(rows)
    worst = float(frame['max_delta'].max())
    if worst > 0.01:
        raise RuntimeError(f'кросс-чек провален: max|Δ|={worst} > 0.01')
    progress(f'кросс-чек ok: max|Δ|={worst}')
    return frame


@section('window')
def section_window(deps):
    import numpy as np
    panels = deps['panels']
    pools = panels['pools']
    meta_l = panels['meta_l']
    len_np = meta_l['n_pooled'].to_numpy()
    kind_np = meta_l['answer_kind'].to_numpy()
    prompt_np = meta_l['prompt_id'].to_numpy()
    dev = set(panels['splits']['dev'])
    in_dev = np.isin(prompt_np, list(dev))
    full_correct = len_np[in_dev & (kind_np == 'correct')]
    full_hall = len_np[in_dev & (kind_np == 'hallucinated')]
    edges = np.asarray(pools['bin_edges'], dtype=float)
    finite = edges[np.isfinite(edges)]
    quotas = np.asarray(pools['bin_quotas'], dtype=int)
    active = quotas > 0
    selected_p = pools['p_lengths']
    selected_h = pools['h_lengths']
    lo = float(min(selected_p.min(), selected_h.min()))
    hi = float(max(selected_p.max(), selected_h.max()))
    window = {
        'n_bins_locked': int(pools['n_bins_actual']),
        'bin_edges_finite': [float(v) for v in finite],
        'bin_quotas': [int(v) for v in quotas],
        'active_bins': int(active.sum()),
        'window_lo': lo,
        'window_hi': hi,
        'coverage_correct_in_window': float(np.mean(
            (full_correct >= lo) & (full_correct <= hi))),
        'coverage_hallucinated_in_window': float(np.mean(
            (full_hall >= lo) & (full_hall <= hi))),
        'full_correct_mean': float(full_correct.mean()),
        'full_hall_mean': float(full_hall.mean()),
        'matched_p_mean': float(selected_p.mean()),
        'matched_h_mean': float(selected_h.mean()),
        'note': 'окно length-matching: доля full dev-пулов correct/hallucinated '
                'с длинами внутри диапазона фактически отобранных строк; '
                'выводы L9M валидны внутри этого окна',
    }
    progress(f"окно [{lo:.0f}, {hi:.0f}]: покрытие correct "
             f"{window['coverage_correct_in_window']:.1%}, hallucinated "
             f"{window['coverage_hallucinated_in_window']:.1%}")
    return window


@section('length_encoding')
def section_length_encoding(deps):
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    panels = deps['panels']
    meta_l = panels['meta_l']
    x16 = panels['x16']
    prompt_np = meta_l['prompt_id'].to_numpy()
    kind_np = meta_l['answer_kind'].to_numpy()
    len_np = meta_l['n_pooled'].to_numpy()
    dev = set(panels['splits']['dev'])
    in_dev = np.isin(prompt_np, list(dev))
    out = {}
    for kind in ('correct', 'hallucinated'):
        mask = in_dev & (kind_np == kind)
        X = np.asarray(x16[mask], dtype=float)
        y = len_np[mask].astype(float)
        norm = np.linalg.norm(X, axis=1)
        pc1 = X[:, 0]
        pred = cross_val_predict(
            make_pipeline(StandardScaler(), Ridge(alpha=1.0)), X, y,
            cv=KFold(n_splits=5, shuffle=True, random_state=0))
        out[kind] = {
            'n_items': int(mask.sum()),
            'spearman_len_norm': float(spearmanr(y, norm).statistic),
            'spearman_len_pc1': float(spearmanr(y, pc1).statistic),
            'ridge_r2_len_from_vectors': float(r2_score(y, pred)),
            'len_mean': float(y.mean()), 'len_std': float(y.std(ddof=0)),
        }
        progress(f"length_encoding[{kind}]: ρ(len,‖v‖)={out[kind]['spearman_len_norm']:.3f}, "
                 f"ρ(len,PC1)={out[kind]['spearman_len_pc1']:.3f}, "
                 f"ridge R²(len|16d)={out[kind]['ridge_r2_len_from_vectors']:.3f}")
    return out


@section('figures')
def section_figures(deps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(FIGURES, exist_ok=True)
    points = deps['panels']['points']
    metrics = deps['crosscheck'].set_index('alpha')
    figure, axes = plt.subplots(1, len(ALPHAS), figsize=(4.4 * len(ALPHAS), 4.4),
                                 sharex=True, sharey=True)
    if len(ALPHAS) == 1:
        axes = [axes]
    p_points = points[points['side'] == 'P']
    for axis, alpha in zip(axes, ALPHAS):
        axis.scatter(p_points['x'], p_points['y'], s=10, alpha=0.55,
                     color='#4d4d4d', label='P (correct)', rasterized=True)
        block = points[(points['alpha'] == alpha)]
        for side, color, marker in (
                ('Q_correct', 'tab:cyan', 'o'),
                ('Q_hallucinated', 'tab:red', 'o')):
            sub = block[block['side'] == side]
            if len(sub):
                axis.scatter(sub['x'], sub['y'], s=10, alpha=0.7, color=color,
                             marker=marker, rasterized=True,
                             label=side.replace('Q_', ''))
        entry = metrics.loc[alpha]
        axis.set_title(
            f"α={alpha}\nr@3={entry['recall@3']:.3f}  p@3={entry['precision@3']:.3f}\n"
            f"ntd_PQ={entry['ntd_PQ']:.2f}  mtd_PQ={entry['mtd_PQ']:.1f}\n"
            f"q_len={entry['q_len_mean']:.1f}", fontsize=9)
        axis.set_xlabel('PCA-2a', fontsize=8)
        axis.tick_params(labelsize=7)
    axes[0].set_ylabel('PCA-2b', fontsize=8)
    axes[0].legend(fontsize=7, loc='upper right')
    figure.suptitle('L9M: динамика length-matched облака Q при росте доли '
                    'галлюцинаций (dev, repeat=0; PCA-2 на P; заголовки — метрики '
                    'PCA-16 из основного прогона)')
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURES, 'l9m_cloud_dynamics.png'), dpi=150)
    plt.close(figure)
    progress('фигура записана')


@section('summary')
def section_summary(deps):
    os.makedirs(CLOUDS, exist_ok=True)
    deps['panels']['points'].to_parquet(os.path.join(CLOUDS, 'points.parquet'))
    deps['crosscheck'].to_csv(os.path.join(CLOUDS, 'panel_metrics.csv'), index=False)
    lock = json.load(open(os.path.join(RAW, 'manifest.json')))
    summary = {
        'created': datetime.now(timezone.utc).isoformat(),
        'sections': SECTION_STATUS,
        'repeat': REPEAT, 'alphas': ALPHAS, 'n_cloud': N_CLOUD,
        'n_bins_locked': N_BINS, 'seeds': list(SEEDS),
        'length_window': deps.get('window'),
        'length_encoding': deps.get('length_encoding'),
        'main_run_manifest': {
            'protocol_sha256': lock.get('protocol_sha256'),
            'grids': lock.get('grids')},
        'runtime_seconds': time.perf_counter() - _STARTED,
    }
    with open(os.path.join(CLOUDS, 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(json_safe(summary), handle, ensure_ascii=False, indent=2)
    manifest = {
        'created': summary['created'],
        'main_protocol_sha256': lock.get('protocol_sha256'),
        'rows_sha256': _sha256_file(ROWS_PATH),
        'sections': SECTION_STATUS,
    }
    with open(os.path.join(CLOUDS, 'manifest.json'), 'w', encoding='utf-8') as handle:
        json.dump(json_safe(manifest), handle, ensure_ascii=False, indent=2)
    progress('summary/manifest записаны')


def main():
    deps = {}
    progress('L9M-viz start')
    section_env(None)
    deps['rows'] = load_rows_packets()
    deps['panels'] = build_panels(deps)
    deps['window'] = section_window(deps)
    deps['length_encoding'] = section_length_encoding(deps)
    crosscheck = section_crosscheck(deps)
    if crosscheck is None:
        print('L9M-VIZ DONE rc=1')
        return
    section_figures(deps)
    section_summary(deps)
    ok = all(SECTION_STATUS.get(name) == 'ok' for name in
             ('env', 'window', 'length_encoding', 'crosscheck', 'figures', 'summary'))
    print('L9M-VIZ DONE rc=' + ('0' if ok else '1'))


if __name__ == '__main__':
    _original_stdout = sys.stdout
    try:
        main()
    except Exception:
        progress('VIZ FAIL:\n' + traceback.format_exc(limit=3))
        print('L9M-VIZ DONE rc=1')
    finally:
        sys.stdout = _original_stdout
        sys.stdout.flush()
