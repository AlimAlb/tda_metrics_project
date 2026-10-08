"""Визуализация C2: облака эмбеддингов для четырёх сравнений цикла вращения.

Иллюстративная фигура в стиле C4 (PCA-2, P серым, Q цветная по углу hsv /
масштабу viridis) для четырёх выборок: cc (круг vs круг — same-law контроль),
arc 0-180, zoom (масштабный свип 0.6..1.4) и arc 180-360 (дополнительная
дуга — вариация, отсутствующая в сетке C2, строится теми же механиками).
Четыре пространства C2 (pixels/CLIP/DINO/VAE). Метрики в заголовках панелей
(rtd, ntd_PQ, precision@3) пересчитаны на показанных облаках по протоколу
C2 (PCA-16 fit на P, rtd_trials=2) и сверяются с raw c2r/.../r0.
Существующая серия C2 не модифицируется.

Запуск на VM из корня репо: python scripts/build_c2_cloud_visualization.py
"""
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np

RAW = 'results/raw/c2_clouds'
FIGURES = 'results/figures'
LOG_PATH = 'logs/c2_viz_run.log'
C2_RAW = 'results/raw/c2/results.parquet'
MNIST_ROOT = 'data'
VAE_CHECKPOINT = 'embeddings/c2_viz_vae.pt'

COMPS = ('cc', 'arc', 'zoom', 'arc360')
COMP_TITLES = {
    'cc': 'контроль: круг vs круг',
    'arc': 'дуга 0–180°',
    'zoom': 'масштабный свип 0.6–1.4',
    'arc360': 'дуга 180–360°',
}
SPACES = ('pixels_pca16', 'clip_pca16', 'dino_pca16', 'vae_latent16')

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
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


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


def _versions():
    versions = {'python': sys.version.split()[0], 'numpy': np.__version__}
    for name in ('torch', 'torchvision', 'pandas', 'matplotlib'):
        try:
            module = __import__(name)
            versions[name] = getattr(module, '__version__', '?')
        except Exception:
            versions[name] = None
    return versions


@section('env')
def section_env(_):
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    progress(f'device={device}, cuda={torch.cuda.is_available()}')
    try:
        import clip
        progress('clip пакет OK')
    except Exception:
        proc = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-q',
             'git+https://github.com/openai/CLIP.git'],
            capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError('pip install clip failed: ' + proc.stderr[-300:])
    return {'device': device}


@section('data')
def section_data(deps):
    import torch
    from tda_metrics.datasets import load_mnist
    from tda_metrics.embeddings import backbone_embeddings
    from tda_metrics.models import encode_vae, train_vae

    device = deps['env']['device']
    images, labels = load_mnist(root=MNIST_ROOT, train=True)
    progress(f'MNIST train: {images.shape}')

    clip_model, preprocess = __import__('clip').load('ViT-B/32', device=device)

    def clip_embedder(imgs):
        return backbone_embeddings(imgs, clip_model, preprocess, device, desc='c2v-clip')

    dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14'
                               ).to(device).eval()

    def dino_embedder(imgs):
        def forward(model, batch):
            import torch as _torch
            with _torch.no_grad():
                return model(batch)
        return backbone_embeddings(imgs, dino_model, preprocess, device,
                                   forward=forward, desc='c2v-dino')

    os.makedirs('embeddings', exist_ok=True)
    vae = train_vae(images, latent_dim=16, epochs=10, seed=42,
                    device=device, checkpoint=VAE_CHECKPOINT, log_every=2)

    def vae_embedder(imgs):
        return encode_vae(vae, imgs, device=device)

    deps['embedders'] = {'clip': clip_embedder, 'dino': dino_embedder,
                         'vae': vae_embedder}
    deps['images'], deps['labels'] = images, labels
    progress('embedders готовы: clip / dino / vae')


@section('clouds')
def section_clouds(deps):
    from tda_metrics.c2_grid import (
        ANGLES_ARC180, ANGLES_CIRCLE, SCALES_ZOOM, cloud_images, seed_p, seed_q,
    )
    from tda_metrics.datasets import select_class, sweep_images, zoom_images

    images, labels = deps['images'], deps['labels']
    clouds = {'P': cloud_images(images, labels, seed_p(0), angles=ANGLES_CIRCLE)}
    clouds['cc'] = cloud_images(images, labels, seed_q(0), angles=ANGLES_CIRCLE)
    clouds['arc'] = cloud_images(images, labels, seed_q(0), angles=ANGLES_ARC180)
    clouds['arc360'] = cloud_images(
        images, labels, seed_q(0), angles=ANGLES_ARC180 + 180.0)
    base = select_class(images, labels, 3, 25, seed_q(0))
    clouds['zoom'] = sweep_images(base, SCALES_ZOOM, zoom_images)
    for key, cloud in clouds.items():
        progress(f'облако {key}: {cloud.shape}')
    deps['clouds'] = clouds
    deps['q_for_comp'] = {'cc': 'cc', 'arc': 'arc', 'zoom': 'zoom',
                          'arc360': 'arc360'}


@section('embed')
def section_embed(deps):
    kinds = {'pixels': None, 'clip': deps['embedders']['clip'],
             'dino': deps['embedders']['dino'], 'vae': deps['embedders']['vae']}
    raw = {}
    for kind, embedder in kinds.items():
        for cloud_id, cloud in deps['clouds'].items():
            if kind == 'pixels':
                matrix = cloud.reshape(len(cloud), -1).astype(np.float64)
            else:
                matrix = np.asarray(embedder(cloud), dtype=np.float64)
            raw[(kind, cloud_id)] = matrix
            progress(f'raw {kind}/{cloud_id}: {matrix.shape}')
    deps['raw'] = raw


@section('metrics')
def section_metrics(deps):
    from tda_metrics.metrics import TopologyMetrics

    metrics_obj = TopologyMetrics(seed=42, device=deps['env']['device'])
    raw_cache = deps['raw']
    kind_of_space = {'pixels_pca16': 'pixels', 'clip_pca16': 'clip',
                     'dino_pca16': 'dino', 'vae_latent16': 'vae'}

    rows = []
    projected16 = {}
    for space in SPACES:
        kind = kind_of_space[space]
        p_raw = raw_cache[(kind, 'P')]
        for comp in COMPS:
            q_raw = raw_cache[(kind, deps['q_for_comp'][comp])]
            p16, q16 = project_pair_with_raw(space, p_raw, q_raw)
            projected16[(space, comp)] = (p16, q16)
            result = metrics_obj.compute_all(
                p16, q16, nhood_sizes=(1, 3, 10), js_k=5,
                rtd_trials=2, rtd_batch=500)
            rows.append({
                'space': space, 'comp': comp,
                'rtd': float(result['rtd']),
                'ntd_PQ': float(result['ntd_PQ']),
                'ntd_QP': float(result['ntd_QP']),
                'mtd_PQ': float(result['mtd_PQ']),
                'mtd_QP': float(result['mtd_QP']),
                'precision@3': float(result['precision@3']),
                'recall@3': float(result['recall@3']),
                'mmd': float(result['mmd']),
                'frechet': float(result['frechet']),
                'js': float(result['js']),
            })
            progress(f'{space}/{comp}: rtd={result["rtd"]:.2f} '
                     f'ntd_PQ={result["ntd_PQ"]:.3f} '
                     f'p@3={result["precision@3"]:.3f}')
    deps['panel_rows'] = rows
    deps['projected16'] = projected16


def project_pair_with_raw(space, p_raw, q_raw):
    from tda_metrics.c2_grid import C2_SPACES
    from tda_metrics.reduction import pca_fit, pca_transform

    info = C2_SPACES[space]
    p_raw = np.asarray(p_raw, dtype=np.float64)
    q_raw = np.asarray(q_raw, dtype=np.float64)
    if info['pca'] is None:
        return p_raw, q_raw
    model = pca_fit(p_raw, n_components=16)
    p16, _ = pca_transform(p_raw, model=model)
    q16, _ = pca_transform(q_raw, model=model)
    return p16, q16


@section('pca2')
def section_pca2(deps):
    from tda_metrics.reduction import pca_fit, pca_transform

    projected = {}
    for space in SPACES:
        kind = {'pixels_pca16': 'pixels', 'clip_pca16': 'clip',
                'dino_pca16': 'dino', 'vae_latent16': 'vae'}[space]
        stacked = [deps['raw'][(kind, 'P')]]
        stacked += [deps['raw'][(kind, deps['q_for_comp'][comp])]
                    for comp in COMPS]
        model = pca_fit(np.vstack(stacked), n_components=2)
        p2, _ = pca_transform(deps['raw'][(kind, 'P')], model=model)
        projected[(space, 'P')] = p2
        for comp in COMPS:
            q2, _ = pca_transform(
                deps['raw'][(kind, deps['q_for_comp'][comp])], model=model)
            projected[(space, comp)] = q2
    deps['projected2'] = projected
    progress('PCA-2-проекции готовы (fit на объединении P и всех Q)')


@section('figures')
def section_figures(deps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tda_metrics.c2_grid import ANGLES_ARC180, ANGLES_CIRCLE, SCALES_ZOOM

    os.makedirs(FIGURES, exist_ok=True)
    angle_by_comp = {
        'cc': np.tile(ANGLES_CIRCLE, 25),
        'arc': np.tile(ANGLES_ARC180, 25),
        'arc360': np.tile(ANGLES_ARC180 + 180.0, 25),
        'zoom': np.tile(SCALES_ZOOM, 25),
    }
    fig, axes = plt.subplots(len(COMPS), len(SPACES),
                             figsize=(3.4 * len(SPACES), 3.4 * len(COMPS)))
    for row, comp in enumerate(COMPS):
        scatter_last = None
        for col, space in enumerate(SPACES):
            ax = axes[row, col]
            p2 = deps['projected2'][(space, 'P')]
            q2 = deps['projected2'][(space, comp)]
            ax.scatter(p2[:, 0], p2[:, 1], s=8, c='lightgray', alpha=0.4,
                       linewidths=0)
            values = angle_by_comp[comp]
            cmap = 'hsv' if comp != 'zoom' else 'viridis'
            vmin, vmax = (0.0, 360.0) if comp != 'zoom' else (
                float(SCALES_ZOOM.min()), float(SCALES_ZOOM.max()))
            scatter_last = ax.scatter(q2[:, 0], q2[:, 1], c=values, cmap=cmap,
                                      s=9, alpha=0.85, linewidths=0,
                                      vmin=vmin, vmax=vmax)
            panel = next(r for r in deps['panel_rows']
                         if r['space'] == space and r['comp'] == comp)
            ax.set_title(
                f"{COMP_TITLES[comp]} | {space}\n"
                f"rtd={panel['rtd']:.2f} ntd_PQ={panel['ntd_PQ']:.3f} "
                f"p@3={panel['precision@3']:.3f}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        label = 'угол, градусы' if comp != 'zoom' else 'масштаб'
        fig.colorbar(scatter_last, ax=list(axes[row, :]), label=label,
                     shrink=0.85)
    handles = [plt.Line2D([0], [0], marker='o', linestyle='', markersize=5,
                          markerfacecolor='lightgray', markeredgecolor='gray',
                          label='P: полный круг (reference)')]
    fig.legend(handles=handles, loc='upper center', ncol=1, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle('C2: облака эмбеддингов цикла вращения — контроль cc, дуга 0–180°, '
                 'масштабный свип, дуга 180–360°\n'
                 'PCA-2 (fit на P+Q); метрики в заголовках посчитаны по протоколу C2 '
                 '(PCA-16 fit на P, rtd_trials=2) на этих же облаках',
                 y=1.035, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = os.path.join(FIGURES, 'c2_cloud_dynamics.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    progress(f'фигура: {out}')


@section('export')
def section_export(deps):
    import pandas as pd
    from tda_metrics.c2_grid import ANGLES_ARC180, ANGLES_CIRCLE, SCALES_ZOOM

    os.makedirs(RAW, exist_ok=True)
    angle_by_comp = {
        'cc': np.tile(ANGLES_CIRCLE, 25),
        'arc': np.tile(ANGLES_ARC180, 25),
        'arc360': np.tile(ANGLES_ARC180 + 180.0, 25),
        'zoom': np.tile(SCALES_ZOOM, 25),
    }
    rows = []
    for space in SPACES:
        p2 = deps['projected2'][(space, 'P')]
        for j in range(len(p2)):
            rows.append({'space': space, 'comp': 'P', 'point_id': j,
                         'value': float(np.nan), 'pc1': float(p2[j, 0]),
                         'pc2': float(p2[j, 1])})
        for comp in COMPS:
            q2 = deps['projected2'][(space, comp)]
            values = angle_by_comp[comp]
            for j in range(len(q2)):
                rows.append({'space': space, 'comp': comp, 'point_id': j,
                             'value': float(values[j]), 'pc1': float(q2[j, 0]),
                             'pc2': float(q2[j, 1])})
    points = pd.DataFrame(rows)
    points.to_parquet(os.path.join(RAW, 'points.parquet'), index=False)
    panels = pd.DataFrame(deps['panel_rows'])
    panels.to_csv(os.path.join(RAW, 'panel_metrics.csv'), index=False)
    deps['points_rows'] = len(points)
    progress(f'points.parquet: {len(points)} строк; panel_metrics: {len(panels)}')


@section('crosscheck')
def section_crosscheck(deps):
    if not os.path.exists(C2_RAW):
        progress(f'{C2_RAW} нет — crosscheck пропущен')
        return None
    import pandas as pd

    raw = pd.read_parquet(C2_RAW)
    raw = raw[raw['experiment_id'].str.contains('/r0$')]
    comparison = []
    for row in deps['panel_rows']:
        if row['comp'] == 'arc360':
            continue
        eid = f"c2r/{row['space']}/{row['comp']}/r0"
        match = raw[raw['experiment_id'] == eid]
        if match.empty or 'rtd' not in match.columns:
            continue
        entry = {'experiment_id': eid, 'space': row['space'], 'comp': row['comp']}
        for metric in ('rtd', 'ntd_PQ', 'precision@3'):
            if metric in match.columns:
                raw_value = float(match.iloc[0][metric])
                entry[metric] = {'raw': raw_value, 'recomputed': row[metric],
                                 'abs_diff': abs(raw_value - row[metric])}
        comparison.append(entry)
    max_diff = max((e['rtd']['abs_diff'] for e in comparison if 'rtd' in e),
                   default=None)
    progress(f'crosscheck: {len(comparison)} строк; max |Δrtd| = {max_diff}')
    return {'comparison': comparison, 'max_abs_diff_rtd': max_diff}


@section('summary')
def section_summary(deps):
    crosscheck = deps.get('crosscheck')
    summary = {
        'created': datetime.now(timezone.utc).isoformat(),
        'sections': SECTION_STATUS,
        'spaces': list(SPACES),
        'comps': list(COMPS),
        'points_rows': deps.get('points_rows'),
        'panel_metrics': deps.get('panel_rows'),
        'crosscheck': crosscheck,
        'honesty_note': (
            'Фигура — качественная иллюстрация в стиле C4: PCA-2 (fit на '
            'объединении P и всех Q) служит только для отображения; метрики '
            'заголовков посчитаны по протоколу C2 (PCA-16 fit на P, '
            'rtd_trials=2) на тех же облаках. cc/arc/zoom воспроизводят '
            'raw c2r/.../r0 (pixels/clip/dino детерминированы; VAE '
            'переобучен тем же рецептом — расхождение оговорено). '
            'arc 180–360° — дополнительная дуга, отсутствующая в сетке C2 '
            '(скан строил арки [0, m)); строится теми же механиками при '
            'равном |Q|, значение метрик — новое, не из raw. Существующая '
            'серия C2 не модифицируется.'),
    }
    manifest = {
        'created': summary['created'],
        'versions': _versions(),
        'models': {
            'clip': 'ViT-B/32 (clip.load, jit)',
            'dino': 'dinov2_vits14 (torch.hub)',
            'vae': 'conv-VAE-16: latent_dim=16, epochs=10, seed=42',
        },
        'metric_params': {'seed': 42, 'rtd_trials': 2, 'rtd_batch': 500,
                          'js_k': 5, 'nhood_sizes': [1, 3, 10],
                          'pca_dim': 16},
        'clouds': {'P': 'полный круг, seed 42 (25 цифр × 18 углов, шаг 20°)',
                   'cc': 'полный круг, seed 7',
                   'arc': 'дуга 0–180° (шаг 10°), seed 7',
                   'zoom': 'масштабный свип 0.6–1.4 (18 значений), seed 7',
                   'arc360': 'дуга 180–360° (шаг 10°), seed 7 — вариация вне сетки C2'},
        'sections': SECTION_STATUS,
        'runtime_seconds': time.perf_counter() - _STARTED,
    }
    with open(os.path.join(RAW, 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(json_safe(summary), handle, ensure_ascii=False, indent=2)
    with open(os.path.join(RAW, 'manifest.json'), 'w', encoding='utf-8') as handle:
        json.dump(json_safe(manifest), handle, ensure_ascii=False, indent=2)
    progress('summary/manifest записаны')


def main():
    deps = {}
    deps['env'] = section_env(None)
    if deps['env'] is None:
        print('C2 VIZ DONE rc=1')
        return
    section_data(deps)
    if SECTION_STATUS.get('data') != 'ok':
        print('C2 VIZ DONE rc=1')
        return
    section_clouds(deps)
    section_embed(deps)
    section_metrics(deps)
    section_pca2(deps)
    section_figures(deps)
    section_export(deps)
    deps['crosscheck'] = section_crosscheck(deps)
    section_summary(deps)
    ok = all(SECTION_STATUS.get(name) == 'ok'
             for name in ('data', 'clouds', 'embed', 'metrics', 'pca2',
                          'figures', 'export'))
    print('C2 VIZ DONE rc=' + ('0' if ok else '1'))


if __name__ == '__main__':
    os.makedirs('logs', exist_ok=True)
    os.makedirs(RAW, exist_ok=True)
    sys.stdout = Tee(LOG_PATH)
    main()
