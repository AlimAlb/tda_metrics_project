"""Визуализация динамики C1: облака PCA-эмбеддингов по шагам mode drop / invention.

Иллюстративные фигуры для C1a/C1b (как PRV — качественная демонстрация
механики, не новое статистическое подтверждение): строки = шаги деформации
(0..5 удалённых/добавленных классов), столбцы = пространства
(pixels_pca16, clip_pca16, vae_latent16). Панель — PCA-2-проекция
(fit на P); P — серые точки, Q — цветные по классам MNIST. В заголовке
панели precision@3/recall@3, посчитанные в ПОЛНОМ рабочем пространстве
на тех же облаках (самосогласовано). Raw C1-серия не трогается;
пересчёт заголовков на свежих облаках сверяется с raw rep-0
(pixels/clip детерминированы; VAE переобучен тем же рецептом —
расхождение оговаривается).

Запуск на VM из корня репо: python scripts/build_c1_cloud_visualization.py
"""
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np

RAW = 'results/raw/c1_clouds'
FIGURES = 'results/figures'
LOG_PATH = 'logs/c1_viz_run.log'
CV_RAW = 'results/raw/cv/results.parquet'
MNIST_ROOT = 'data'
VAE_CHECKPOINT = 'embeddings/c1_viz_vae.pt'

SPACES = ('pixels_pca16', 'clip_pca16', 'vae_latent16')
SPACE_TITLES = {
    'pixels_pca16': 'pixels / PCA-16',
    'clip_pca16': 'CLIP / PCA-16',
    'vae_latent16': 'conv-VAE latent-16',
}
N_CLOUD = 1000
SEED_P = 42
SEED_Q = 7
STEPS = (0, 1, 2, 3, 4, 5)

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
    for name in ('torch', 'torchvision', 'transformers', 'pandas', 'matplotlib'):
        try:
            module = __import__(name)
            versions[name] = getattr(module, '__version__', '?')
        except Exception:
            versions[name] = None
    return versions


def series_spec(series):
    """Список (step, classes_P, classes_Q) серии; 0-шаг — same-law базлайн."""
    if series == 'drop':
        return [(nd, list(range(10)), list(range(10 - nd))) for nd in STEPS]
    return [(na, list(range(5)), list(range(5 + na))) for na in STEPS]


@section('env')
def section_env(_):
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    progress(f'device={device}, torch={torch.__version__}, cuda={torch.cuda.is_available()}')
    try:
        import clip
        progress(f'clip пакет OK')
    except Exception:
        progress('clip не установлен — ставлю')
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
        return backbone_embeddings(imgs, clip_model, preprocess, device, desc='c1v-clip')

    os.makedirs('embeddings', exist_ok=True)
    vae = train_vae(images, latent_dim=16, epochs=10, seed=42,
                    device=device, checkpoint=VAE_CHECKPOINT, log_every=2)

    def vae_embedder(imgs):
        return encode_vae(vae, imgs, device=device)

    deps['embedders'] = {'clip': clip_embedder, 'vae': vae_embedder}
    deps['images'], deps['labels'] = images, labels
    progress('embedders готовы: clip / vae')


@section('pools')
def section_pools(deps):
    from tda_metrics.cv_grid import prepare_pools
    pools = prepare_pools(deps['images'], deps['labels'],
                          deps['embedders'], spaces=SPACES)
    for name in SPACES:
        if name not in pools:
            raise RuntimeError(f'пул {name} не построен')
        progress(f'пул {name}: {pools[name][0].shape}')
    deps['pools'] = pools


@section('clouds')
def section_clouds(deps):
    from tda_metrics.cv_grid import class_balanced_sample
    from tda_metrics.metrics import TopologyMetrics

    metrics_obj = TopologyMetrics(seed=0)
    rows = []
    clouds = {}
    for series in ('drop', 'invent'):
        for step, classes_p, classes_q in series_spec(series):
            for space in SPACES:
                pool, labels = deps['pools'][space]
                P, p_classes = class_balanced_sample(
                    pool, labels, classes_p, N_CLOUD, SEED_P, return_class_ids=True)
                Q, q_classes = class_balanced_sample(
                    pool, labels, classes_q, N_CLOUD, SEED_Q, return_class_ids=True)
                pr = metrics_obj.improved_precision_recall(P, Q, nhood_sizes=(3,))
                key = (series, step, space)
                clouds[key] = {'P': P, 'Q': Q, 'p_classes': p_classes,
                               'q_classes': q_classes}
                rows.append({
                    'series': series, 'step': step, 'space': space,
                    'n_P': len(P), 'n_Q': len(Q),
                    'n_classes_Q': len(classes_q),
                    'classes_Q': json.dumps(classes_q),
                    'precision@3': float(pr['precision@3']),
                    'recall@3': float(pr['recall@3']),
                })
                progress(f'{series} step={step} {space}: '
                         f"p@3={pr['precision@3']:.3f} r@3={pr['recall@3']:.3f}")
    deps['clouds'] = clouds
    deps['panel_rows'] = rows


@section('pca2')
def section_pca2(deps):
    from tda_metrics.reduction import pca_fit, pca_transform

    projected = {}
    for series in ('drop', 'invent'):
        for space in SPACES:
            P = deps['clouds'][(series, 0, space)]['P']
            model = pca_fit(P, n_components=2)
            P2, _ = pca_transform(P, model=model, n_components=2)
            projected[(series, space, 'P', 0)] = P2
            for step in STEPS:
                Q = deps['clouds'][(series, step, space)]['Q']
                Q2, _ = pca_transform(Q, model=model, n_components=2)
                projected[(series, space, 'Q', step)] = Q2
            deps.setdefault('pca_models', {})[(series, space)] = model
    deps['projected'] = projected
    progress('PCA-2-проекции готовы (fit на P каждой серии/пространства)')


@section('figures')
def section_figures(deps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(FIGURES, exist_ok=True)
    tab10 = matplotlib.colormaps['tab10']
    for series in ('drop', 'invent'):
        limits = {}
        for space in SPACES:
            pts_all = [deps['projected'][(series, space, 'P', 0)]]
            pts_all += [deps['projected'][(series, space, 'Q', step)]
                        for step in STEPS]
            stacked = np.concatenate(pts_all, axis=0)
            margin_x = 0.05 * (stacked[:, 0].max() - stacked[:, 0].min())
            margin_y = 0.05 * (stacked[:, 1].max() - stacked[:, 1].min())
            limits[space] = (
                (stacked[:, 0].min() - margin_x, stacked[:, 0].max() + margin_x),
                (stacked[:, 1].min() - margin_y, stacked[:, 1].max() + margin_y),
            )
        fig, axes = plt.subplots(len(STEPS), len(SPACES),
                                 figsize=(3.4 * len(SPACES), 3.2 * len(STEPS)))
        for row, step in enumerate(STEPS):
            for col, space in enumerate(SPACES):
                ax = axes[row, col]
                P2 = deps['projected'][(series, space, 'P', 0)]
                Q2 = deps['projected'][(series, space, 'Q', step)]
                clouds_key = (series, step, space)
                q_classes = deps['clouds'][clouds_key]['q_classes']
                ax.scatter(P2[:, 0], P2[:, 1], s=6, c='lightgray', alpha=0.35,
                           linewidths=0)
                for cls in sorted(set(q_classes.tolist())):
                    mask = q_classes == cls
                    ax.scatter(Q2[mask, 0], Q2[mask, 1], s=7, alpha=0.85,
                               linewidths=0, c=tab10(cls))
                panel = next(r for r in deps['panel_rows']
                             if r['series'] == series and r['step'] == step
                             and r['space'] == space)
                tag = 'nd' if series == 'drop' else 'na'
                ax.set_title(
                    f"{SPACE_TITLES[space]}\n{tag}={step} "
                    f"(Q: {panel['n_classes_Q']} кл.) | "
                    f"r@3={panel['recall@3']:.3f} p@3={panel['precision@3']:.3f}",
                    fontsize=8.5)
                ax.set_xticks([])
                ax.set_yticks([])
                (xlim, ylim) = limits[space]
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
        handles = [
            plt.Line2D([0], [0], marker='o', linestyle='', markersize=5,
                       markerfacecolor='lightgray', markeredgecolor='gray',
                       label='P (reference)'),
        ] + [
            plt.Line2D([0], [0], marker='o', linestyle='', markersize=5,
                       markerfacecolor=tab10(cls),
                       markeredgecolor='none', label=f'Q: класс {cls}')
            for cls in range(10)
        ]
        fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=7,
                   frameon=False, bbox_to_anchor=(0.5, 0.0))
        label = 'drop (C1a: Q = классы 0..9-nd)' if series == 'drop' \
            else 'invent (C1b: Q = классы 0..4+na)'
        fig.suptitle(f'C1 cloud dynamics: {label}\n'
                     'PCA-2-проекция (fit на P); метрики в заголовках посчитаны '
                     'в полном рабочем пространстве на этих же облаках',
                     fontsize=10)
        fig.tight_layout(rect=(0, 0.045, 1, 0.96))
        out = os.path.join(FIGURES, f'c1_cloud_dynamics_{series}.png')
        fig.savefig(out, dpi=130)
        plt.close(fig)
        progress(f'фигура: {out}')


@section('export')
def section_export(deps):
    import pandas as pd

    os.makedirs(RAW, exist_ok=True)
    rows = []
    for (series, step, space), entry in deps['clouds'].items():
        P2 = deps['projected'][(series, space, 'P', 0)]
        Q2 = deps['projected'][(series, space, 'Q', step)]
        for side, pts, classes in (
            ('P', P2, entry['p_classes']),
            ('Q', Q2, entry['q_classes']),
        ):
            for i in range(pts.shape[0]):
                rows.append({
                    'series': series, 'step': step, 'space': space,
                    'cloud': side, 'point_index': i,
                    'class_id': int(classes[i]),
                    'x': float(pts[i, 0]), 'y': float(pts[i, 1]),
                })
    points = pd.DataFrame(rows)
    points.to_parquet(os.path.join(RAW, 'points.parquet'), index=False)
    panels = pd.DataFrame(deps['panel_rows'])
    panels.to_csv(os.path.join(RAW, 'panel_metrics.csv'), index=False)
    deps['points_rows'] = len(points)
    progress(f'points.parquet: {len(points)} строк; panel_metrics: {len(panels)}')


@section('crosscheck')
def section_crosscheck(deps):
    if not os.path.exists(CV_RAW):
        progress(f'{CV_RAW} нет — crosscheck пропущен')
        return None
    import pandas as pd

    raw = pd.read_parquet(CV_RAW)
    metric_cols = {}
    for candidate in ('precision@3', 'recall@3'):
        if candidate in raw.columns:
            metric_cols[candidate] = candidate
        elif candidate.replace('@', '_') in raw.columns:
            metric_cols[candidate] = candidate.replace('@', '_')
    if not metric_cols:
        raise RuntimeError(f'нет precision@3/recall@3 в raw cv: {list(raw.columns)}')
    raw = raw[raw['experiment_id'].str.contains('/r0$')]
    comparison = []
    for row in deps['panel_rows']:
        if row['step'] == 0:
            continue
        family = 'c1a' if row['series'] == 'drop' else 'c1b'
        eid = (f"{family}/{row['space']}/"
               f"{'d' if row['series'] == 'drop' else 'a'}{row['step']}/r0")
        match = raw[raw['experiment_id'] == eid]
        if match.empty:
            continue
        entry = {'experiment_id': eid, 'series': row['series'], 'space': row['space']}
        for metric, col in metric_cols.items():
            raw_value = float(match.iloc[0][col])
            entry[metric] = {'raw': raw_value, 'recomputed': row[metric],
                             'abs_diff': abs(raw_value - row[metric])}
        comparison.append(entry)
    max_diff = max((entry['precision@3']['abs_diff'] for entry in comparison),
                  default=None)
    progress(f'crosscheck: {len(comparison)} строк; max |Δprecision@3| = {max_diff}')
    return {'comparison': comparison, 'max_abs_diff_precision@3': max_diff}


@section('summary')
def section_summary(deps):
    crosscheck = deps.get('crosscheck')
    summary = {
        'created': datetime.now(timezone.utc).isoformat(),
        'sections': SECTION_STATUS,
        'spaces': list(SPACES),
        'steps': list(STEPS),
        'n_cloud': N_CLOUD,
        'seeds': {'P': SEED_P, 'Q': SEED_Q},
        'points_rows': deps.get('points_rows'),
        'panel_metrics': deps.get('panel_rows'),
        'crosscheck': crosscheck,
        'honesty_note': (
            'Фигуры — качественная 2D-иллюстрация динамики C1: PCA-2-проекция '
            '(fit на P) служит только для отображения; precision@3/recall@3 в '
            'заголовках посчитаны в полном рабочем пространстве (PCA-16/VAE-16) '
            'на тех же облаках. pixels/clip детерминированы и воспроизводят '
            'raw rep-0; VAE переобучен тем же рецептом (latent 16, 10 эпох, '
            'seed 42) — отдельный экземпляр, оговорено в crosscheck. Существующая '
            'C1-серия не модифицируется и не подменяется.'),
    }
    manifest = {
        'created': summary['created'],
        'versions': _versions(),
        'models': {
            'clip': 'ViT-B/32 (clip.load,jit)',
            'vae': 'conv-VAE-16: latent_dim=16, epochs=10, seed=42, MSE+KL',
        },
        'mnist': 'train, torchvision, root=data',
        'n_cloud': N_CLOUD,
        'seeds': {'P': SEED_P, 'Q': SEED_Q},
        'spaces': list(SPACES),
        'series_spec': {s: [[step, cp, cq] for step, cp, cq in series_spec(s)]
                        for s in ('drop', 'invent')},
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
        print('C1 VIZ DONE rc=1')
        return
    section_data(deps)
    if SECTION_STATUS.get('data') != 'ok':
        print('C1 VIZ DONE rc=1')
        return
    section_pools(deps)
    section_clouds(deps)
    section_pca2(deps)
    section_figures(deps)
    section_export(deps)
    deps['crosscheck'] = section_crosscheck(deps)
    section_summary(deps)
    ok = all(SECTION_STATUS.get(name) == 'ok'
             for name in ('data', 'pools', 'clouds', 'pca2', 'figures', 'export'))
    print('C1 VIZ DONE rc=' + ('0' if ok else '1'))


if __name__ == '__main__':
    os.makedirs('logs', exist_ok=True)
    os.makedirs(RAW, exist_ok=True)
    sys.stdout = Tee(LOG_PATH)
    main()
