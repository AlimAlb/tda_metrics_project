"""Пайплайн экспериментов и корреляционный анализ.

Эксперимент = функция data_fn(step) -> облако P и model_fn(step) -> облако Q,
метрики считаются на каждом шаге через `metrics.compute_all`.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

__all__ = [
    'DISTANCE_METRICS', 'SIMILARITY_METRICS', 'METRIC_GROUPS', 'ALL_METRICS',
    'constant_cloud', 'translated_cloud', 'run_experiment', 'show_steps',
    'plot_clouds', 'plot_trajectories',
    'trajectories_correlation', 'plot_correlation_heatmap', 'group_correlation_summary',
]

DISTANCE_METRICS = ['mtd_PQ', 'mtd_QP', 'rtd', 'mmd', 'frechet', 'js']
SIMILARITY_METRICS = [
    'precision@1', 'precision@3', 'precision@10',
    'recall@1', 'recall@3', 'recall@10',
]

METRIC_GROUPS = {
    'precision/recall': SIMILARITY_METRICS,
    'симметричные': ['rtd', 'mmd', 'frechet', 'js'],
    'асимм. топологические': ['mtd_PQ', 'mtd_QP'],
}

ALL_METRICS = [m for metrics in METRIC_GROUPS.values() for m in metrics]


def constant_cloud(cloud):
    return lambda step: cloud


def translated_cloud(cloud, eps):
    """Сдвиг облака на eps * step вдоль оси x."""
    return lambda step: cloud + np.array([eps * step, 0.0])


def run_experiment(name, data_fn, model_fn, n_steps, metrics_obj, extra=None):
    """Все метрики на каждом шаге; data_fn/model_fn(step) -> облако точек."""
    rows = []
    for step in tqdm(range(n_steps), desc=name):
        row = {'experiment': name, 'step': step}
        if extra is not None:
            row.update(extra)
        row.update(metrics_obj.compute_all(data_fn(step), model_fn(step)))
        rows.append(row)
    return pd.DataFrame(rows)


def show_steps(n_steps, n=5):
    return np.linspace(0, n_steps - 1, n).astype(int)


def plot_clouds(data_fn, model_fn, steps, title=''):
    steps = np.atleast_1d(steps)
    fig, axes = plt.subplots(1, len(steps), figsize=(3.8 * len(steps), 3.8), squeeze=False)
    for ax, step in zip(axes[0], steps):
        P, Q = data_fn(step), model_fn(step)
        ax.scatter(P[:, 0], P[:, 1], s=6, alpha=0.6, label='P (данные)')
        ax.scatter(Q[:, 0], Q[:, 1], s=6, alpha=0.6, label='Q (модель)')
        ax.set_title(f'шаг {step}', fontsize=10)
        ax.set_aspect('equal')
        ax.legend(fontsize=7, loc='upper right')
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_trajectories(df, title=''):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for ax, metric in zip(axes.flat, DISTANCE_METRICS):
        ax.plot(df['step'], df[metric], marker='.', markersize=4)
        ax.set_title(metric)
        ax.set_xlabel('шаг')
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

    precisions = [m for m in SIMILARITY_METRICS if m.startswith('precision')]
    recalls = [m for m in SIMILARITY_METRICS if m.startswith('recall')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for metric in precisions:
        axes[0].plot(df['step'], df[metric], marker='.', markersize=4, label=metric)
    for metric in recalls:
        axes[1].plot(df['step'], df[metric], marker='.', markersize=4, label=metric)
    for ax, label in zip(axes, ('precision', 'recall')):
        ax.set_title(label)
        ax.set_xlabel('шаг')
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def trajectories_correlation(df):
    """Средняя по экспериментам корреляция Спирмена между траекториями метрик.

    PR-метрики инвертируются (1 - x); константные траектории отбрасываются
    (корреляция не определена: RTD в сдвиговых экспериментах, Fréchet
    в экспериментах со смесью).
    """
    inverted = df.copy()
    pr_cols = [c for c in inverted.columns if c.startswith(('precision', 'recall'))]
    inverted[pr_cols] = 1.0 - inverted[pr_cols]

    matrices = []
    for _, part in inverted.groupby('experiment'):
        usable = [m for m in ALL_METRICS if m in part.columns and part[m].nunique() > 1]
        if len(part) < 4 or len(usable) < 2:
            continue
        corr = part[usable].corr(method='spearman').reindex(index=ALL_METRICS, columns=ALL_METRICS)
        matrices.append(corr.values)
    averaged = np.nanmean(np.stack(matrices), axis=0)
    return pd.DataFrame(averaged, index=ALL_METRICS, columns=ALL_METRICS)


def plot_correlation_heatmap(corr, title=''):
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), labels=corr.columns, rotation=90)
    ax.set_yticks(range(len(corr)), labels=corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            if not np.isnan(corr.values[i, j]):
                ax.text(j, i, f'{corr.values[i, j]:.2f}', ha='center', va='center', fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.8)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def group_correlation_summary(corr):
    """Средняя корреляция внутри групп и между группами."""
    rows = []
    for group_a, metrics_a in METRIC_GROUPS.items():
        for group_b, metrics_b in METRIC_GROUPS.items():
            values = [
                corr.loc[a, b]
                for a in metrics_a
                for b in metrics_b
                if a != b and not np.isnan(corr.loc[a, b])
            ]
            rows.append({
                'группа A': group_a,
                'группа B': group_b,
                'средняя корреляция': float(np.mean(values)),
            })
    return pd.DataFrame(rows).pivot(index='группа A', columns='группа B', values='средняя корреляция')