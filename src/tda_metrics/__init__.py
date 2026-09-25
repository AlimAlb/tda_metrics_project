"""Топологические и статистические метрики схожести распределений.

Модули:
    samplers    — сэмплеры облаков точек для 2D-экспериментов;
    metrics     — класс TopologyMetrics (MTD, RTD, improved PR, MMD, Fréchet, JS);
    experiments — пайплайн экспериментов и корреляционный анализ;
    embeddings  — CLIP-эмбеддинги (тяжелая секция, импортируется явно).
"""
from tda_metrics.samplers import (
    sample_gaussian,
    sample_uniform_cube,
    sample_ball,
    sample_ring,
    sample_gaussian_mixture,
)
from tda_metrics.metrics import TopologyMetrics, np_random_seed
from tda_metrics.experiments import (
    DISTANCE_METRICS,
    SIMILARITY_METRICS,
    METRIC_GROUPS,
    ALL_METRICS,
    constant_cloud,
    translated_cloud,
    run_experiment,
    show_steps,
    plot_clouds,
    plot_trajectories,
    trajectories_correlation,
    plot_correlation_heatmap,
    group_correlation_summary,
)

__all__ = [
    'sample_gaussian', 'sample_uniform_cube', 'sample_ball', 'sample_ring',
    'sample_gaussian_mixture',
    'TopologyMetrics', 'np_random_seed',
    'DISTANCE_METRICS', 'SIMILARITY_METRICS', 'METRIC_GROUPS', 'ALL_METRICS',
    'constant_cloud', 'translated_cloud', 'run_experiment', 'show_steps',
    'plot_clouds', 'plot_trajectories',
    'trajectories_correlation', 'plot_correlation_heatmap', 'group_correlation_summary',
]