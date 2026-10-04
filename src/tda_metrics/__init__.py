"""Топологические и статистические метрики схожести распределений.

Модули:
    samplers          — сэмплеры облаков точек для 2D-экспериментов;
    metrics           — класс TopologyMetrics (MTD, RTD, improved PR, MMD, Fréchet, JS);
    experiments       — пайплайн экспериментов и корреляционный анализ;
    reduction         — PCA-сжатие признаков (SVD);
    experiment_schema — единая строка результата, ResultStore, run_grid (мастер-план §6.2);
    nlp_datasets      — адаптер HaluEval QA к единому контракту (мастер-план §L0);
    embeddings        — CLIP-эмбеддинги (тяжелая секция, импортируется явно);
    datasets          — параметрические датасеты на MNIST (явный импорт);
    models            — VAE и DCGAN на torch (явный импорт).
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
    permutation_test,
    bootstrap_ci,
)
from tda_metrics.reduction import pca_fit, pca_transform
from tda_metrics.experiment_schema import (
    ExperimentConfig,
    ResultRow,
    ResultStore,
    DuplicateKeyError,
    ManifestMismatchError,
    ResultStoreError,
    row_from_compute_all,
    failed_row,
    run_grid,
)
from tda_metrics.nlp_datasets import (
    load_halueval_qa,
    load_halueval_qa_canonical,
    make_prompt_id,
    to_canonical,
    audit_pairs,
)

__all__ = [
    'sample_gaussian', 'sample_uniform_cube', 'sample_ball', 'sample_ring',
    'sample_gaussian_mixture',
    'TopologyMetrics', 'np_random_seed',
    'DISTANCE_METRICS', 'SIMILARITY_METRICS', 'METRIC_GROUPS', 'ALL_METRICS',
    'constant_cloud', 'translated_cloud', 'run_experiment', 'show_steps',
    'plot_clouds', 'plot_trajectories',
    'trajectories_correlation', 'plot_correlation_heatmap', 'group_correlation_summary',
    'permutation_test', 'bootstrap_ci',
    'pca_fit', 'pca_transform',
    'ExperimentConfig', 'ResultRow', 'ResultStore',
    'DuplicateKeyError', 'ManifestMismatchError', 'ResultStoreError',
    'row_from_compute_all', 'failed_row', 'run_grid',
    'load_halueval_qa', 'load_halueval_qa_canonical', 'make_prompt_id',
    'to_canonical', 'audit_pairs',
]