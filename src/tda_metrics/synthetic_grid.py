"""Конфигурации и раннеры синтетических экспериментов S0/S1/S3/S4 (мастер-план §Часть I).

Строит списки ExperimentConfig и облака (P, Q) по конвенциям проекта:
P — данные (seed 42 + 1000·k), Q — модель (seed 7 + 1000·k), повторы k.
Единственная пара с одинаковыми облаками — режим copy в S0 (это суть
эксперимента «точная копия» мастера). Метрики модуль не вызывает:
make_runner принимает metrics_fn (инъекция; TopologyMetrics.compute_all
на VM, заглушка в тестах). RTD на независимых облаках — exploratory
random-coupling; параметры метрик всей серии фиксированы в METRIC_PARAMS.
"""
from tda_metrics.experiment_schema import ExperimentConfig
from tda_metrics.mixtures import (
    make_mode_reference,
    sample_q_dropping_invention,
    sample_reference,
)
from tda_metrics.samplers import (
    reflect_cloud,
    rotate_cloud,
    sample_gaussian,
    sample_gaussian_mixture,
    sample_ring,
    scale_cloud,
    stretch_cloud,
    translate_cloud,
)

__all__ = [
    'METRIC_PARAMS',
    'build_clouds',
    'build_s0_configs',
    'build_s1_configs',
    'build_s1_samelaw_configs',
    'build_s3_configs',
    'build_s4_configs',
    'make_runner',
    'parse_experiment_id',
    's1_base_cloud',
    's3_reference',
]

METRIC_PARAMS = (
    '{"nhood_sizes":[1,3,10],"ntd_repeats":5,"rtd_batch":500,"rtd_trials":2,'
    '"rtd_mode":"random-coupling-exploratory"}'
)

SEED_P = 42
SEED_Q = 7
SEED_STEP = 1000

S0_DISTRIBUTIONS = ('gaussian', 'ring', 'mixture')
S0_NS = (250, 500, 1000)
S0_MODES = ('copy', 'samelaw')
S0_REPEATS = 5

S1_N = 1000
S1_RADIUS = 1.5
S1_THICKNESS = 1.0
S1_REPEATS = 5
S1_TRANSFORMATIONS = {
    'shift': (0.0, 0.3, 0.6, 1.2, 1.8, 2.4, 3.0),
    'rotation': (0.0, 15.0, 30.0, 60.0, 120.0, 180.0, 270.0),
    'reflection': (0, 1),
    'scale': (0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5),
    'thickness': (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75),
    'anisotropy': (1.0, 1.1, 1.25, 1.5, 2.0, 2.75, 4.0),
}

S3_GRID = (0.0, 0.25, 0.5, 0.75)
S3_N = 1000
S3_REPEATS = 10

S4_LEVELS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75)
S4_REPEATS = 10


def _seed_p(k):
    return SEED_P + SEED_STEP * k


def _seed_q(k):
    return SEED_Q + SEED_STEP * k


def _config(experiment_id, dataset, n, seed, **extra):
    return ExperimentConfig(
        experiment_id=experiment_id,
        experiment_family='synthetic',
        dataset=dataset,
        representation='raw_2d',
        n_P=n,
        n_Q=n,
        seed=seed,
        metric_params=METRIC_PARAMS,
        **extra,
    )


def _s0_cloud(distribution, n, seed):
    if distribution == 'gaussian':
        return sample_gaussian(n, dim=2, seed=seed)
    if distribution == 'ring':
        return sample_ring(n, radius=S1_RADIUS, thickness=S1_THICKNESS, seed=seed)
    if distribution == 'mixture':
        return sample_gaussian_mixture(n, radius=1.2, n_components=8, seed=seed)
    raise ValueError(f'неизвестное распределение: {distribution}')


def s3_reference():
    """Референс S3/S4: 8 мод на радиусе 8, разброс 1."""
    return make_mode_reference(n_modes=8, radius=8.0, spread=1.0)


def s1_base_cloud(n, seed, thickness=S1_THICKNESS):
    """Базовый закон S1: толстое кольцо."""
    return sample_ring(n, radius=S1_RADIUS, thickness=thickness, seed=seed)


def build_s0_configs():
    """S0: полы метрик vs n — точная копия и same-law, 3 закона × 3 n × 5 повторов (90)."""
    configs = []
    for distribution in S0_DISTRIBUTIONS:
        for n in S0_NS:
            for mode in S0_MODES:
                for k in range(S0_REPEATS):
                    configs.append(_config(
                        f's0_{distribution}_{mode}_n{n}_k{k}', distribution, n, k,
                    ))
    return configs


def build_s1_configs():
    """S1: карта инвариантов — 6 трансформаций × уровни × 5 повторов (185).

    Отражение бинарно (0/1): изометрия не имеет естественной шкалы уровней,
    остальные трансформации — 5–7 уровней включая базовый. Трансформация и
    уровень кодируются в experiment_id (в §6.2 отдельного поля нет).
    """
    configs = []
    for transformation, levels in S1_TRANSFORMATIONS.items():
        for level in levels:
            for k in range(S1_REPEATS):
                configs.append(_config(
                    f's1_{transformation}_l{level}_k{k}', 'ring', S1_N, k,
                ))
    return configs


def build_s1_samelaw_configs():
    """S1: same-law строки — по одной на трансформацию с её базовым уровнем (30)."""
    base_levels = {
        'shift': 0.0,
        'rotation': 0.0,
        'reflection': 0,
        'scale': 1.0,
        'thickness': S1_THICKNESS,
        'anisotropy': 1.0,
    }
    configs = []
    for transformation, level in base_levels.items():
        for k in range(S1_REPEATS):
            configs.append(_config(
                f's1samelaw_{transformation}_k{k}', 'ring', S1_N, k,
            ))
    return configs


def build_s3_configs():
    """S3: 2D mode dropping × invention — сетка 4×4 × 10 повторов (160); (0,0) — same-law."""
    configs = []
    for d in S3_GRID:
        for i in S3_GRID:
            for k in range(S3_REPEATS):
                configs.append(_config(
                    f's3_d{d}_i{i}_k{k}', 'modes8', S3_N, k,
                    drop_fraction=d, invent_fraction=i,
                ))
    return configs


def build_s4_configs():
    """S4: раннее обнаружение — срезы по d (i=0) и по i (d=0), 8 уровней × 10 повторов (160)."""
    configs = []
    for axis in ('drop', 'invent'):
        for level in S4_LEVELS:
            for k in range(S4_REPEATS):
                extra = (
                    {'drop_fraction': level, 'invent_fraction': 0.0}
                    if axis == 'drop'
                    else {'drop_fraction': 0.0, 'invent_fraction': level}
                )
                configs.append(_config(
                    f's4_{axis}_l{level}_k{k}', 'modes8', S3_N, k, **extra,
                ))
    return configs


def _transform_s1(transformation, level, cloud):
    if transformation == 'shift':
        return translate_cloud(cloud, [float(level), 0.0])
    if transformation == 'rotation':
        return rotate_cloud(cloud, float(level))
    if transformation == 'reflection':
        return reflect_cloud(cloud, 'x') if level else cloud
    if transformation == 'scale':
        return scale_cloud(cloud, float(level))
    if transformation == 'anisotropy':
        return stretch_cloud(cloud, float(level), 1.0)
    raise ValueError(f'неизвестная трансформация: {transformation}')


def build_clouds(config):
    """Чистая функция: (P, Q) по конфигу; детерминирована, метрики не зовёт."""
    eid = config.experiment_id
    k = config.seed if config.seed is not None else 0

    if eid.startswith('s0_'):
        _, distribution, mode, n_tag, k_tag = eid.split('_')
        n = int(n_tag[1:])
        p = _s0_cloud(distribution, n, _seed_p(k))
        if mode == 'copy':
            return p, p.copy()
        return p, _s0_cloud(distribution, n, _seed_q(k))

    if eid.startswith('s1samelaw_'):
        transformation = eid.split('_')[1]
        p = s1_base_cloud(S1_N, _seed_p(k))
        q = s1_base_cloud(S1_N, _seed_q(k))
        return p, q

    if eid.startswith('s1_'):
        _, transformation, level_tag, k_tag = eid.split('_')
        level = float(level_tag[1:]) if transformation != 'reflection' else int(level_tag[1:])
        p = s1_base_cloud(S1_N, _seed_p(k))
        if transformation == 'thickness':
            q = s1_base_cloud(S1_N, _seed_q(k), thickness=level)
        else:
            q = _transform_s1(transformation, level, s1_base_cloud(S1_N, _seed_q(k)))
        return p, q

    if eid.startswith('s3_'):
        _, d_tag, i_tag, k_tag = eid.split('_')
        ref = s3_reference()
        p = sample_reference(ref, S3_N, _seed_p(k))
        q = sample_q_dropping_invention(
            ref, S3_N, _seed_q(k),
            drop_fraction=float(d_tag[1:]), invent_fraction=float(i_tag[1:]),
        )
        return p, q

    if eid.startswith('s4_'):
        _, axis, level_tag, k_tag = eid.split('_')
        level = float(level_tag[1:])
        ref = s3_reference()
        p = sample_reference(ref, S3_N, _seed_p(k))
        d = level if axis == 'drop' else 0.0
        i = level if axis == 'invent' else 0.0
        q = sample_q_dropping_invention(ref, S3_N, _seed_q(k), drop_fraction=d, invent_fraction=i)
        return p, q

    raise ValueError(f'неизвестный experiment_id: {eid}')


def parse_experiment_id(eid):
    """Разбор experiment_id в dict полей (для таблиц и графиков)."""
    parts = eid.split('_')
    if parts[0] == 's0':
        return {
            'experiment': 's0', 'distribution': parts[1], 'mode': parts[2],
            'n': int(parts[3][1:]), 'repeat': int(parts[4][1:]),
        }
    if parts[0] == 's1samelaw':
        return {
            'experiment': 's1samelaw', 'transformation': parts[1],
            'repeat': int(parts[2][1:]),
        }
    if parts[0] == 's1':
        transformation = parts[1]
        level = parts[2][1:]
        return {
            'experiment': 's1', 'transformation': transformation,
            'level': int(level) if transformation == 'reflection' else float(level),
            'repeat': int(parts[3][1:]),
        }
    if parts[0] == 's3':
        return {
            'experiment': 's3', 'drop': float(parts[1][1:]),
            'invent': float(parts[2][1:]), 'repeat': int(parts[3][1:]),
        }
    if parts[0] == 's4':
        return {
            'experiment': 's4', 'axis': parts[1], 'level': float(parts[2][1:]),
            'repeat': int(parts[3][1:]),
        }
    raise ValueError(f'неизвестный experiment_id: {eid}')


def make_runner(metrics_fn):
    """runner(config) → dict метрик с кэшем облаков по experiment_id (в рамках grid)."""
    cache = {}

    def runner(config):
        eid = config.experiment_id
        if eid not in cache:
            cache[eid] = build_clouds(config)
        p, q = cache[eid]
        return metrics_fn(p, q)

    return runner