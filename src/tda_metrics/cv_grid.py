"""Конфигурации и раннеры CV-экспериментов C0/C1 (мастер-план §Часть II).

Строит same-law контроли шести представлений (C0) и зеркальные классовые
эксперименты mode dropping / mode invention на MNIST (C1a/C1b) поверх единой
schema `experiment_schema.ExperimentConfig`. Конвенция сидов — как в
`synthetic_grid`: P — «данные» (seed 42 + 1000·k), Q — «модель» (seed 7 +
1000·k), k — номер повтора. Классовый состав облаков строится через
`class_balanced_sample` по заранее подготовленным пулам (`prepare_pools`):
один пул на представление, PCA (если есть) обучена один раз на полном пуле
и используется как проекция и для P, и для Q — утечка информации через
раздельное обучение PCA на P/Q исключена. Модуль не вызывает ни эмбеддеры,
ни метрики напрямую: build_clouds принимает готовые пулы, make_runner —
функцию метрик (инъекция; TopologyMetrics.compute_all на VM, заглушка
в тестах).
"""
from __future__ import annotations

import numpy as np

from tda_metrics.experiment_schema import ExperimentConfig
from tda_metrics.reduction import pca_fit, pca_transform

__all__ = [
    'SPACES',
    'class_balanced_sample',
    'build_c0_configs',
    'build_c1_configs',
    'build_clouds',
    'make_runner',
    'parse_experiment_id',
    'prepare_pools',
]

SPACES = {
    'pixels_pca16':      dict(dataset='mnist',  kind='pixels', pca=16, model=''),
    'clip_raw':          dict(dataset='mnist',  kind='clip',   pca=None, model='clip-vit-b32'),
    'clip_pca16':        dict(dataset='mnist',  kind='clip',   pca=16, model='clip-vit-b32'),
    'dino_pca16':        dict(dataset='mnist',  kind='dino',   pca=16, model='dinov2-vits14'),
    'vae_latent16':      dict(dataset='mnist',  kind='vae',    pca=None, model='conv-vae-16'),
    'cifar_pixels_pca16': dict(dataset='cifar10', kind='pixels', pca=16, model=''),
}

MNIST_SPACES = tuple(name for name, info in SPACES.items() if info['dataset'] == 'mnist')

N_CLOUD = 1000
C0_REPEATS = 5
C1_REPEATS = 5
C1_LEVELS = (1, 2, 3, 4, 5)

SEED_P = 42
SEED_Q = 7
SEED_STEP = 1000

_DEFAULT_PCA_DIM = 16


def _seed_p(k):
    return SEED_P + SEED_STEP * k


def _seed_q(k):
    return SEED_Q + SEED_STEP * k


def class_balanced_sample(pool, labels, classes, n, seed, return_class_ids=False):
    """Сбалансированная выборка n строк pool по списку классов.

    Квоты — `n // k` на класс плюс +1 первым `n % k` классам (k = число
    классов, порядок — как в classes). Внутри класса строки берутся без
    возвращения через `default_rng(seed).permutation` одного общего
    генератора (последовательные классы продолжают его состояние — разные
    seed дают разные выборки, тот же seed детерминирован). Нехватка строк
    у класса — ValueError. При return_class_ids=True возвращает дополнительно
    метку класса каждой строки результата (в порядке следования классов).
    """
    pool = np.asarray(pool)
    labels = np.asarray(labels)
    if len(pool) != len(labels):
        raise ValueError(f'pool и labels разной длины: {len(pool)} != {len(labels)}')
    classes = list(classes)
    k = len(classes)
    if k == 0:
        raise ValueError('пустой список классов')
    quotas = [n // k + (1 if i < n % k else 0) for i in range(k)]
    rng = np.random.default_rng(seed)
    rows, class_ids = [], []
    for cls, quota in zip(classes, quotas):
        idx = np.flatnonzero(labels == cls)
        if len(idx) < quota:
            raise ValueError(
                f'класс {cls!r}: доступно {len(idx)} строк, требуется {quota}'
            )
        chosen = rng.permutation(idx)[:quota]
        rows.append(pool[chosen])
        class_ids.append(np.full(quota, cls))
    sample = np.concatenate(rows, axis=0)
    if return_class_ids:
        return sample, np.concatenate(class_ids, axis=0)
    return sample


def _common_config_kwargs(info, seed, **extra):
    return dict(
        dataset=info['dataset'],
        model=info['model'],
        representation=info['kind'],
        pca_dim=info['pca'],
        n_P=N_CLOUD, n_Q=N_CLOUD,
        seed=seed,
        metric_params=None,
        **extra,
    )


def build_c0_configs():
    """C0: same-law контроли — 6 пространств × 5 повторов (30).

    Оба облака сбалансированы по всем 10 классам MNIST/CIFAR-10.
    """
    configs = []
    for space, info in SPACES.items():
        for k in range(C0_REPEATS):
            configs.append(ExperimentConfig(
                experiment_id=f'c0/{space}/r{k}',
                experiment_family='c0_samelaw',
                **_common_config_kwargs(info, k),
            ))
    return configs


def build_c1_configs():
    """C1: mode dropping (C1a) / mode invention (C1b) — 5 MNIST-пространств
    × уровни 1..5 × 5 повторов, две зеркальные серии (250).

    drop: P — все 10 классов, Q — классы 0..(9-nd); drop_fraction = nd/10.
    invent: P — классы 0..4, Q — классы 0..4 плюс 5..(4+na); invent_fraction = na/10.
    """
    configs = []
    for space in MNIST_SPACES:
        info = SPACES[space]
        for nd in C1_LEVELS:
            for k in range(C1_REPEATS):
                configs.append(ExperimentConfig(
                    experiment_id=f'c1a/{space}/d{nd}/r{k}',
                    experiment_family='c1_drop',
                    **_common_config_kwargs(info, k, drop_fraction=nd / 10),
                ))
        for na in C1_LEVELS:
            for k in range(C1_REPEATS):
                configs.append(ExperimentConfig(
                    experiment_id=f'c1b/{space}/a{na}/r{k}',
                    experiment_family='c1_invent',
                    **_common_config_kwargs(info, k, invent_fraction=na / 10),
                ))
    return configs


def _parse_space(name, eid):
    if name not in SPACES:
        raise ValueError(
            f'неизвестное пространство {name!r} в experiment_id {eid!r}; '
            f'допустимые: {sorted(SPACES)}'
        )
    return name


def _parse_int(text, field, eid):
    try:
        return int(text)
    except ValueError:
        raise ValueError(
            f'непарсимое поле {field} {text!r} в experiment_id {eid!r}'
        ) from None


def parse_experiment_id(eid):
    """Разбор experiment_id (c0/c1a/c1b) в dict полей; мусорный eid -> ValueError.

    space сверяется с реестром SPACES (опечатка — ValueError со списком
    допустимых, а не KeyError в build_clouds); числовые суффиксы парсятся
    int()-ом с контекстной ошибкой (пустой или нечисловой repeat/level).
    """
    parts = eid.split('/')
    family = parts[0]
    if family == 'c0':
        if len(parts) != 3 or not parts[2].startswith('r'):
            raise ValueError(f'неизвестный experiment_id: {eid}')
        return {
            'experiment': 'c0', 'space': _parse_space(parts[1], eid),
            'repeat': _parse_int(parts[2][1:], 'repeat', eid),
        }
    if family == 'c1a':
        if len(parts) != 4 or not parts[2].startswith('d') or not parts[3].startswith('r'):
            raise ValueError(f'неизвестный experiment_id: {eid}')
        return {
            'experiment': 'c1a', 'space': _parse_space(parts[1], eid),
            'level': _parse_int(parts[2][1:], 'level', eid),
            'repeat': _parse_int(parts[3][1:], 'repeat', eid),
        }
    if family == 'c1b':
        if len(parts) != 4 or not parts[2].startswith('a') or not parts[3].startswith('r'):
            raise ValueError(f'неизвестный experiment_id: {eid}')
        return {
            'experiment': 'c1b', 'space': _parse_space(parts[1], eid),
            'level': _parse_int(parts[2][1:], 'level', eid),
            'repeat': _parse_int(parts[3][1:], 'repeat', eid),
        }
    raise ValueError(f'неизвестный experiment_id: {eid}')


def _select_classes(config):
    """Классовый состав и сиды P/Q по experiment_id конфигурации.

    Возвращает (classes_P, classes_Q, seed_P, seed_Q).
    """
    parsed = parse_experiment_id(config.experiment_id)
    k = parsed['repeat']
    seed_p, seed_q = _seed_p(k), _seed_q(k)
    experiment = parsed['experiment']
    if experiment == 'c0':
        classes_p = list(range(10))
        classes_q = list(range(10))
    elif experiment == 'c1a':
        nd = parsed['level']
        classes_p = list(range(10))
        classes_q = list(range(10 - nd))
    elif experiment == 'c1b':
        na = parsed['level']
        classes_p = list(range(5))
        classes_q = list(range(5 + na))
    else:
        raise ValueError(f'неизвестное семейство эксперимента: {experiment!r}')
    return classes_p, classes_q, seed_p, seed_q


def build_clouds(config, pools):
    """Чистая функция: (P, Q) по конфигу и готовым пулам представлений.

    pools — dict space -> (матрица (N, d), labels (N,)); PCA в матрицах уже
    применена на этапе prepare_pools. Детерминирована при фиксированных pools.
    """
    parsed = parse_experiment_id(config.experiment_id)
    classes_p, classes_q, seed_p, seed_q = _select_classes(config)
    pool, labels = pools[parsed['space']]
    p = class_balanced_sample(pool, labels, classes_p, config.n_P, seed_p)
    q = class_balanced_sample(pool, labels, classes_q, config.n_Q, seed_q)
    return p, q


def make_runner(metrics_fn, pools):
    """runner(config) -> dict метрик с кэшем облаков по experiment_id (в рамках grid)."""
    cache = {}

    def runner(config):
        eid = config.experiment_id
        if eid not in cache:
            cache[eid] = build_clouds(config, pools)
        p, q = cache[eid]
        return metrics_fn(p, q)

    return runner


def _flatten_pixels(images):
    images = np.asarray(images, dtype=np.float64)
    return images.reshape(len(images), -1)


def prepare_pools(images, labels, embedders, pca_spaces=None, spaces=None):
    """Пулы представлений для одного датасета: dict space -> (матрица, labels).

    images/labels — один датасет (MNIST либо CIFAR train). embedders — dict
    kind -> callable(images) -> матрица (N, d) для 'clip'/'dino'/'vae';
    'pixels' эмбеддера не требует — строки это flatten картинок в float64
    без нормализации. PCA обучается через pca_fit один раз на ПОЛНОМ пуле
    пространства (reference) и проецируется той же моделью — те же строки,
    что идут в пул (никогда отдельно на P/Q).

    pca_spaces, если задан, явно перечисляет пространства, к которым
    применяется PCA (override реестра SPACES); по умолчанию (None) PCA
    применяется к пространствам с `pca` из SPACES, заданным не None, с
    числом компонент из этого поля (запасное значение 16, если пространство
    переопределено через pca_spaces, а в реестре pca=None).

    spaces, если задан, — фильтр выхода: строятся и возвращаются только
    перечисленные пространства (пересечение с пространствами доступного
    kind, как и без фильтра; пустой список — пустой словарь пулов);
    невалидное имя — ValueError со списком допустимых. Основной сценарий —
    датасетные вызовы без ловушки слияния: MNIST-вызов с
    spaces=MNIST_SPACES и CIFAR-вызов с spaces=('cifar_pixels_pca16',)
    делают слияние словарей пулов безопасным безусловно — каждый вызов
    несёт только пространства своего датасета.

    Решение по неоднозначности 'pixels': в реестре два пространства имеют
    kind='pixels' (MNIST и CIFAR), различаясь только dataset; сигнатура
    спеки не несёт параметра dataset, поэтому функция — чистая склейка по
    kind и строит пул для ВСЕХ пространств, чей kind доступен (embedders
    плюс всегда 'pixels'), из одних и тех же переданных images. При вызове
    с MNIST-данными и spaces=None оба pixel-пространства (pixels_pca16,
    cifar_pixels_pca16) получат один и тот же (в данном случае MNIST) пул.
    Правильная сборка CIFAR-пула — отдельный вызов с CIFAR images/labels;
    при слиянии словарей пулов (без spaces-фильтра) запись
    'cifar_pixels_pca16', полученную из MNIST-вызова, перезаписывать
    результатом CIFAR-вызова.
    """
    labels = np.asarray(labels)
    if spaces is not None:
        spaces = list(spaces)
        unknown = [name for name in spaces if name not in SPACES]
        if unknown:
            raise ValueError(
                f'неизвестные пространства: {unknown}; '
                f'допустимые: {sorted(SPACES)}'
            )
    available_kinds = set(embedders) | {'pixels'}
    candidates = {
        name: info for name, info in SPACES.items()
        if info['kind'] in available_kinds
        and (spaces is None or name in spaces)
    }
    raw_by_kind = {}
    pools = {}
    for name, info in candidates.items():
        kind = info['kind']
        if kind not in raw_by_kind:
            if kind == 'pixels':
                raw_by_kind[kind] = _flatten_pixels(images)
            else:
                raw_by_kind[kind] = np.asarray(embedders[kind](images), dtype=np.float64)
        raw = raw_by_kind[kind]
        pca_dim = info['pca']
        apply_pca = (pca_dim is not None) if pca_spaces is None else (name in pca_spaces)
        if apply_pca:
            n_components = pca_dim if pca_dim is not None else _DEFAULT_PCA_DIM
            model = pca_fit(raw, n_components=n_components)
            projected, _ = pca_transform(raw, model=model)
            pools[name] = (projected, labels)
        else:
            pools[name] = (raw, labels)
    return pools
