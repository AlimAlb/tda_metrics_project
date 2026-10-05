"""Инварианты билдеров CV-экспериментов C0/C1 (same-law контроли, mode drop/invent)."""
from collections import Counter

import numpy as np
import pytest

from tda_metrics.cv_grid import (
    MNIST_SPACES,
    SPACES,
    build_c0_configs,
    build_c1_configs,
    build_clouds,
    class_balanced_sample,
    make_runner,
    parse_experiment_id,
    prepare_pools,
)
from tda_metrics.cv_grid import _select_classes
from tda_metrics.reduction import pca_fit, pca_transform


def _unique_keys(configs):
    keys = [c.key() for c in configs]
    assert len(keys) == len(set(keys))
    return keys


MNIST_SPACE_NAMES = tuple(name for name, info in SPACES.items() if info['dataset'] == 'mnist')


# --------------------------------------------------------------------------
# 1. Билдеры конфигов
# --------------------------------------------------------------------------

def test_c0_config_count_and_space_coverage():
    configs = build_c0_configs()
    assert len(configs) == 6 * 5
    _unique_keys(configs)
    spaces_seen = {parse_experiment_id(c.experiment_id)['space'] for c in configs}
    assert spaces_seen == set(SPACES)


def test_c0_fields_match_registry():
    configs = {c.experiment_id: c for c in build_c0_configs()}
    clip_pca = configs['c0/clip_pca16/r2']
    assert clip_pca.dataset == 'mnist'
    assert clip_pca.model == 'clip-vit-b32'
    assert clip_pca.pca_dim == 16
    assert clip_pca.n_P == clip_pca.n_Q == 1000
    assert clip_pca.seed == 2
    assert clip_pca.metric_params is None
    assert clip_pca.drop_fraction is None and clip_pca.invent_fraction is None

    clip_raw = configs['c0/clip_raw/r0']
    assert clip_raw.pca_dim is None

    cifar = configs['c0/cifar_pixels_pca16/r4']
    assert cifar.dataset == 'cifar10'
    assert cifar.model == ''
    assert cifar.pca_dim == 16


def test_c1_config_count_and_mnist_space_coverage():
    configs = build_c1_configs()
    assert len(configs) == 5 * (5 + 5) * 5
    _unique_keys(configs)
    spaces_seen = {parse_experiment_id(c.experiment_id)['space'] for c in configs}
    assert spaces_seen == set(MNIST_SPACE_NAMES)
    assert len(MNIST_SPACE_NAMES) == 5


def test_c1_fields_match_registry_drop_and_invent():
    configs = {c.experiment_id: c for c in build_c1_configs()}

    drop = configs['c1a/dino_pca16/d3/r1']
    assert drop.experiment_family == 'c1_drop'
    assert drop.drop_fraction == 0.3
    assert drop.invent_fraction is None
    assert drop.seed == 1
    assert drop.n_P == drop.n_Q == 1000
    assert drop.pca_dim == 16
    assert drop.model == 'dinov2-vits14'

    invent = configs['c1b/vae_latent16/a5/r4']
    assert invent.experiment_family == 'c1_invent'
    assert invent.invent_fraction == 0.5
    assert invent.drop_fraction is None
    assert invent.seed == 4
    assert invent.pca_dim is None
    assert invent.model == 'conv-vae-16'


def test_all_cv_configs_unique_keys_and_metric_params_none():
    combined = build_c0_configs() + build_c1_configs()
    _unique_keys(combined)
    assert all(c.metric_params is None for c in combined)


# --------------------------------------------------------------------------
# 2. class_balanced_sample
# --------------------------------------------------------------------------

@pytest.fixture
def labeled_pool():
    labels = np.repeat(np.arange(10), 300)
    n = len(labels)
    pool = np.stack(
        [labels.astype(float), np.arange(n, dtype=float), np.arange(n, dtype=float) * 0.5],
        axis=1,
    )
    return pool, labels


def test_quotas_exact_with_remainder(labeled_pool):
    pool, labels = labeled_pool
    sample, class_ids = class_balanced_sample(
        pool, labels, classes=[0, 1, 2, 3], n=10, seed=0, return_class_ids=True,
    )
    counts = Counter(class_ids.tolist())
    assert counts == {0: 3, 1: 3, 2: 2, 3: 2}
    assert sample.shape == (10, 3)


def test_quotas_exact_divisible(labeled_pool):
    pool, labels = labeled_pool
    _, class_ids = class_balanced_sample(
        pool, labels, classes=list(range(10)), n=1000, seed=0, return_class_ids=True,
    )
    counts = Counter(class_ids.tolist())
    assert counts == {c: 100 for c in range(10)}


def test_deterministic_by_seed(labeled_pool):
    pool, labels = labeled_pool
    a = class_balanced_sample(pool, labels, [0, 1, 2], 30, seed=7)
    b = class_balanced_sample(pool, labels, [0, 1, 2], 30, seed=7)
    np.testing.assert_array_equal(a, b)


def test_different_seed_gives_different_sample(labeled_pool):
    pool, labels = labeled_pool
    a = class_balanced_sample(pool, labels, [0, 1, 2], 30, seed=1)
    b = class_balanced_sample(pool, labels, [0, 1, 2], 30, seed=2)
    assert not np.array_equal(a, b)


def test_shortage_raises_value_error(labeled_pool):
    pool, labels = labeled_pool
    with pytest.raises(ValueError):
        class_balanced_sample(pool, labels, classes=[0], n=400, seed=0)


def test_return_class_ids_matches_values(labeled_pool):
    pool, labels = labeled_pool
    sample, class_ids = class_balanced_sample(
        pool, labels, classes=[5, 7], n=20, seed=3, return_class_ids=True,
    )
    assert set(class_ids.tolist()) == {5, 7}
    np.testing.assert_array_equal(sample[:, 0], class_ids.astype(float))


# --------------------------------------------------------------------------
# 3. _select_classes + build_clouds
# --------------------------------------------------------------------------

@pytest.fixture
def class_encoded_pools():
    """Пулы различимы по space: значения сдвинуты на 100·idx пространства.

    Колонка 0 по-прежнему кодирует класс (label + сдвиг), поэтому счетчики
    состава P/Q внутри одного пространства сопоставимы, но регрессия
    «build_clouds взял не тот пул» видима: значения разных пространств
    не совпадают.
    """
    labels = np.repeat(np.arange(10), 300)
    pools = {}
    for idx, name in enumerate(SPACES):
        pool = np.stack([labels.astype(float) + 100.0 * idx] * 3, axis=1)
        pools[name] = (pool, labels)
    return pools


def test_fixture_pools_distinguishable_by_space(class_encoded_pools):
    first_values = {
        name: float(pool[0, 0]) for name, (pool, _) in class_encoded_pools.items()
    }
    assert len(set(first_values.values())) == len(SPACES)


def test_select_classes_c0_same_for_p_and_q():
    config = [c for c in build_c0_configs() if c.experiment_id == 'c0/clip_pca16/r1'][0]
    classes_p, classes_q, seed_p, seed_q = _select_classes(config)
    assert classes_p == classes_q == list(range(10))
    assert seed_p == 42 + 1000 and seed_q == 7 + 1000


def test_select_classes_c1a_drops_classes():
    config = [c for c in build_c1_configs() if c.experiment_id == 'c1a/clip_pca16/d3/r0'][0]
    classes_p, classes_q, _, _ = _select_classes(config)
    assert classes_p == list(range(10))
    assert classes_q == list(range(7))


def test_select_classes_c1b_invents_classes():
    config = [c for c in build_c1_configs() if c.experiment_id == 'c1b/clip_pca16/a3/r0'][0]
    classes_p, classes_q, _, _ = _select_classes(config)
    assert classes_p == list(range(5))
    assert classes_q == list(range(8))


def test_build_clouds_shapes(class_encoded_pools):
    config = [c for c in build_c0_configs() if c.experiment_id == 'c0/pixels_pca16/r0'][0]
    p, q = build_clouds(config, class_encoded_pools)
    assert p.shape == (1000, 3)
    assert q.shape == (1000, 3)


def test_build_clouds_c0_same_composition_p_and_q(class_encoded_pools):
    config = [c for c in build_c0_configs() if c.experiment_id == 'c0/pixels_pca16/r3'][0]
    p, q = build_clouds(config, class_encoded_pools)
    assert Counter(p[:, 0].tolist()) == Counter(q[:, 0].tolist())


@pytest.mark.parametrize('nd', [1, 2, 3, 4, 5])
def test_build_clouds_c1a_composition_matches_class_balanced_sample(class_encoded_pools, nd):
    space = 'clip_pca16'
    config = [
        c for c in build_c1_configs()
        if c.experiment_id == f'c1a/{space}/d{nd}/r0'
    ][0]
    pool, labels = class_encoded_pools[space]
    p, q = build_clouds(config, class_encoded_pools)
    expected_p = class_balanced_sample(pool, labels, list(range(10)), 1000, seed=42)
    expected_q = class_balanced_sample(pool, labels, list(range(10 - nd)), 1000, seed=7)
    np.testing.assert_array_equal(p, expected_p)
    np.testing.assert_array_equal(q, expected_q)


@pytest.mark.parametrize('na', [1, 2, 3, 4, 5])
def test_build_clouds_c1b_composition_matches_class_balanced_sample(class_encoded_pools, na):
    space = 'dino_pca16'
    config = [
        c for c in build_c1_configs()
        if c.experiment_id == f'c1b/{space}/a{na}/r0'
    ][0]
    pool, labels = class_encoded_pools[space]
    p, q = build_clouds(config, class_encoded_pools)
    expected_p = class_balanced_sample(pool, labels, list(range(5)), 1000, seed=42)
    expected_q = class_balanced_sample(pool, labels, list(range(5 + na)), 1000, seed=7)
    np.testing.assert_array_equal(p, expected_p)
    np.testing.assert_array_equal(q, expected_q)


@pytest.mark.parametrize('space', list(SPACES))
def test_build_clouds_routes_to_pool_of_own_space(class_encoded_pools, space):
    config = [c for c in build_c0_configs() if c.experiment_id == f'c0/{space}/r0'][0]
    p, q = build_clouds(config, class_encoded_pools)
    pool, labels = class_encoded_pools[space]
    expected_p = class_balanced_sample(pool, labels, list(range(10)), 1000, seed=42)
    expected_q = class_balanced_sample(pool, labels, list(range(10)), 1000, seed=7)
    np.testing.assert_array_equal(p, expected_p)
    np.testing.assert_array_equal(q, expected_q)


# --------------------------------------------------------------------------
# 4. parse_experiment_id
# --------------------------------------------------------------------------

def test_parse_experiment_id_round_trip_all_builders():
    for config in build_c0_configs():
        parsed = parse_experiment_id(config.experiment_id)
        assert parsed['experiment'] == 'c0'
        assert parsed['space'] in SPACES
        assert 0 <= parsed['repeat'] < 5
    for config in build_c1_configs():
        parsed = parse_experiment_id(config.experiment_id)
        assert parsed['experiment'] in ('c1a', 'c1b')
        assert parsed['space'] in MNIST_SPACE_NAMES
        assert 1 <= parsed['level'] <= 5
        assert 0 <= parsed['repeat'] < 5


def test_parse_experiment_id_rejects_garbage():
    with pytest.raises(ValueError):
        parse_experiment_id('unknown/space/r0')
    with pytest.raises(ValueError):
        parse_experiment_id('c0/clip_pca16')
    with pytest.raises(ValueError):
        parse_experiment_id('c1a/clip_pca16/x3/r0')


@pytest.mark.parametrize('bad_eid', [
    'c0/typo_space/r0',
    'c1a/typo_space/d1/r0',
    'c1b/typo_space/a1/r0',
    'c0/clip_pca16/r',
    'c0/clip_pca16/rx',
    'c1a/clip_pca16/d/r0',
])
def test_parse_experiment_id_contextual_value_errors(bad_eid):
    with pytest.raises(ValueError):
        parse_experiment_id(bad_eid)


def test_parse_experiment_id_unknown_space_lists_valid():
    with pytest.raises(ValueError, match='pixels_pca16'):
        parse_experiment_id('c0/typo_space/r0')


# --------------------------------------------------------------------------
# 5. prepare_pools с мок-эмбеддерами
# --------------------------------------------------------------------------

@pytest.fixture
def mock_dataset():
    rng = np.random.default_rng(0)
    images = (rng.random((40, 28, 28)) * 255).astype(np.uint8)
    labels = rng.integers(0, 10, size=40)
    return images, labels


@pytest.fixture
def mock_embedders(mock_dataset):
    images, _ = mock_dataset
    n = len(images)
    rng = np.random.default_rng(1)

    def _dominant_axis_matrix(d, scale=50.0):
        raw = rng.normal(size=(n, d))
        raw[:, 0] *= scale
        return raw

    clip_raw = _dominant_axis_matrix(32)
    dino_raw = _dominant_axis_matrix(24)
    vae_raw = rng.normal(size=(n, 7))

    embedders = {
        'clip': lambda imgs, _raw=clip_raw: _raw,
        'dino': lambda imgs, _raw=dino_raw: _raw,
        'vae': lambda imgs, _raw=vae_raw: _raw,
    }
    return embedders, {'clip': clip_raw, 'dino': dino_raw, 'vae': vae_raw}


def test_prepare_pools_shapes_all_spaces(mock_dataset, mock_embedders):
    images, labels = mock_dataset
    embedders, _ = mock_embedders
    pools = prepare_pools(images, labels, embedders)
    assert set(pools) == set(SPACES)
    assert pools['pixels_pca16'][0].shape == (40, 16)
    assert pools['cifar_pixels_pca16'][0].shape == (40, 16)
    assert pools['clip_raw'][0].shape == (40, 32)
    assert pools['clip_pca16'][0].shape == (40, 16)
    assert pools['dino_pca16'][0].shape == (40, 16)
    assert pools['vae_latent16'][0].shape == (40, 7)


def test_prepare_pools_raw_space_has_no_pca(mock_dataset, mock_embedders):
    images, labels = mock_dataset
    embedders, raws = mock_embedders
    pools = prepare_pools(images, labels, embedders)
    np.testing.assert_array_equal(pools['clip_raw'][0], raws['clip'])
    np.testing.assert_array_equal(pools['vae_latent16'][0], raws['vae'])


def test_prepare_pools_pca_fitted_on_full_pool(mock_dataset, mock_embedders):
    images, labels = mock_dataset
    embedders, raws = mock_embedders
    pools = prepare_pools(images, labels, embedders)
    expected, _ = pca_transform(raws['clip'], model=pca_fit(raws['clip'], n_components=16))
    np.testing.assert_allclose(pools['clip_pca16'][0], expected)


def test_prepare_pools_pca_first_component_is_dominant_axis(mock_dataset, mock_embedders):
    images, labels = mock_dataset
    embedders, raws = mock_embedders
    _, components, _ = pca_fit(raws['dino'], n_components=16)
    assert abs(components[0, 0]) > 0.95


def test_prepare_pools_labels_propagated(mock_dataset, mock_embedders):
    images, labels = mock_dataset
    embedders, _ = mock_embedders
    pools = prepare_pools(images, labels, embedders)
    for space in SPACES:
        np.testing.assert_array_equal(pools[space][1], labels)


def test_prepare_pools_pca_spaces_overrides_registry(mock_dataset, mock_embedders):
    images, labels = mock_dataset
    embedders, raws = mock_embedders
    pools = prepare_pools(images, labels, embedders, pca_spaces=('clip_raw',))
    expected, _ = pca_transform(raws['clip'], model=pca_fit(raws['clip'], n_components=16))
    np.testing.assert_allclose(pools['clip_raw'][0], expected)
    np.testing.assert_array_equal(pools['clip_pca16'][0], raws['clip'])
    np.testing.assert_array_equal(pools['dino_pca16'][0], raws['dino'])
    np.testing.assert_array_equal(pools['vae_latent16'][0], raws['vae'])
    np.testing.assert_array_equal(
        pools['pixels_pca16'][0],
        images.reshape(len(images), -1).astype(np.float64),
    )


def test_prepare_pools_pca_spaces_empty_disables_all_pca(mock_dataset, mock_embedders):
    images, labels = mock_dataset
    embedders, raws = mock_embedders
    pools = prepare_pools(images, labels, embedders, pca_spaces=())
    assert pools['pixels_pca16'][0].shape == (40, 784)
    np.testing.assert_array_equal(pools['clip_pca16'][0], raws['clip'])
    np.testing.assert_array_equal(pools['dino_pca16'][0], raws['dino'])


def test_prepare_pools_spaces_filter_returns_only_requested(mock_dataset, mock_embedders):
    images, labels = mock_dataset
    embedders, _ = mock_embedders
    pools = prepare_pools(
        images, labels, embedders, spaces=('clip_raw', 'cifar_pixels_pca16'),
    )
    assert set(pools) == {'clip_raw', 'cifar_pixels_pca16'}
    assert pools['clip_raw'][0].shape == (40, 32)
    assert pools['cifar_pixels_pca16'][0].shape == (40, 16)


def test_prepare_pools_spaces_filter_intersects_available_kinds(mock_dataset):
    images, labels = mock_dataset
    pools = prepare_pools(
        images, labels, embedders={}, spaces=('pixels_pca16', 'clip_raw'),
    )
    assert set(pools) == {'pixels_pca16'}


def test_prepare_pools_spaces_filter_mnist_spaces_scenario(mock_dataset, mock_embedders):
    images, labels = mock_dataset
    embedders, _ = mock_embedders
    pools = prepare_pools(images, labels, embedders, spaces=MNIST_SPACES)
    assert set(pools) == set(MNIST_SPACE_NAMES)


def test_prepare_pools_spaces_invalid_name_raises(mock_dataset, mock_embedders):
    images, labels = mock_dataset
    embedders, _ = mock_embedders
    with pytest.raises(ValueError, match='nonexistent_space'):
        prepare_pools(
            images, labels, embedders, spaces=('clip_raw', 'nonexistent_space'),
        )


# --------------------------------------------------------------------------
# 6. make_runner
# --------------------------------------------------------------------------

def test_make_runner_caches_clouds_by_experiment_id(class_encoded_pools):
    config = [c for c in build_c0_configs() if c.experiment_id == 'c0/pixels_pca16/r0'][0]
    build_calls = []

    def metrics_fn(p, q):
        build_calls.append((p.shape, q.shape))
        return {'mmd': 0.0}

    runner = make_runner(metrics_fn, class_encoded_pools)
    assert runner(config) == {'mmd': 0.0}
    assert runner(config) == {'mmd': 0.0}
    assert len(build_calls) == 2
    assert build_calls[0] == ((1000, 3), (1000, 3))


def test_make_runner_builds_clouds_once_per_eid(monkeypatch, class_encoded_pools):
    import tda_metrics.cv_grid as cv_grid

    config = [c for c in build_c0_configs() if c.experiment_id == 'c0/clip_pca16/r0'][0]
    build_call_count = {'n': 0}
    original_build_clouds = cv_grid.build_clouds

    def counting_build_clouds(cfg, pools):
        build_call_count['n'] += 1
        return original_build_clouds(cfg, pools)

    monkeypatch.setattr(cv_grid, 'build_clouds', counting_build_clouds)
    runner = cv_grid.make_runner(lambda p, q: {'mmd': 0.0}, class_encoded_pools)
    runner(config)
    runner(config)
    runner(config)
    assert build_call_count['n'] == 1
