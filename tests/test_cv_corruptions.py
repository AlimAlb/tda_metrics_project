"""Инварианты matched-MMD порч CIFAR (C3): numpy-порчи, CRN-пары, конфиги, parse."""
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from tda_metrics.cv_corruptions import (
    CORRUPTIONS,
    SEVERITY_GRID,
    brightness,
    build_c3_matched_configs,
    build_c3_scan_configs,
    build_pair,
    class_balanced_indices,
    contrast,
    corrupt_images,
    gaussian_blur,
    gaussian_noise,
    parse_experiment_id,
    select_matched_severities,
)
from tda_metrics.experiment_schema import ExperimentConfig

N_CLASSES = 10
PER_CLASS = 12
POOL_SIZE = N_CLASSES * PER_CLASS
N_CLOUD = 20

_MEAN_MMD = {
    'gaussian_noise': {0.05: 10.0, 0.1: 20.0, 0.2: 35.0, 0.3: 50.0, 0.5: 80.0, 0.75: 120.0},
    'gaussian_blur': {0.05: 12.0, 0.1: 18.0, 0.2: 40.0, 0.3: 60.0, 0.5: 90.0, 0.75: 130.0},
    'brightness': {0.05: 5.0, 0.1: 15.0, 0.2: 30.0, 0.3: 45.0, 0.5: 70.0, 0.75: 105.0},
    'contrast': {0.05: 8.0, 0.1: 16.0, 0.2: 32.0, 0.3: 48.0, 0.5: 72.0, 0.75: 110.0},
}

_EXPECTED_SELECTION = {
    'gaussian_noise': (0.3, 50.0),
    'gaussian_blur': (0.2, 40.0),
    'brightness': (0.3, 45.0),
    'contrast': (0.3, 48.0),
}


def _unique_keys(configs):
    keys = [c.key() for c in configs]
    assert len(keys) == len(set(keys))
    return keys


def _all_classes(labels):
    return [int(c) for c in np.unique(labels)]


def _config_by_eid(configs, eid):
    return [c for c in configs if c.experiment_id == eid][0]


@pytest.fixture
def cifar_like():
    """Мини-CIFAR: (n,32,32,3) uint8, 10 классов по 12 изображений."""
    rng = np.random.default_rng(5)
    labels = np.repeat(np.arange(N_CLASSES), PER_CLASS)
    images = rng.integers(0, 256, size=(POOL_SIZE, 32, 32, 3), dtype=np.uint8)
    return images, labels


@pytest.fixture
def pca_embeddings():
    """PCA-эмбеддинги пула: строки уникальны (столбец 0 — номер строки)."""
    rng = np.random.default_rng(6)
    embeddings = rng.normal(size=(POOL_SIZE, 4))
    embeddings[:, 0] = np.arange(POOL_SIZE)
    return embeddings


@pytest.fixture
def textured_images():
    rng = np.random.default_rng(11)
    return rng.integers(0, 256, size=(4, 32, 32, 3), dtype=np.uint8)


@pytest.fixture
def flat_images():
    return np.full((3, 32, 32, 3), 128, dtype=np.uint8)


@pytest.fixture
def moderate_images():
    rng = np.random.default_rng(12)
    return rng.integers(60, 180, size=(4, 32, 32, 3), dtype=np.uint8)


@pytest.fixture
def synthetic_scan_frame():
    """Строки schema Этапа-1: 4 порчи x 6 severity x 2 повтора.

    Средний MMD по повторам каждой (corruption, severity) — целое число из
    _MEAN_MMD (отклонения ±1 по повторам). Целые значения выбраны сознательно:
    кейс gaussian_blur (40 и 60 при target 50) — гарантированное равенство
    расстояний в float64.
    """
    rows = []
    for corruption, means in _MEAN_MMD.items():
        for severity, mean_mmd in means.items():
            for k in (0, 1):
                rows.append({
                    'corruption': corruption,
                    'severity': severity,
                    'seed': k,
                    'mmd': mean_mmd - 1.0 + 2.0 * k,
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 1. Порчи
# --------------------------------------------------------------------------

def test_registry_and_grid():
    assert set(CORRUPTIONS) == {
        'gaussian_noise', 'gaussian_blur', 'brightness', 'contrast',
    }
    assert SEVERITY_GRID == (0.05, 0.1, 0.2, 0.3, 0.5, 0.75)


@pytest.mark.parametrize('name', sorted(CORRUPTIONS))
def test_severity_zero_is_identity(textured_images, name):
    out = corrupt_images(textured_images, name, 0.0)
    np.testing.assert_array_equal(out, textured_images)
    assert out.dtype == np.uint8
    assert out.shape == textured_images.shape


@pytest.mark.parametrize('name', sorted(CORRUPTIONS))
def test_positive_severity_changes_array(textured_images, name):
    out = corrupt_images(textured_images, name, 0.3)
    assert out.shape == textured_images.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, textured_images)


@pytest.mark.parametrize('name', sorted(CORRUPTIONS))
def test_float_input_gives_same_result_as_uint8(textured_images, name):
    from_uint8 = corrupt_images(textured_images, name, 0.2)
    from_float = corrupt_images(textured_images.astype(np.float64), name, 0.2)
    assert from_float.dtype == np.uint8
    np.testing.assert_array_equal(from_float, from_uint8)


@pytest.mark.parametrize('name', sorted(CORRUPTIONS))
def test_values_within_uint8_bounds(textured_images, name):
    out = corrupt_images(textured_images, name, 0.75)
    assert int(out.min()) >= 0
    assert int(out.max()) <= 255


def test_noise_std_grows_with_severity_on_flat(flat_images):
    low = gaussian_noise(flat_images, 0.05)
    high = gaussian_noise(flat_images, 0.75)
    assert low.std() > 0
    assert high.std() > low.std()


def test_blur_variation_decreases_with_severity(textured_images):
    soft = gaussian_blur(textured_images, 0.1)
    hard = gaussian_blur(textured_images, 0.75)
    assert soft.std() < textured_images.std()
    assert hard.std() < soft.std()


def test_brightness_mean_increases(moderate_images):
    out = brightness(moderate_images, 0.5)
    assert out.mean() > moderate_images.mean()


def test_contrast_std_increases(moderate_images):
    out = contrast(moderate_images, 0.5)
    assert out.std() > moderate_images.std()


# --------------------------------------------------------------------------
# 2. class_balanced_indices
# --------------------------------------------------------------------------

def test_exact_quotas_with_remainder():
    labels = np.repeat(np.arange(10), 30)
    idx = class_balanced_indices(labels, list(range(10)), 23, seed=42)
    counts = Counter(labels[idx].tolist())
    assert counts == {0: 3, 1: 3, 2: 3, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2}


def test_indices_deterministic_by_seed():
    labels = np.repeat(np.arange(10), 30)
    a = class_balanced_indices(labels, list(range(10)), 40, seed=7)
    b = class_balanced_indices(labels, list(range(10)), 40, seed=7)
    np.testing.assert_array_equal(a, b)


def test_different_seed_gives_different_indices():
    labels = np.repeat(np.arange(10), 30)
    a = class_balanced_indices(labels, list(range(10)), 40, seed=1)
    b = class_balanced_indices(labels, list(range(10)), 40, seed=2)
    assert not np.array_equal(a, b)


def test_indices_are_unique_valid_positions():
    labels = np.repeat(np.arange(5), 20)
    idx = class_balanced_indices(labels, [0, 2, 4], 13, seed=3)
    assert len(idx) == 13
    assert len(set(idx.tolist())) == 13
    assert idx.min() >= 0 and idx.max() < len(labels)
    assert set(labels[idx].tolist()) == {0, 2, 4}


def test_shortage_raises_value_error():
    labels = np.repeat(np.arange(10), 5)
    with pytest.raises(ValueError):
        class_balanced_indices(labels, [0, 1], 12, seed=0)


# --------------------------------------------------------------------------
# 3. Билдеры конфигов
# --------------------------------------------------------------------------

def test_scan_config_counts():
    assert len(build_c3_scan_configs(50, repeats=2)) == 4 * 6 * 2
    assert len(build_c3_scan_configs(50)) == 4 * 6 * 3


def test_scan_configs_fields_and_grid_coverage():
    configs = build_c3_scan_configs(50, repeats=2)
    _unique_keys(configs)
    assert all(c.metric_params is None for c in configs)
    seen = {(c.corruption, c.severity) for c in configs}
    assert seen == {(name, s) for name in CORRUPTIONS for s in SEVERITY_GRID}
    config = _config_by_eid(configs, 'c3scan/brightness/s0.3/r1')
    assert config.experiment_family == 'c3_scan'
    assert config.dataset == 'cifar10'
    assert config.representation == 'clip'
    assert config.model == 'clip-vit-b32'
    assert config.pca_dim == 16
    assert config.n_P == config.n_Q == 50
    assert config.severity == 0.3
    assert config.corruption == 'brightness'
    assert config.seed == 1


def test_matched_config_count_fields_and_unique_keys(synthetic_scan_frame):
    matched = build_c3_matched_configs(50.0, synthetic_scan_frame, 40, repeats=2)
    assert len(matched) == 4 * 2 + 2
    _unique_keys(matched)
    _unique_keys(build_c3_scan_configs(40, repeats=2) + matched)
    assert all(c.metric_params is None for c in matched)
    assert all(c.experiment_family == 'c3_final' for c in matched)
    by_eid = {c.experiment_id: c for c in matched}
    for corruption, (severity, _mmd) in _EXPECTED_SELECTION.items():
        config = by_eid[f'c3final/{corruption}/r1']
        assert config.corruption == corruption
        assert config.severity == severity
        assert config.dataset == 'cifar10'
        assert config.model == 'clip-vit-b32'
        assert config.representation == 'clip'
        assert config.pca_dim == 16
        assert config.n_P == config.n_Q == 40
        assert config.seed == 1
    clean = by_eid['c3final/clean/r0']
    assert clean.corruption == 'clean'
    assert clean.severity is None
    assert clean.experiment_family == 'c3_final'
    assert clean.n_P == clean.n_Q == 40
    assert clean.seed == 0


# --------------------------------------------------------------------------
# 4. select_matched_severities
# --------------------------------------------------------------------------

def test_select_matched_severities_nearest(synthetic_scan_frame):
    selected = select_matched_severities(synthetic_scan_frame, 50.0)
    assert set(selected) == set(CORRUPTIONS)
    for corruption, (severity, mmd) in _EXPECTED_SELECTION.items():
        chosen_severity, chosen_mmd = selected[corruption]
        assert chosen_severity == severity
        assert chosen_mmd == pytest.approx(mmd)


def test_select_matched_severities_tie_prefers_smaller_severity():
    rows = [
        {'corruption': 'gaussian_noise', 'severity': 0.2, 'seed': 0, 'mmd': 40.0},
        {'corruption': 'gaussian_noise', 'severity': 0.3, 'seed': 0, 'mmd': 60.0},
    ]
    selected = select_matched_severities(pd.DataFrame(rows), 50.0)
    assert selected['gaussian_noise'][0] == 0.2


# --------------------------------------------------------------------------
# 5. build_pair (CRN) + parse round-trip
# --------------------------------------------------------------------------

def test_build_pair_scan_p_by_indices_q_corrupted_same_indices(cifar_like, pca_embeddings):
    images, labels = cifar_like
    configs = build_c3_scan_configs(N_CLOUD, repeats=1)
    config = _config_by_eid(configs, 'c3scan/gaussian_noise/s0.3/r0')
    p, q_images = build_pair(config, images, pca_embeddings, labels)
    assert p.shape == (N_CLOUD, 4)
    assert q_images.shape == (N_CLOUD, 32, 32, 3)
    assert q_images.dtype == np.uint8
    idx = class_balanced_indices(labels, _all_classes(labels), N_CLOUD, seed=42)
    np.testing.assert_array_equal(p, pca_embeddings[idx])
    np.testing.assert_array_equal(
        q_images, corrupt_images(images[idx], 'gaussian_noise', 0.3),
    )
    np.testing.assert_array_equal(
        corrupt_images(images[idx], 'gaussian_noise', 0.0), images[idx],
    )


def test_build_pair_clean_control_q_equals_p_images(
    cifar_like, pca_embeddings, synthetic_scan_frame,
):
    images, labels = cifar_like
    matched = build_c3_matched_configs(50.0, synthetic_scan_frame, N_CLOUD, repeats=1)
    config = _config_by_eid(matched, 'c3final/clean/r0')
    p, q_images = build_pair(config, images, pca_embeddings, labels)
    idx = class_balanced_indices(labels, _all_classes(labels), N_CLOUD, seed=42)
    np.testing.assert_array_equal(p, pca_embeddings[idx])
    np.testing.assert_array_equal(q_images, images[idx])


def test_build_pair_different_repeats_use_different_indices(cifar_like, pca_embeddings):
    images, labels = cifar_like
    configs = build_c3_scan_configs(N_CLOUD, repeats=2)
    p0, _ = build_pair(
        _config_by_eid(configs, 'c3scan/contrast/s0.5/r0'), images, pca_embeddings, labels,
    )
    p1, _ = build_pair(
        _config_by_eid(configs, 'c3scan/contrast/s0.5/r1'), images, pca_embeddings, labels,
    )
    assert not np.array_equal(p0, p1)


def test_build_pair_size_mismatch_raises(cifar_like, pca_embeddings):
    images, labels = cifar_like
    config = ExperimentConfig(
        experiment_id='c3scan/gaussian_noise/s0.3/r0',
        experiment_family='c3_scan',
        dataset='cifar10', model='clip-vit-b32', representation='clip',
        pca_dim=16, n_P=20, n_Q=19,
        corruption='gaussian_noise', severity=0.3, seed=0,
    )
    with pytest.raises(ValueError, match='n_P'):
        build_pair(config, images, pca_embeddings, labels)


def test_build_pair_pool_length_mismatch_raises(cifar_like, pca_embeddings):
    images, labels = cifar_like
    config = ExperimentConfig(
        experiment_id='c3scan/gaussian_noise/s0.3/r0',
        experiment_family='c3_scan',
        dataset='cifar10', model='clip-vit-b32', representation='clip',
        pca_dim=16, n_P=20, n_Q=20,
        corruption='gaussian_noise', severity=0.3, seed=0,
    )
    with pytest.raises(ValueError, match='одной длины'):
        build_pair(config, images, pca_embeddings[:100], labels)


def test_parse_round_trip_scan_and_matched(synthetic_scan_frame):
    for config in build_c3_scan_configs(50, repeats=2):
        parsed = parse_experiment_id(config.experiment_id)
        assert parsed['experiment'] == 'c3scan'
        assert parsed['corruption'] == config.corruption
        assert parsed['severity'] == config.severity
        assert parsed['repeat'] == config.seed
    for config in build_c3_matched_configs(50.0, synthetic_scan_frame, 50, repeats=2):
        parsed = parse_experiment_id(config.experiment_id)
        assert parsed['experiment'] == 'c3final'
        assert parsed['corruption'] == config.corruption
        assert parsed['repeat'] == config.seed


# --------------------------------------------------------------------------
# 6. parse мусор + диспетчер corrupt_images
# --------------------------------------------------------------------------

@pytest.mark.parametrize('bad_eid', [
    'c3scan/typo_corruption/s0.3/r0',
    'c3scan/gaussian_noise/s0.4/r0',
    'c3scan/gaussian_noise/sabc/r0',
    'c3scan/gaussian_noise/s/r0',
    'c3scan/gaussian_noise/s0.3',
    'c3scan/gaussian_noise/s0.3/r',
    'c3scan/gaussian_noise/s0.3/rx',
    'c3scan/gaussian_noise/s0.3/x0',
    'c3scan/gaussian_noise',
    'c3final/typo_corruption/r0',
    'c3final/clean/r',
    'c3final/clean/rx',
    'c3final/clean',
    'c3final/gaussian_noise/x1',
    'c0/clip_pca16/r0',
    'garbage',
])
def test_parse_rejects_garbage(bad_eid):
    with pytest.raises(ValueError):
        parse_experiment_id(bad_eid)


def test_parse_unknown_corruption_lists_valid():
    with pytest.raises(ValueError, match='gaussian_noise'):
        parse_experiment_id('c3scan/typo_corruption/s0.3/r0')


def test_parse_severity_outside_grid_lists_grid():
    with pytest.raises(ValueError, match='0.05'):
        parse_experiment_id('c3scan/gaussian_noise/s0.4/r0')


def test_corrupt_images_unknown_name_raises(textured_images):
    with pytest.raises(ValueError, match='no_such_corruption'):
        corrupt_images(textured_images, 'no_such_corruption', 0.3)


@pytest.mark.parametrize('name', sorted(CORRUPTIONS))
def test_corrupt_images_accepts_full_severity_grid(textured_images, name):
    for severity in SEVERITY_GRID:
        out = corrupt_images(textured_images, name, severity)
        assert out.shape == textured_images.shape
        assert out.dtype == np.uint8
