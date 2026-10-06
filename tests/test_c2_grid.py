"""Инварианты C2-грида: цикл вращения MNIST digit 3 в четырёх представлениях (мастер-план §C2).

numpy/pandas-only: torch/CLIP/DINO/VAE не нужны — сырые представления
инъектируются в project_pair заглушкой raw_fn (детерминированные матрицы
из картинок); облака маленькие (n_digits=3..5, короткие сетки углов).
"""
import numpy as np
import pytest

from tda_metrics.c2_grid import (
    ANGLES_ARC180,
    ANGLES_CIRCLE,
    C2_COMPS,
    C2_SPACES,
    DIGIT,
    FULL_SPACES,
    N_GRID_VALUES,
    N_POINTS,
    RESAMPLES,
    SCAN_ALPHA_MAX,
    SCAN_REPEATS,
    SCAN_SUBSET_ALPHAS,
    SCALES_ZOOM,
    SUBSET_SPACES,
    angles_for,
    build_c2_resample_configs,
    build_c2_scan_configs,
    cloud_images,
    pair_clouds,
    parse_c2_experiment_id,
    project_pair,
    seed_p,
    seed_q,
)
from tda_metrics.datasets import rotate_images, select_class, sweep_images, zoom_images

FEW_ANGLES = np.arange(0.0, 40.0, 10.0)


def _unique_keys(configs):
    keys = [c.key() for c in configs]
    assert len(keys) == len(set(keys))
    return keys


# --------------------------------------------------------------------------
# 1. Билдеры конфигов
# --------------------------------------------------------------------------

def test_c2_resample_count_ids_and_keys():
    configs = build_c2_resample_configs()
    assert len(configs) == 96
    eids = [c.experiment_id for c in configs]
    assert len(set(eids)) == 96
    _unique_keys(configs)


def test_c2_resample_full_coverage_and_fields():
    configs = {c.experiment_id: c for c in build_c2_resample_configs()}
    for space in C2_SPACES:
        for comp in C2_COMPS:
            for k in RESAMPLES:
                assert f'c2r/{space}/{comp}/r{k}' in configs

    pixels = configs['c2r/pixels_pca16/cc/r0']
    assert pixels.experiment_family == 'c2_resample'
    assert pixels.dataset == 'mnist'
    assert pixels.model == ''
    assert pixels.representation == 'pixels_pca16'
    assert pixels.pca_dim == 16
    assert pixels.n_P == pixels.n_Q == N_POINTS == 450
    assert pixels.seed == 0
    assert pixels.severity is None and pixels.alpha is None
    assert pixels.metric_params is None

    clip = configs['c2r/clip_pca16/arc/r3']
    assert clip.model == 'clip-vit-b32'
    assert clip.representation == 'clip_pca16'
    assert clip.pca_dim == 16
    assert clip.seed == 3

    dino = configs['c2r/dino_pca16/zoom/r5']
    assert dino.model == 'dinov2-vits14'
    assert dino.representation == 'dino_pca16'
    assert dino.seed == 5

    vae = configs['c2r/vae_latent16/zoom/r7']
    assert vae.model == 'conv-vae-16'
    assert vae.representation == 'vae_latent16'
    assert vae.pca_dim is None
    assert vae.seed == 7


def test_c2_scan_count_keys_and_severity():
    configs = build_c2_scan_configs()
    assert len(configs) == 60
    _unique_keys(configs)
    by_eid = {c.experiment_id: c for c in configs}
    for c in configs:
        parsed = parse_c2_experiment_id(c.experiment_id)
        assert c.experiment_family == 'c2_scan'
        assert c.severity == float(parsed['alpha_max'])
        assert c.n_P == c.n_Q == 450
        assert c.metric_params is None
    for space in FULL_SPACES:
        for alpha_max in SCAN_ALPHA_MAX:
            for k in range(SCAN_REPEATS):
                assert f'c2s/{space}/a{alpha_max}/r{k}' in by_eid
    for space in SUBSET_SPACES:
        for alpha_max in SCAN_ALPHA_MAX:
            present = f'c2s/{space}/a{alpha_max}/r0' in by_eid
            assert present == (alpha_max in SCAN_SUBSET_ALPHAS)


def test_all_c2_configs_combined_unique_keys():
    combined = build_c2_resample_configs() + build_c2_scan_configs()
    assert len(combined) == 156
    _unique_keys(combined)
    assert list(RESAMPLES) == list(range(8))


# --------------------------------------------------------------------------
# 2. parse_c2_experiment_id
# --------------------------------------------------------------------------

def test_parse_valid_resample_and_scan_ids():
    parsed = parse_c2_experiment_id('c2r/pixels_pca16/zoom/r7')
    assert parsed == {'family': 'c2_resample', 'space': 'pixels_pca16',
                      'comp': 'zoom', 'repeat': 7}
    parsed = parse_c2_experiment_id('c2s/clip_pca16/a180/r0')
    assert parsed == {'family': 'c2_scan', 'space': 'clip_pca16',
                      'alpha_max': 180, 'repeat': 0}


def test_parse_round_trip_all_configs():
    for config in build_c2_resample_configs() + build_c2_scan_configs():
        parsed = parse_c2_experiment_id(config.experiment_id)
        assert parsed['space'] in C2_SPACES
        assert parsed['repeat'] >= 0


@pytest.mark.parametrize('bad_eid', [
    '',
    'nope/pixels_pca16/cc/r0',
    'c2x/pixels_pca16/cc/r0',
    'c2r/typo_space/cc/r0',
    'c2s/typo_space/a180/r0',
    'c2r/pixels_pca16',
    'c2s/pixels_pca16/a180',
    'c2r/pixels_pca16/cc/r0/x',
    'c2r/pixels_pca16/spiral/r0',
    'c2r/pixels_pca16/cc/r',
    'c2r/pixels_pca16/cc/rx',
    'c2s/pixels_pca16/a/r0',
    'c2s/pixels_pca16/ax/r0',
    'c2s/pixels_pca16/a180.5/r0',
])
def test_parse_rejects_garbage(bad_eid):
    with pytest.raises(ValueError):
        parse_c2_experiment_id(bad_eid)


def test_parse_unknown_space_lists_valid():
    with pytest.raises(ValueError, match='clip_pca16'):
        parse_c2_experiment_id('c2r/typo_space/cc/r0')


def test_parse_unknown_comp_lists_valid():
    with pytest.raises(ValueError, match='zoom'):
        parse_c2_experiment_id('c2r/pixels_pca16/spiral/r0')


# --------------------------------------------------------------------------
# 3. angles_for
# --------------------------------------------------------------------------

def test_angles_for_named_grids():
    np.testing.assert_array_equal(angles_for('cc'), ANGLES_CIRCLE)
    np.testing.assert_array_equal(angles_for('circle'), ANGLES_CIRCLE)
    np.testing.assert_array_equal(angles_for('arc'), ANGLES_ARC180)
    np.testing.assert_array_equal(angles_for('arc180'), ANGLES_ARC180)


@pytest.mark.parametrize('alpha_max', list(SCAN_ALPHA_MAX))
def test_angles_for_scan_grid_shape(alpha_max):
    angles = angles_for(alpha_max)
    assert len(angles) == N_GRID_VALUES == 18
    assert angles[0] == 0.0
    assert np.all(angles < alpha_max)
    assert np.all(np.diff(angles) > 0)


def test_angles_for_a360_equals_circle_grid():
    np.testing.assert_allclose(angles_for(360), ANGLES_CIRCLE)
    np.testing.assert_allclose(angles_for('arc360'), ANGLES_CIRCLE)


def test_angles_for_a180_equals_arc180_grid():
    np.testing.assert_allclose(angles_for(180), ANGLES_ARC180)


@pytest.mark.parametrize('bad', ['zoom', 'arcx', 'circle0', 'cir', 0, -90, 400.0])
def test_angles_for_garbage_raises(bad):
    with pytest.raises(ValueError):
        angles_for(bad)


# --------------------------------------------------------------------------
# 4. cloud_images + конвенция сидов
# --------------------------------------------------------------------------

@pytest.fixture
def mnist_digit3():
    rng = np.random.default_rng(0)
    images = (rng.random((60, 28, 28)) * 255).astype(np.uint8)
    labels = np.full(60, DIGIT, dtype=np.int64)
    return images, labels


def test_cloud_images_shape_and_dtype(mnist_digit3):
    images, labels = mnist_digit3
    cloud = cloud_images(images, labels, 42, n_digits=3, angles=FEW_ANGLES)
    assert cloud.shape == (3 * len(FEW_ANGLES), 28, 28)
    assert cloud.dtype == np.uint8


def test_cloud_images_deterministic_by_seed(mnist_digit3):
    images, labels = mnist_digit3
    a = cloud_images(images, labels, 42, n_digits=3, angles=FEW_ANGLES)
    b = cloud_images(images, labels, 42, n_digits=3, angles=FEW_ANGLES)
    np.testing.assert_array_equal(a, b)


def test_cloud_images_different_seed_different_cloud(mnist_digit3):
    images, labels = mnist_digit3
    a = cloud_images(images, labels, 42, n_digits=3, angles=FEW_ANGLES)
    b = cloud_images(images, labels, 7, n_digits=3, angles=FEW_ANGLES)
    assert not np.array_equal(a, b)


def test_cloud_images_matches_sweep_of_selected_digits(mnist_digit3):
    images, labels = mnist_digit3
    cloud = cloud_images(images, labels, 42, n_digits=3, angles=FEW_ANGLES)
    base = select_class(images, labels, DIGIT, 3, 42)
    expected = sweep_images(base, FEW_ANGLES, rotate_images)
    np.testing.assert_array_equal(cloud, expected)


def test_seed_convention():
    assert seed_p(0) == 42
    assert seed_q(0) == 7
    assert seed_p(3) == 42 + 300
    assert seed_q(3) == 7 + 300


# --------------------------------------------------------------------------
# 5. project_pair с заглушкой raw_fn
# --------------------------------------------------------------------------

@pytest.fixture
def stub_raw_fn():
    """Заглушка сырых представлений: детерминированные матрицы из картинок."""
    def raw_fn(kind, cloud):
        cloud = np.asarray(cloud, dtype=np.float64)
        n = len(cloud)
        if kind == 'pixels':
            return cloud.reshape(n, -1)
        if kind == 'clip':
            return cloud[:, ::4, ::4].reshape(n, 49)
        if kind == 'dino':
            return cloud[:, ::2, ::2].reshape(n, 196)[:, :24]
        if kind == 'vae':
            return cloud[:, :4, :4].reshape(n, 16)
        raise ValueError(f'неизвестный kind: {kind!r}')
    return raw_fn


@pytest.fixture
def small_pair(mnist_digit3):
    images, labels = mnist_digit3
    p = cloud_images(images, labels, 42, n_digits=5, angles=FEW_ANGLES)
    q = cloud_images(images, labels, 7, n_digits=5, angles=FEW_ANGLES)
    return p, q


def test_project_pair_pixels_pca16_shapes(small_pair, stub_raw_fn):
    p, q = project_pair(small_pair[0], small_pair[1], 'pixels_pca16', stub_raw_fn)
    assert p.shape == (20, 16)
    assert q.shape == (20, 16)


@pytest.mark.parametrize('space', ['clip_pca16', 'dino_pca16', 'pixels_pca16'])
def test_project_pair_pca_fitted_on_p(small_pair, mnist_digit3, stub_raw_fn, space):
    """PCA обучена на P: P-проекция не зависит от Q, Q-проекция меняется с Q."""
    images, labels = mnist_digit3
    p_cloud, q_cloud = small_pair
    other_q = cloud_images(images, labels, 107, n_digits=5, angles=FEW_ANGLES)
    p1, q1 = project_pair(p_cloud, q_cloud, space, stub_raw_fn)
    p2, q2 = project_pair(p_cloud, other_q, space, stub_raw_fn)
    np.testing.assert_array_equal(p1, p2)
    assert not np.allclose(q1, q2)


def test_project_pair_vae_latent16_no_pca(small_pair, stub_raw_fn):
    p_cloud, q_cloud = small_pair
    p, q = project_pair(p_cloud, q_cloud, 'vae_latent16', stub_raw_fn)
    assert p.shape == (20, 16)
    np.testing.assert_array_equal(p, stub_raw_fn('vae', p_cloud))
    np.testing.assert_array_equal(q, stub_raw_fn('vae', q_cloud))


def test_project_pair_unknown_space_raises(small_pair, stub_raw_fn):
    with pytest.raises(ValueError, match='clip_raw'):
        project_pair(small_pair[0], small_pair[1], 'clip_raw', stub_raw_fn)


def test_project_pair_raw_fn_row_mismatch_raises(small_pair):
    def bad_raw_fn(kind, cloud):
        return np.zeros((3, 5))
    with pytest.raises(ValueError):
        project_pair(small_pair[0], small_pair[1], 'pixels_pca16', bad_raw_fn)


# --------------------------------------------------------------------------
# 6. pair_clouds
# --------------------------------------------------------------------------

def test_pair_clouds_cc_both_circle(mnist_digit3):
    images, labels = mnist_digit3
    parsed = parse_c2_experiment_id('c2r/pixels_pca16/cc/r0')
    p, q = pair_clouds(images, labels, parsed, 0, n_digits=3)
    assert p.shape == q.shape == (3 * 18, 28, 28)
    np.testing.assert_array_equal(
        p, cloud_images(images, labels, 42, n_digits=3, angles=ANGLES_CIRCLE))
    np.testing.assert_array_equal(
        q, cloud_images(images, labels, 7, n_digits=3, angles=ANGLES_CIRCLE))


def test_pair_clouds_arc_q_uses_arc180_grid(mnist_digit3):
    images, labels = mnist_digit3
    parsed = parse_c2_experiment_id('c2r/pixels_pca16/arc/r1')
    p, q = pair_clouds(images, labels, parsed, 1, n_digits=3)
    assert p.shape == q.shape
    np.testing.assert_array_equal(
        q, cloud_images(images, labels, seed_q(1), n_digits=3, angles=ANGLES_ARC180))
    assert not np.array_equal(p, q)


def test_pair_clouds_zoom_q_uses_scale_sweep(mnist_digit3):
    images, labels = mnist_digit3
    parsed = parse_c2_experiment_id('c2r/pixels_pca16/zoom/r2')
    p, q = pair_clouds(images, labels, parsed, 2, n_digits=3)
    assert p.shape == q.shape
    base = select_class(images, labels, DIGIT, 3, seed_q(2))
    np.testing.assert_array_equal(q, sweep_images(base, SCALES_ZOOM, zoom_images))
    assert not np.array_equal(p, q)


def test_pair_clouds_scan_a360_q_equals_circle_grid(mnist_digit3):
    images, labels = mnist_digit3
    parsed = parse_c2_experiment_id('c2s/pixels_pca16/a360/r0')
    p, q = pair_clouds(images, labels, parsed, 0, n_digits=3)
    np.testing.assert_array_equal(
        q, cloud_images(images, labels, 7, n_digits=3, angles=ANGLES_CIRCLE))


def test_pair_clouds_scan_a180_q_equals_arc180_grid(mnist_digit3):
    images, labels = mnist_digit3
    parsed = parse_c2_experiment_id('c2s/vae_latent16/a180/r1')
    _, q = pair_clouds(images, labels, parsed, 1, n_digits=3)
    np.testing.assert_array_equal(
        q, cloud_images(images, labels, seed_q(1), n_digits=3, angles=ANGLES_ARC180))


def test_pair_clouds_unknown_family_raises(mnist_digit3):
    images, labels = mnist_digit3
    with pytest.raises(ValueError):
        pair_clouds(images, labels, {'family': 'junk'}, 0)
