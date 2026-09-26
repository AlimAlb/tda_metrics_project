"""Инварианты PCA (SVD): дисперсии, детерминизм, линейность фит-матрицы."""
import numpy as np
import pytest

from tda_metrics.reduction import pca_fit, pca_transform
from tda_metrics.samplers import sample_ring


@pytest.fixture(scope='module')
def wide_ring():
    points = sample_ring(2000, radius=1.0, thickness=0.3, seed=42)
    return np.hstack([points, np.zeros((2000, 8))])


def test_explained_ratio_sorted_and_bounded(wide_ring):
    _, components, ratio = pca_fit(wide_ring, n_components=2)
    assert ratio.shape == (10,)
    assert np.all(np.diff(ratio) <= 1e-12)
    assert ratio.sum() == pytest.approx(1.0)
    assert components.shape == (2, 10)


def test_two_components_capture_ring_plane(wide_ring):
    _, info_total = pca_transform(wide_ring, n_components=2)
    assert info_total['total'] > 0.95


def test_projection_of_fit_data_is_centered(wide_ring):
    projection, _ = pca_transform(wide_ring, n_components=2)
    assert np.allclose(projection.mean(axis=0), 0.0, atol=1e-10)


def test_pca_is_deterministic(wide_ring):
    first, _ = pca_transform(wide_ring, n_components=3)
    second, _ = pca_transform(wide_ring, n_components=3)
    assert np.array_equal(first, second)


def test_fixed_model_is_linear(wide_ring):
    model = pca_fit(wide_ring, n_components=4)
    shift = np.full(10, 2.0)
    base, _ = pca_transform(wide_ring, model=model)
    shifted, _ = pca_transform(wide_ring + shift, model=model)
    expected_shift = shift @ model[1].T
    assert np.allclose(shifted - base, expected_shift, atol=1e-8)


def test_components_bounds():
    X = np.random.default_rng(0).normal(size=(50, 8))
    with pytest.raises(ValueError):
        pca_fit(X, n_components=0)
    with pytest.raises(ValueError):
        pca_fit(X, n_components=51)