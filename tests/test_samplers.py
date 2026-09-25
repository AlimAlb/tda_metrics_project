"""Инварианты сэмплеров: моменты, границы, детерминизм."""
import numpy as np
import pytest

from tda_metrics.samplers import (
    sample_gaussian,
    sample_uniform_cube,
    sample_ball,
    sample_ring,
    sample_gaussian_mixture,
)

N = 20000
ATOL = 0.05


def test_gaussian_moments():
    points = sample_gaussian(N, seed=0)
    assert points.shape == (N, 2)
    assert np.allclose(points.mean(axis=0), 0.0, atol=ATOL)
    assert np.allclose(np.cov(points.T), np.eye(2), atol=ATOL)


def test_uniform_cube_bounds_and_mean():
    points = sample_uniform_cube(N, seed=0)
    assert points.min() >= -0.5 and points.max() <= 0.5
    assert np.allclose(points.mean(axis=0), 0.0, atol=ATOL)


def test_ball_uniform_by_volume():
    points = sample_ball(N, radius=1.0, seed=0)
    radii = np.linalg.norm(points, axis=1)
    assert radii.max() <= 1.0
    share_inside_half = (radii < 0.5).mean()
    assert abs(share_inside_half - 0.25) < 0.02


def test_ring_radius_stats():
    radius, thickness = 5.0, 1.5
    points = sample_ring(N, radius=radius, thickness=thickness, seed=0)
    radii = np.linalg.norm(points, axis=1)
    assert abs(radii.mean() - radius) < ATOL
    assert abs(radii.std() - thickness / 4.0) < ATOL


@pytest.mark.parametrize('radius', [0.0, 0.5, 1.0, 1.3])
@pytest.mark.parametrize('n_components', [4, 8, 16])
def test_mixture_moments_preserved(radius, n_components):
    points = sample_gaussian_mixture(N, radius=radius, n_components=n_components, seed=0)
    assert np.allclose(points.mean(axis=0), 0.0, atol=ATOL)
    assert np.allclose(np.cov(points.T), np.eye(2), atol=ATOL)


def test_mixture_constraints():
    with pytest.raises(ValueError):
        sample_gaussian_mixture(N, radius=1.5)
    with pytest.raises(ValueError):
        sample_gaussian_mixture(N, radius=0.5, n_components=2)
    with pytest.raises(ValueError):
        sample_gaussian_mixture(N, radius=0.5, n_components=1)


def test_determinism():
    a = sample_gaussian_mixture(100, radius=1.0, seed=42)
    b = sample_gaussian_mixture(100, radius=1.0, seed=42)
    assert np.array_equal(a, b)