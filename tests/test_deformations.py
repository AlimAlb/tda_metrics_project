"""Инварианты деформаций облаков из samplers.py (S1: карта инвариантов)."""
import numpy as np
import pytest

from tda_metrics.samplers import (
    reflect_cloud,
    rotate_cloud,
    scale_cloud,
    stretch_cloud,
    translate_cloud,
)


def _pairwise(X):
    diff = X[:, None, :] - X[None, :, :]
    return np.linalg.norm(diff, axis=-1)


@pytest.fixture
def cloud():
    rng = np.random.default_rng(11)
    return rng.normal(size=(40, 2)) * 3.0


@pytest.mark.parametrize('angle', [0.0, 37.0, 90.0, 143.5, 180.0, 270.0])
def test_rotation_is_isometry(cloud, angle):
    rotated = rotate_cloud(cloud, angle)
    np.testing.assert_allclose(_pairwise(rotated), _pairwise(cloud), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('axis', ['x', 'y'])
def test_reflection_is_isometry(cloud, axis):
    reflected = reflect_cloud(cloud, axis)
    np.testing.assert_allclose(_pairwise(reflected), _pairwise(cloud), rtol=1e-10, atol=1e-10)


def test_reflection_flips_expected_coordinate(cloud):
    np.testing.assert_allclose(reflect_cloud(cloud, 'x')[:, 0], cloud[:, 0])
    np.testing.assert_allclose(reflect_cloud(cloud, 'x')[:, 1], -cloud[:, 1])
    np.testing.assert_allclose(reflect_cloud(cloud, 'y')[:, 1], cloud[:, 1])
    np.testing.assert_allclose(reflect_cloud(cloud, 'y')[:, 0], -cloud[:, 0])


def test_translation_preserves_internal_distances(cloud):
    shift = np.array([5.0, -7.25])
    moved = translate_cloud(cloud, shift)
    np.testing.assert_allclose(_pairwise(moved), _pairwise(cloud), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(moved, cloud + shift)


def test_uniform_scaling_scales_distances(cloud):
    factor = 2.5
    scaled = scale_cloud(cloud, factor)
    np.testing.assert_allclose(_pairwise(scaled), factor * _pairwise(cloud), rtol=1e-12)


def test_rotation_zero_and_scale_one_are_identity(cloud):
    np.testing.assert_allclose(rotate_cloud(cloud, 0.0), cloud, atol=1e-12)
    np.testing.assert_allclose(scale_cloud(cloud, 1.0), cloud, atol=1e-14)


def test_stretch_changes_distances(cloud):
    stretched = stretch_cloud(cloud, 2.0, 1.0)
    assert not np.allclose(_pairwise(stretched), _pairwise(cloud))
    np.testing.assert_allclose(stretched[:, 0], 2.0 * cloud[:, 0])
    np.testing.assert_allclose(stretched[:, 1], cloud[:, 1])


def test_deformations_validate_arguments(cloud):
    with pytest.raises(ValueError):
        translate_cloud(cloud, (1.0, 2.0, 3.0))
    with pytest.raises(ValueError):
        reflect_cloud(cloud, 'z')
    with pytest.raises(ValueError):
        scale_cloud(cloud, 0.0)
    with pytest.raises(ValueError):
        scale_cloud(cloud, -1.0)
    with pytest.raises(ValueError):
        stretch_cloud(cloud, 1.0, 0.0)