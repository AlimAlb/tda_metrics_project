"""Инварианты параметрических датасетов: тождественность, детерминизм, формы."""
import numpy as np
import pytest

from tda_metrics.datasets import (
    angles_grid,
    rotate_images,
    zoom_images,
    sweep_images,
    select_class,
    scale_grid,
)


@pytest.fixture(scope='module')
def images():
    rng = np.random.default_rng(0)
    return (rng.random((8, 28, 28)) * 255).astype(np.uint8)


def test_rotate_zero_is_identity(images):
    assert np.array_equal(rotate_images(images, 0.0), images)


def test_rotate_full_circle_roundtrip(images):
    diff = np.abs(rotate_images(images, 360.0).astype(int) - images.astype(int))
    assert diff.mean() < 8.0


def test_rotate_is_deterministic(images):
    assert np.array_equal(rotate_images(images, 37.5), rotate_images(images, 37.5))


def test_rotate_fills_corners_with_black(images):
    rotated = rotate_images(images, 45.0)
    assert rotated[:, :3, :3].max() < 5


def test_zoom_one_is_close_to_identity(images):
    diff = np.abs(zoom_images(images, 1.0).astype(int) - images.astype(int))
    assert diff.mean() < 5.0


def test_zoom_keeps_canvas(images):
    for scale in (0.6, 1.4):
        assert zoom_images(images, scale).shape == images.shape


def test_sweep_shapes_and_order(images):
    swept = sweep_images(images, [0.0, 90.0], rotate_images)
    assert swept.shape == (16, 28, 28)
    assert np.array_equal(swept[0::2], images)
    assert np.array_equal(swept[1::2], rotate_images(images, 90.0))


def test_sweep_single_value_matches_transform(images):
    assert np.array_equal(
        sweep_images(images, [37.5], rotate_images),
        rotate_images(images, 37.5),
    )


def test_select_class_filters_and_counts():
    labels = np.array([3, 5, 3, 3, 5, 3, 3, 3, 3, 3, 3, 3])
    images = np.zeros((12, 28, 28), dtype=np.uint8)
    chosen = select_class(images, labels, digit=3, n=5, seed=42)
    assert chosen.shape == (5, 28, 28)
    assert select_class(images, labels, digit=3, n=5, seed=42).shape == (5, 28, 28)
    assert select_class(images, labels, digit=3, n=100, seed=42).shape == (10, 28, 28)


def test_angles_and_scale_grids():
    assert len(angles_grid(10.0)) == 36
    assert angles_grid(10.0)[0] == 0.0
    assert angles_grid(10.0)[-1] == 350.0
    scales = scale_grid(0.6, 1.4, n=9)
    assert scales[0] == 0.6 and scales[-1] == 1.4 and len(scales) == 9