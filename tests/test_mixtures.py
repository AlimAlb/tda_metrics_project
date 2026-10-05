"""Инварианты строителей смесей mode dropping × invention (мастер-план §9.3, §Часть I/S3)."""
import numpy as np
import pytest

from tda_metrics.mixtures import (
    ModeReference,
    make_mode_reference,
    make_alpha_replacement,
    nested_alphas,
    sample_q_dropping_invention,
    sample_reference,
)

GRID = [0.0, 0.25, 0.5, 0.75]


def test_centers_on_circle_and_zero_mean():
    ref = make_mode_reference(n_modes=8, radius=8.0, spread=1.0)
    np.testing.assert_allclose(np.linalg.norm(ref.centers, axis=1), 8.0, rtol=1e-12)
    np.testing.assert_allclose(ref.centers.mean(axis=0), 0.0, atol=1e-12)
    assert ref.n_modes == 8


def test_reference_requires_valid_params():
    with pytest.raises(ValueError):
        make_mode_reference(n_modes=1)
    with pytest.raises(ValueError):
        make_mode_reference(n_modes=4, radius=-1.0)


@pytest.mark.parametrize('d', GRID)
@pytest.mark.parametrize('i', GRID)
def test_q_size_constant_over_grid(d, i):
    ref = make_mode_reference()
    q = sample_q_dropping_invention(ref, n=320, seed=7, drop_fraction=d, invent_fraction=i)
    assert q.shape == (320, 2)


def test_reference_equals_zero_grid():
    ref = make_mode_reference()
    n = 400
    same = sample_reference(ref, n, seed=42)
    zero = sample_q_dropping_invention(ref, n, seed=42, drop_fraction=0.0, invent_fraction=0.0)
    np.testing.assert_array_equal(same, zero)


def test_dropped_modes_absent():
    ref = make_mode_reference()
    q = sample_q_dropping_invention(ref, n=800, seed=7, drop_fraction=0.75)
    kept = ref.centers[:2]
    dropped = ref.centers[2:]
    dist_kept = np.linalg.norm(q[:, None, :] - kept[None, :, :], axis=2).min(axis=1)
    dist_dropped = np.linalg.norm(q[:, None, :] - dropped[None, :, :], axis=2).min(axis=1)
    assert np.mean(dist_kept < dist_dropped) > 0.95


def test_invented_modes_symmetric_no_mean_shift():
    ref = make_mode_reference()
    n = 800
    base = sample_q_dropping_invention(ref, n, seed=42, drop_fraction=0.0, invent_fraction=0.0)
    for i in (0.25, 0.5, 0.75):
        invented = sample_q_dropping_invention(ref, n, seed=42, drop_fraction=0.0, invent_fraction=i)
        np.testing.assert_allclose(invented.mean(axis=0), base.mean(axis=0), atol=0.3)


def _noise_sum_of_squares(q, active_centers):
    dist = np.linalg.norm(q[:, None, :] - active_centers[None, :, :], axis=2)
    return float((dist.min(axis=1) ** 2).sum())


def test_common_random_numbers_noise_invariant():
    ref = make_mode_reference()
    n = 800
    expected = 2.0 * ref.spread ** 2 * n
    values = {}
    for d in GRID:
        for i in GRID:
            q = sample_q_dropping_invention(ref, n, seed=7, drop_fraction=d, invent_fraction=i)
            n_kept = ref.n_modes - int(round(d * ref.n_modes))
            n_pairs = int(round(i * ref.n_modes / 2.0))
            kept = ref.centers[:n_kept]
            pair_angles = np.pi * (np.arange(n_pairs) + 0.5) / max(n_pairs, 1) + np.pi / 4.0
            directions = np.stack([np.cos(pair_angles), np.sin(pair_angles)], axis=1)
            invented = 14.0 * np.concatenate([directions, -directions], axis=0) if n_pairs else np.zeros((0, 2))
            active = np.concatenate([kept, invented], axis=0)
            values[(d, i)] = _noise_sum_of_squares(q, active)
            np.testing.assert_allclose(values[(d, i)], expected, rtol=0.2)
    assert max(values.values()) - min(values.values()) < 0.05 * expected


def test_determinism_by_seed():
    ref = make_mode_reference()
    a = sample_q_dropping_invention(ref, n=200, seed=42, drop_fraction=0.25)
    b = sample_q_dropping_invention(ref, n=200, seed=42, drop_fraction=0.25)
    c = sample_q_dropping_invention(ref, n=200, seed=43, drop_fraction=0.25)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_invalid_fractions_rejected():
    ref = make_mode_reference()
    with pytest.raises(ValueError):
        sample_q_dropping_invention(ref, n=100, seed=1, drop_fraction=1.0)
    with pytest.raises(ValueError):
        sample_q_dropping_invention(ref, n=100, seed=1, drop_fraction=-0.1)
    with pytest.raises(ValueError):
        sample_q_dropping_invention(ref, n=100, seed=1, invent_fraction=1.5)


def test_invent_offset_must_clear_reference():
    ref = make_mode_reference(radius=8.0, spread=1.0)
    with pytest.raises(ValueError):
        sample_q_dropping_invention(ref, n=100, seed=1, invent_fraction=0.5, invent_offset=9.0)


def test_alpha_replacement_extremes():
    correct = np.arange(20).reshape(10, 2).astype(float)
    hallucinated = -correct
    permutation = np.array([3, 1, 4, 7, 0, 9, 2, 5, 8, 6])
    np.testing.assert_array_equal(
        make_alpha_replacement(correct, hallucinated, 0.0, permutation), correct
    )
    np.testing.assert_array_equal(
        make_alpha_replacement(correct, hallucinated, 1.0, permutation), hallucinated
    )


def test_alpha_replacement_exact_count_and_nestedness():
    correct = np.zeros((10, 1))
    hallucinated = np.ones((10, 1))
    permutation = np.array([5, 2, 8, 0, 9, 3, 7, 1, 6, 4])
    q_half = make_alpha_replacement(correct, hallucinated, 0.5, permutation)
    assert int(q_half.sum()) == 5
    masks = nested_alphas(permutation, [0.2, 0.5, 0.8])
    assert masks[0.2].sum() == 2
    assert masks[0.5].sum() == 5
    assert masks[0.8].sum() == 8
    assert np.all(masks[0.2] <= masks[0.5])
    assert np.all(masks[0.5] <= masks[0.8])
    replaced = np.zeros(10, dtype=bool)
    replaced[permutation[:5]] = True
    np.testing.assert_array_equal(masks[0.5], replaced)


def test_alpha_replacement_validates_inputs():
    correct = np.zeros((5, 1))
    hallucinated = np.ones((5, 1))
    with pytest.raises(ValueError):
        make_alpha_replacement(correct, np.ones((4, 1)), 0.5, np.arange(5))
    with pytest.raises(ValueError):
        make_alpha_replacement(correct, hallucinated, 0.5, np.arange(4))
    with pytest.raises(ValueError):
        make_alpha_replacement(correct, hallucinated, 1.5, np.arange(5))


def test_mode_reference_is_immutable_dataclass():
    ref = make_mode_reference()
    with pytest.raises(AttributeError):
        ref.spread = 2.0
    assert isinstance(ref, ModeReference)