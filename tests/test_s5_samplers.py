"""Инварианты сэмплеров S5: антиподальный drop, invention-замена, моментно-согласованная пара."""
import numpy as np
import pytest

from tda_metrics.mixtures import make_mode_reference
from tda_metrics.s5_samplers import (
    make_matched_pair_s5c,
    moment_diagnostics,
    population_moments,
    sample_from_population,
    sample_s5a,
    sample_s5b,
)

ref = make_mode_reference(n_modes=8, radius=8.0, spread=1.0)
EXTERNAL = np.array([[14.0, 0.0], [-14.0, 0.0]])


def _mode_counts(points, centers):
    dist = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
    return np.bincount(dist.argmin(axis=1), minlength=centers.shape[0])


def _min_dist(points, centers):
    dist = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
    return dist.min(axis=1)


def test_s5a_symmetry_and_counts():
    q, info = sample_s5a(ref, 480, 7, [(0, 4)])
    assert q.shape == (480, 2)
    np.testing.assert_allclose(q.mean(axis=0), 0.0, atol=1.0)
    assert _min_dist(q, ref.centers[[0, 4]]).min() > 3.0
    counts = _mode_counts(q, ref.centers[[1, 2, 3, 5, 6, 7]])
    assert counts.sum() == 480
    assert counts.max() - counts.min() <= 1
    assert info['dropped_pairs'] == [(0, 4)]
    assert info['n_modes_kept'] == 6
    assert info['dropped_mass'] == pytest.approx(0.25)

    q3, info3 = sample_s5a(ref, 480, 7, [(0, 4), (1, 5), (2, 6)])
    assert info3['dropped_mass'] == pytest.approx(0.75)
    assert info3['n_modes_kept'] == 2
    counts3 = _mode_counts(q3, ref.centers[[3, 7]])
    assert counts3.sum() == 480
    assert counts3.max() - counts3.min() <= 1


def test_s5a_validation():
    with pytest.raises(ValueError):
        sample_s5a(ref, 480, 7, [(0, 1)])
    with pytest.raises(ValueError):
        sample_s5a(ref, 480, 7, [(0, 4), (0, 4)])
    with pytest.raises(ValueError):
        sample_s5a(ref, 480, 7, [(0, 4), (4, 0)])
    with pytest.raises(ValueError):
        sample_s5a(ref, 480, 7, [(0, 4), (1, 5), (2, 6), (3, 7)])


def test_s5a_determinism():
    a = sample_s5a(ref, 240, 42, [(0, 4)])
    b = sample_s5a(ref, 240, 42, [(0, 4)])
    c = sample_s5a(ref, 240, 43, [(0, 4)])
    np.testing.assert_array_equal(a[0], b[0])
    assert not np.array_equal(a[0], c[0])


def test_s5b_exact_replacement():
    q, mask, info = sample_s5b(ref, 480, 7, 60)
    assert q.shape == (480, 2)
    assert mask.dtype == bool
    assert mask.sum() == 60
    assert info['n_replaced'] == 60
    assert info['alpha_realized'] == pytest.approx(0.125)
    assert info['invent_offset'] == pytest.approx(14.0)

    assert _min_dist(q[mask], EXTERNAL).max() < 5.0
    assert _min_dist(q[~mask], ref.centers).max() < 5.0
    ext_counts = _mode_counts(q[mask], EXTERNAL)
    assert ext_counts.tolist() == [30, 30]
    base_counts = _mode_counts(q[~mask], ref.centers)
    assert base_counts.sum() == 420
    assert base_counts.max() - base_counts.min() <= 1
    assert 52 <= base_counts.min() and base_counts.max() <= 53

    q0, mask0, info0 = sample_s5b(ref, 480, 7, 0)
    assert q0.shape == (480, 2)
    assert mask0.sum() == 0
    assert info0['alpha_realized'] == 0.0
    assert _min_dist(q0, ref.centers).max() < 5.0

    qn, maskn, infon = sample_s5b(ref, 480, 7, 480)
    assert maskn.sum() == 480
    assert infon['alpha_realized'] == pytest.approx(1.0)
    assert _min_dist(qn, EXTERNAL).max() < 5.0


def test_s5b_validation():
    with pytest.raises(ValueError):
        sample_s5b(ref, 480, 7, -1)
    with pytest.raises(ValueError):
        sample_s5b(ref, 480, 7, 481)
    with pytest.raises(ValueError):
        sample_s5b(ref, 480, 7, 60, invent_offset=10.0)


def test_s5c_population_moments_match():
    pop_p, pop_q = make_matched_pair_s5c()
    mean_p, cov_p = population_moments(pop_p)
    mean_q, cov_q = population_moments(pop_q)
    np.testing.assert_allclose(mean_p, 0.0, atol=1e-12)
    np.testing.assert_allclose(mean_q, 0.0, atol=1e-12)
    np.testing.assert_allclose(cov_p, cov_q, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cov_p, 33.0 * np.eye(2), rtol=1e-12, atol=1e-12)
    assert pop_p['mode_centers'].shape == (8, 2)
    assert pop_q['mode_centers'].shape == (4, 2)


def test_s5c_sample_moments_close():
    pop_p, pop_q = make_matched_pair_s5c()
    p = sample_from_population(pop_p, 4000, 42)
    q = sample_from_population(pop_q, 4000, 7)
    d = moment_diagnostics(p, q)
    assert d['mean_diff'] < 1.0
    assert d['cov_diff'] < 5.0


def test_s5c_moment_diagnostics_detects_shift():
    p = np.random.default_rng(0).normal(0.0, 1.0, size=(2000, 2))
    q = np.random.default_rng(1).normal(0.0, 1.0, size=(2000, 2)) + np.array([2.0, 0.0])
    d = moment_diagnostics(p, q)
    assert d['mean_diff'] == pytest.approx(2.0, abs=0.2)
    assert d['mean_diff'] > 1.0


def test_sample_from_population_determinism():
    pop = dict(mode_centers=ref.centers, spread=1.0)
    a = sample_from_population(pop, 200, 42)
    b = sample_from_population(pop, 200, 42)
    c = sample_from_population(pop, 200, 43)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
