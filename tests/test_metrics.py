"""Инварианты метрик: нули на копии, симметричность, границы, ограничения.

Требует heavy-стек (ripser++, MTopDiv, RTD, tensorflow) — запускать на Colab.
"""
import numpy as np
import pytest

from tda_metrics.metrics import TopologyMetrics
from tda_metrics.samplers import sample_gaussian, sample_ball, sample_ring
from tda_metrics.experiments import permutation_test, bootstrap_ci


@pytest.fixture(scope='module')
def metrics():
    return TopologyMetrics(seed=42, device='cpu')


@pytest.fixture(scope='module')
def clouds():
    P = sample_gaussian(500, seed=42)
    Q = sample_ball(500, radius=2.0, seed=7)
    return P, Q


def test_copy_is_zero_and_perfect(metrics):
    P = sample_gaussian(500, seed=42)
    copy = P.copy()
    assert metrics.mtd(P, copy) == 0.0
    assert metrics.rtd(P, copy) == 0.0
    assert metrics.mmd(P, copy) == 0.0
    assert metrics.frechet_distance(P, copy) == 0.0
    assert metrics.js_divergence(P, copy) == 0.0
    pr = metrics.improved_precision_recall(P, copy)
    assert all(v == 1.0 for v in pr.values())


def test_symmetry_of_statistical_metrics(metrics, clouds):
    P, Q = clouds
    assert metrics.mmd(P, Q) == pytest.approx(metrics.mmd(Q, P), abs=1e-9)
    assert metrics.frechet_distance(P, Q) == pytest.approx(metrics.frechet_distance(Q, P), abs=1e-9)
    assert metrics.js_divergence(P, Q) == pytest.approx(metrics.js_divergence(Q, P), abs=1e-9)


def test_rtd_is_symmetric_and_requires_equal_sizes(metrics, clouds):
    P, Q = clouds
    assert metrics.rtd(P, Q) == pytest.approx(metrics.rtd(Q, P))
    with pytest.raises(ValueError):
        metrics.rtd(P, Q[:100])
    with pytest.raises(ValueError):
        metrics.rtd(P, Q, trials=0)


def test_mtd_is_deterministic(metrics):
    P = sample_ring(300, radius=5.0, thickness=1.5, seed=42)
    Q = sample_ring(300, radius=5.0, thickness=1.5, seed=7)
    assert metrics.mtd(P, Q) == metrics.mtd(P, Q)


def test_js_bounds(metrics, clouds):
    P, Q = clouds
    same_law = sample_gaussian(500, seed=7)
    assert 0.0 <= metrics.js_divergence(P, Q) <= 1.0
    assert metrics.js_divergence(P, Q) > metrics.js_divergence(P, same_law)


def test_pr_between_0_and_1(metrics, clouds):
    P, Q = clouds
    pr = metrics.improved_precision_recall(P, Q)
    assert all(0.0 <= v <= 1.0 for v in pr.values())
    assert set(pr) == {'precision@1', 'precision@3', 'precision@10',
                       'recall@1', 'recall@3', 'recall@10'}


def test_compute_all_keys(metrics, clouds):
    P, Q = clouds
    result = metrics.compute_all(P, Q)
    assert set(result) == {'mtd_PQ', 'mtd_QP', 'rtd', 'mmd', 'frechet', 'js',
                           'ntd_PQ', 'ntd_QP',
                           'precision@1', 'precision@3', 'precision@10',
                           'recall@1', 'recall@3', 'recall@10'}


def test_mtd_homology_keys(metrics, clouds):
    P, Q = clouds
    scores = metrics.mtd_homology(P, Q)
    assert set(scores) == {'H0', 'H1'}
    assert scores['H1'] == pytest.approx(metrics.mtd(P, Q), rel=1e-6)


def test_compute_all_skip(metrics, clouds):
    P, Q = clouds
    result = metrics.compute_all(P, Q, skip=('rtd', 'ntd'))
    assert 'rtd' not in result and 'ntd_PQ' not in result and 'ntd_QP' not in result
    assert 'mmd' in result and 'mtd_PQ' in result


def test_permutation_detects_shift(metrics):
    P = sample_gaussian(200, seed=42)
    Q = sample_gaussian(200, mean=(3.0, 0.0), seed=7)
    result = permutation_test(P, Q, metrics, n_permutations=8, seed=42)
    assert result.loc['mmd', 'z'] > 3.0
    assert result.loc['mmd', 'p_value'] <= 1.0 / 9.0
    assert result.loc['precision@3', 'p_value'] <= 1.0 / 9.0


def test_permutation_same_law_is_not_significant(metrics):
    P = sample_gaussian(200, seed=42)
    Q = sample_gaussian(200, seed=7)
    result = permutation_test(P, Q, metrics, n_permutations=8, seed=42)
    assert result.loc['mmd', 'p_value'] > 0.05
    assert 0.0 <= result['p_value'].min() <= result['p_value'].max() <= 1.0


def test_bootstrap_ci_brackets_mean_and_skips_rtd(metrics):
    P = sample_gaussian(200, seed=42)
    Q = sample_ball(200, radius=2.0, seed=7)
    summary = bootstrap_ci(P, Q, metrics, n_resamples=8, seed=42)
    assert 'rtd' not in summary.index and 'ntd_PQ' not in summary.index
    assert (summary['ci_2.5%'] <= summary['среднее']).all()
    assert (summary['среднее'] <= summary['ci_97.5%']).all()
    assert (summary['ширина'] >= 0.0).all()


def test_mtd0_zero_for_exact_subset(metrics):
    P = sample_gaussian(200, seed=42)
    assert metrics.mtd0(P[:50], P) == pytest.approx(0.0, abs=1e-12)


def test_mtd0_consistent_with_ripser_h0(metrics, clouds):
    """Кросс-чек двух независимых реализаций H0: scipy single-linkage (mtd0)
    против ripser++ (mtd_homology, dim=1 возвращает H0-сумму)."""
    P, Q = clouds
    assert metrics.mtd0(P, Q) * len(P) == pytest.approx(metrics.mtd_homology(P, Q)['H0'], rel=1e-4)


def test_ntd_same_law_near_one(metrics):
    P = sample_gaussian(500, seed=42)
    Q = sample_gaussian(500, seed=7)
    for value in (metrics.ntd(P, Q, n_repeats=30), metrics.ntd(Q, P, n_repeats=30)):
        assert 0.5 < value < 2.0


def test_ntd_model_inside_data(metrics):
    wide = sample_gaussian(500, cov=np.eye(2) * 4.0, seed=42)
    narrow = sample_gaussian(500, cov=np.eye(2) * 0.25, seed=7)
    assert metrics.ntd(narrow, wide) < 1.0
    assert metrics.ntd(wide, narrow) > 1.0


def test_ntd_mode_invention_hits_precision_direction(metrics):
    Y = sample_gaussian(500, cov=np.eye(2) * 4.0, seed=99)
    X = np.vstack([
        sample_gaussian(450, cov=np.eye(2) * 4.0, seed=42),
        sample_gaussian(50, mean=(12.0, 0.0), cov=np.eye(2) * 0.5, seed=7),
    ])
    precision_like = metrics.ntd(X, Y)
    recall_like = metrics.ntd(Y, X)
    assert precision_like > 1.0
    assert recall_like < precision_like


def test_ntd_deterministic(metrics):
    P = sample_gaussian(300, seed=42)
    Q = sample_gaussian(300, seed=7)
    assert metrics.ntd(P, Q) == metrics.ntd(P, Q)