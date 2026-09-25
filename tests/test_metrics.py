"""Инварианты метрик: нули на копии, симметричность, границы, ограничения.

Требует heavy-стек (ripser++, MTopDiv, RTD, tensorflow) — запускать на Colab.
"""
import numpy as np
import pytest

from tda_metrics.metrics import TopologyMetrics
from tda_metrics.samplers import sample_gaussian, sample_ball, sample_ring


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
                           'precision@1', 'precision@3', 'precision@10',
                           'recall@1', 'recall@3', 'recall@10'}


def test_mtd_homology_keys(metrics, clouds):
    P, Q = clouds
    scores = metrics.mtd_homology(P, Q)
    assert set(scores) == {'H0', 'H1'}
    assert scores['H1'] == pytest.approx(metrics.mtd(P, Q), rel=1e-6)