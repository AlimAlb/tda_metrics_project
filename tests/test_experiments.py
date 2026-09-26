"""Корреляционный анализ на синтетических таблицах: без GPU-зависимостей."""
import numpy as np
import pandas as pd
import pytest

from tda_metrics.experiments import (
    ALL_METRICS,
    group_correlation_summary,
    trajectories_correlation,
    run_experiment,
)


class _StubMetrics:
    def compute_all(self, P, Q, **kwargs):
        return {'mmd': float(np.mean(P)), 'rtd': float(np.mean(Q))}


def test_perfectly_correlated_metrics():
    df = pd.DataFrame({
        'experiment': ['e'] * 10,
        'step': range(10),
        'rtd': np.arange(10),
        'mmd': 2 * np.arange(10) + 1,
    })
    corr = trajectories_correlation(df)
    assert corr.loc['rtd', 'mmd'] == pytest.approx(1.0)
    assert corr.loc['mmd', 'rtd'] == pytest.approx(1.0)


def test_constant_trajectories_excluded():
    df = pd.DataFrame({
        'experiment': ['e'] * 10,
        'step': range(10),
        'rtd': np.arange(10),
        'mmd': 2 * np.arange(10) + 1,
        'js': 3.14,
    })
    corr = trajectories_correlation(df)
    assert np.isnan(corr.loc['js', 'js'])
    assert np.isnan(corr.loc['js', 'rtd'])
    assert corr.loc['rtd', 'mmd'] == pytest.approx(1.0)


def test_pr_inversion_affects_sign():
    df = pd.DataFrame({
        'experiment': ['e'] * 10,
        'step': range(10),
        'precision@3': 1.0 - np.arange(10) / 10,
        'mmd': np.arange(10),
    })
    corr = trajectories_correlation(df)
    assert corr.loc['precision@3', 'mmd'] == pytest.approx(1.0)


def test_group_summary_shape():
    df = pd.DataFrame({
        'experiment': ['e'] * 10,
        'step': range(10),
        'rtd': np.arange(10),
        'mmd': np.arange(10) * 2,
        'js': np.arange(10) + 5,
    })
    corr = trajectories_correlation(df)
    summary = group_correlation_summary(corr)
    assert summary.shape == (3, 3)
    assert set(summary.index) == set(summary.columns)


def test_run_experiment_constant_extra():
    df = run_experiment(
        'e', lambda s: np.zeros((5, 2)), lambda s: np.ones((5, 2)),
        3, _StubMetrics(), extra={'alpha_deg': 90.0},
    )
    assert list(df['alpha_deg']) == [90.0] * 3
    assert list(df['mmd']) == [0.0] * 3


def test_run_experiment_callable_extra():
    df = run_experiment(
        'e', lambda s: np.zeros((5, 2)), lambda s: np.ones((5, 2)),
        3, _StubMetrics(), extra=lambda s: {'alpha_deg': 30.0 * s},
    )
    assert list(df['alpha_deg']) == [0.0, 30.0, 60.0]
    assert df.shape == (3, 5)