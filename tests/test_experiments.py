"""Корреляционный анализ на синтетических таблицах: без GPU-зависимостей."""
import numpy as np
import pandas as pd
import pytest

from tda_metrics.experiments import (
    ALL_METRICS,
    group_correlation_summary,
    trajectories_correlation,
)


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