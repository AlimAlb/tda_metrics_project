"""Инварианты эмпирического power-анализа (мастер-план §5.6): чистая механика.

Heavy-стек не нужен: metrics_fn — мок (словарь синтетических значений),
сэмплеры — чистые numpy-функции без TDA. Вероятностные утверждения
(same-law power ~ alpha) проверяются диапазоном с фиксированным seed, а не
хрупким равенством — тот же принцип, что в test_statistics.py.
"""
import numpy as np
import pandas as pd
import pytest

from tda_metrics.power import (
    aggregate_power,
    empirical_power,
    power_curve,
    select_min_n,
)


# ---------- power_curve ----------

def test_power_curve_separated_samples_power_one():
    control = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 1.03, 0.97, 1.0])
    effect = np.full(20, 100.0)
    table = power_curve(effect, control, alpha_grid=(0.05, 0.10, 0.20))
    assert np.all(table['power'] == 1.0)


def test_power_curve_better_values_power_zero():
    control = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 1.03, 0.97, 1.0])
    effect = np.full(20, -100.0)
    table = power_curve(effect, control, alpha_grid=(0.05, 0.10, 0.20), alternative='greater')
    assert np.all(table['power'] == 0.0)


def test_power_curve_same_law_power_near_alpha():
    rng = np.random.default_rng(12345)
    control = rng.normal(loc=0.0, scale=1.0, size=2000)
    effect = rng.normal(loc=0.0, scale=1.0, size=2000)
    table = power_curve(effect, control, alpha_grid=(0.05, 0.10, 0.20))
    row05 = table[table['alpha'] == 0.05].iloc[0]
    row10 = table[table['alpha'] == 0.10].iloc[0]
    row20 = table[table['alpha'] == 0.20].iloc[0]
    assert 0.02 <= row05['power'] <= 0.09
    assert 0.06 <= row10['power'] <= 0.15
    assert 0.14 <= row20['power'] <= 0.27


def test_power_curve_monotone_in_alpha():
    rng = np.random.default_rng(7)
    control = rng.normal(loc=0.0, scale=1.0, size=500)
    effect = rng.normal(loc=0.8, scale=1.0, size=500)
    table = power_curve(effect, control, alpha_grid=(0.05, 0.10, 0.20))
    powers = table.sort_values('alpha')['power'].to_numpy()
    assert np.all(np.diff(powers) >= 0.0)


def test_power_curve_twosided_catches_both_directions():
    control = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 1.03, 0.97, 1.0])
    high = np.full(10, 100.0)
    low = np.full(10, -100.0)
    table_high = power_curve(high, control, alpha_grid=(0.10,), alternative='twosided')
    table_low = power_curve(low, control, alpha_grid=(0.10,), alternative='twosided')
    assert table_high.loc[0, 'power'] == 1.0
    assert table_low.loc[0, 'power'] == 1.0
    lower, upper = table_high.loc[0, 'threshold']
    assert lower < upper


def test_power_curve_z_mean_effect_hand_checked():
    control = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    effect = np.array([6.0, 9.0])
    table = power_curve(effect, control, alpha_grid=(0.1,))
    mean_control = 3.0
    sd_control = np.std(control, ddof=1)
    expected_z = np.mean([(6.0 - mean_control) / sd_control, (9.0 - mean_control) / sd_control])
    assert table.loc[0, 'mean_control'] == pytest.approx(mean_control)
    assert table.loc[0, 'sd_control'] == pytest.approx(sd_control)
    assert table.loc[0, 'mean_effect'] == pytest.approx(7.5)
    assert table.loc[0, 'z_mean_effect'] == pytest.approx(expected_z)


def test_power_curve_z_mean_effect_zero_sigma():
    control = np.array([5.0, 5.0, 5.0])
    same = power_curve(np.array([5.0, 5.0]), control, alpha_grid=(0.2,))
    assert same.loc[0, 'z_mean_effect'] == 0.0
    different = power_curve(np.array([5.0, 6.0]), control, alpha_grid=(0.2,))
    assert np.isinf(different.loc[0, 'z_mean_effect'])


def test_power_curve_rows_cover_full_alpha_grid():
    control = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    effect = np.array([10.0, 11.0])
    alpha_grid = (0.05, 0.10, 0.20)
    table = power_curve(effect, control, alpha_grid=alpha_grid)
    assert list(table['alpha']) == list(alpha_grid)
    assert set(table.columns) == {
        'alpha', 'power', 'threshold', 'n_control', 'n_effect',
        'mean_control', 'sd_control', 'mean_effect', 'z_mean_effect',
    }
    assert np.all(table['n_control'] == len(control))
    assert np.all(table['n_effect'] == len(effect))


def test_power_curve_invalid_input():
    with pytest.raises(ValueError):
        power_curve([1.0], [1.0], alpha_grid=(0.05,))
    with pytest.raises(ValueError):
        power_curve([1.0], [1.0, np.nan], alpha_grid=(0.05,))
    with pytest.raises(ValueError):
        power_curve([1.0], [1.0, 2.0], alpha_grid=())
    with pytest.raises(ValueError):
        power_curve([1.0], [1.0, 2.0], alpha_grid=(0.0,))
    with pytest.raises(ValueError):
        power_curve([1.0], [1.0, 2.0], alpha_grid=(1.0,))
    with pytest.raises(ValueError):
        power_curve([1.0], [1.0, 2.0], alpha_grid=(0.05,), alternative='bad')


def test_power_curve_inf_effect_is_detect_nan_is_not():
    control = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    effect = np.array([np.inf, np.nan, 2.5])
    table = power_curve(effect, control, alpha_grid=(0.05, 0.10, 0.20))
    assert np.allclose(table['power'].to_numpy(), 1.0 / 3.0)
    twosided = power_curve(
        np.array([-np.inf, np.inf, np.nan, 2.5]), control,
        alpha_grid=(0.10,), alternative='twosided',
    )
    assert np.isclose(twosided.loc[0, 'power'], 0.5)


# ---------- empirical_power ----------

def _mock_control_sampler(n, seed):
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(n, 2))
    q = rng.normal(size=(n, 2))
    return p, q


def _mock_effect_sampler(n, seed):
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(n, 2))
    q = rng.normal(loc=5.0, size=(n, 2))
    return p, q


def _mock_metrics_fn(p, q):
    return {
        'mean_diff': float(np.mean(q) - np.mean(p)),
        'size_sum': float(len(p) + len(q)),
    }


def test_empirical_power_shape_and_columns():
    table = empirical_power(
        _mock_effect_sampler, _mock_control_sampler, _mock_metrics_fn,
        n_grid=(10, 20), n_control=3, n_effect=4, seed=0,
    )
    assert list(table.columns) == ['n', 'kind', 'repeat', 'metric', 'value']
    n_metrics = 2
    expected_rows = len((10, 20)) * (3 + 4) * n_metrics
    assert len(table) == expected_rows
    assert set(table['kind'].unique()) == {'control', 'effect'}
    assert set(table['metric'].unique()) == {'mean_diff', 'size_sum'}


def test_empirical_power_respects_repeat_counts():
    table = empirical_power(
        _mock_effect_sampler, _mock_control_sampler, _mock_metrics_fn,
        n_grid=(15,), n_control=5, n_effect=2, seed=1,
    )
    control_repeats = table[table['kind'] == 'control']['repeat'].unique()
    effect_repeats = table[table['kind'] == 'effect']['repeat'].unique()
    assert sorted(control_repeats) == [0, 1, 2, 3, 4]
    assert sorted(effect_repeats) == [0, 1]


def test_empirical_power_deterministic():
    first = empirical_power(
        _mock_effect_sampler, _mock_control_sampler, _mock_metrics_fn,
        n_grid=(10, 20), n_control=3, n_effect=3, seed=42,
    )
    second = empirical_power(
        _mock_effect_sampler, _mock_control_sampler, _mock_metrics_fn,
        n_grid=(10, 20), n_control=3, n_effect=3, seed=42,
    )
    pd.testing.assert_frame_equal(first, second)
    third = empirical_power(
        _mock_effect_sampler, _mock_control_sampler, _mock_metrics_fn,
        n_grid=(10, 20), n_control=3, n_effect=3, seed=43,
    )
    assert not first['value'].equals(third['value'])


def test_empirical_power_invalid_input():
    with pytest.raises(ValueError):
        empirical_power(
            _mock_effect_sampler, _mock_control_sampler, _mock_metrics_fn,
            n_grid=(10,), n_control=0, n_effect=3, seed=0,
        )
    with pytest.raises(ValueError):
        empirical_power(
            _mock_effect_sampler, _mock_control_sampler, _mock_metrics_fn,
            n_grid=(10,), n_control=3, n_effect=3, seed=0, alpha_grid=(),
        )


# ---------- aggregate_power ----------

def test_aggregate_power_matches_manual_power_curve():
    rng = np.random.default_rng(99)
    control_values = rng.normal(size=20)
    effect_values = rng.normal(loc=3.0, size=20)
    records = []
    for i, value in enumerate(control_values):
        records.append({'n': 100, 'kind': 'control', 'repeat': i, 'metric': 'm', 'value': value})
    for i, value in enumerate(effect_values):
        records.append({'n': 100, 'kind': 'effect', 'repeat': i, 'metric': 'm', 'value': value})
    frame = pd.DataFrame.from_records(records)

    aggregated = aggregate_power(frame, alpha_grid=(0.05, 0.10, 0.20))
    manual = power_curve(effect_values, control_values, alpha_grid=(0.05, 0.10, 0.20))

    assert list(aggregated['power']) == list(manual['power'])
    assert list(aggregated['n']) == [100] * 3
    assert list(aggregated['metric']) == ['m'] * 3


def test_aggregate_power_target_met_column():
    records = []
    for i in range(10):
        records.append({'n': 50, 'kind': 'control', 'repeat': i, 'metric': 'm', 'value': 1.0 + 0.01 * i})
    for i in range(10):
        records.append({'n': 50, 'kind': 'effect', 'repeat': i, 'metric': 'm', 'value': 100.0})
    frame = pd.DataFrame.from_records(records)
    aggregated = aggregate_power(frame, alpha_grid=(0.05, 0.10, 0.20))
    assert np.all(aggregated['target_met'])


def test_aggregate_power_multiple_n_and_metrics():
    records = []
    for n in (50, 100):
        for metric, loc in (('a', 0.0), ('b', 5.0)):
            for i in range(5):
                records.append({'n': n, 'kind': 'control', 'repeat': i, 'metric': metric, 'value': float(i)})
            for i in range(5):
                records.append({'n': n, 'kind': 'effect', 'repeat': i, 'metric': metric, 'value': loc + float(i)})
    frame = pd.DataFrame.from_records(records)
    aggregated = aggregate_power(frame, alpha_grid=(0.05, 0.10, 0.20))
    assert set(aggregated['n'].unique()) == {50, 100}
    assert set(aggregated['metric'].unique()) == {'a', 'b'}
    assert len(aggregated) == 2 * 2 * 3


def test_aggregate_power_missing_columns():
    with pytest.raises(ValueError):
        aggregate_power(pd.DataFrame({'n': [1], 'metric': ['m'], 'value': [1.0]}))


def test_aggregate_power_raises_when_group_has_too_few_controls():
    single_control = pd.DataFrame.from_records([
        {'n': 100, 'kind': 'control', 'repeat': 0, 'metric': 'm', 'value': 1.0},
        {'n': 100, 'kind': 'effect', 'repeat': 0, 'metric': 'm', 'value': 2.0},
        {'n': 100, 'kind': 'effect', 'repeat': 1, 'metric': 'm', 'value': 3.0},
    ])
    with pytest.raises(ValueError):
        aggregate_power(single_control, alpha_grid=(0.05,))
    no_control = pd.DataFrame.from_records([
        {'n': 100, 'kind': 'effect', 'repeat': 0, 'metric': 'm', 'value': 2.0},
    ])
    with pytest.raises(ValueError):
        aggregate_power(no_control, alpha_grid=(0.05,))


# ---------- select_min_n ----------

def test_select_min_n_picks_minimal_n():
    frame = pd.DataFrame.from_records([
        {'n': 100, 'metric': 'm', 'alpha': 0.05, 'power': 0.3},
        {'n': 250, 'metric': 'm', 'alpha': 0.05, 'power': 0.6},
        {'n': 500, 'metric': 'm', 'alpha': 0.05, 'power': 0.85},
        {'n': 1000, 'metric': 'm', 'alpha': 0.05, 'power': 0.95},
    ])
    result = select_min_n(frame, target_power=0.8, alpha=0.05)
    assert result == {'m': 500}


def test_select_min_n_none_when_target_never_met():
    frame = pd.DataFrame.from_records([
        {'n': 100, 'metric': 'm', 'alpha': 0.05, 'power': 0.1},
        {'n': 250, 'metric': 'm', 'alpha': 0.05, 'power': 0.2},
    ])
    result = select_min_n(frame, target_power=0.8, alpha=0.05)
    assert result == {'m': None}


def test_select_min_n_multiple_metrics_mixed():
    frame = pd.DataFrame.from_records([
        {'n': 100, 'metric': 'good', 'alpha': 0.05, 'power': 0.9},
        {'n': 100, 'metric': 'bad', 'alpha': 0.05, 'power': 0.1},
        {'n': 250, 'metric': 'bad', 'alpha': 0.05, 'power': 0.3},
    ])
    result = select_min_n(frame, target_power=0.8, alpha=0.05)
    assert result == {'good': 100, 'bad': None}


def test_select_min_n_respects_alpha_filter():
    frame = pd.DataFrame.from_records([
        {'n': 100, 'metric': 'm', 'alpha': 0.05, 'power': 0.3},
        {'n': 100, 'metric': 'm', 'alpha': 0.20, 'power': 0.9},
    ])
    assert select_min_n(frame, target_power=0.8, alpha=0.05) == {'m': None}
    assert select_min_n(frame, target_power=0.8, alpha=0.20) == {'m': 100}


def test_select_min_n_nondefault_alpha_with_float_noise():
    frame = pd.DataFrame.from_records([
        {'n': 100, 'metric': 'm', 'alpha': 0.10 + 1e-12, 'power': 0.85},
        {'n': 250, 'metric': 'm', 'alpha': 0.20 + 1e-12, 'power': 0.95},
    ])
    assert select_min_n(frame, target_power=0.8, alpha=0.10) == {'m': 100}
    assert select_min_n(frame, target_power=0.8, alpha=0.20) == {'m': 250}
