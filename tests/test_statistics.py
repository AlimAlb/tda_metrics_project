"""Инварианты статистического протокола (мастер-план §5, §9.5): механика без heavy-стека.

Все проверки точные: либо hand-verified арифметика (BH, контрасты, z), либо
структурные инварианты, выполняющиеся при любом исходе генератора (целостность
групп в bootstrap, состав значений permutation-матрицы). Вероятностные
утверждения (доля +1 у честной монеты) используются только с запасом,
исключающим нарушение при любом seed (границы Хёффдинга на 10^4 бросков).
"""
import numpy as np
import pandas as pd
import pytest

from tda_metrics.statistics import (
    benjamini_hochberg,
    contrast_to_control,
    first_detected_step,
    grouped_bootstrap_indices,
    metric_contrast_table,
    paired_permutation_labels,
)

CLASSIC_P = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205]
CLASSIC_ADJ = [0.008, 0.032, 0.0672, 0.0672, 0.0672, 0.08, 8 * 0.074 / 7, 0.205]
CONTROL_DIST = [1.0, 1.1, 0.9, 1.05, 0.95]


def test_bh_classic_example():
    reject, adjusted = benjamini_hochberg(CLASSIC_P, alpha=0.05)
    assert adjusted == pytest.approx(CLASSIC_ADJ)
    assert list(reject) == [True, True, False, False, False, False, False, False]


def test_bh_four_values_hand_checked():
    reject, adjusted = benjamini_hochberg([0.005, 0.02, 0.04, 0.5], alpha=0.05)
    assert adjusted == pytest.approx([0.02, 0.04, 4 * 0.04 / 3, 0.5])
    assert list(reject) == [True, True, False, False]


def test_bh_restores_original_order():
    shuffled = [0.205, 0.001, 0.074, 0.008, 0.042, 0.06, 0.041, 0.039]
    expected = [0.205, 0.008, 8 * 0.074 / 7, 0.032, 0.0672, 0.08, 0.0672, 0.0672]
    reject, adjusted = benjamini_hochberg(shuffled, alpha=0.05)
    assert adjusted == pytest.approx(expected)
    assert list(reject) == [False, True, False, True, False, False, False, False]


def test_bh_single_pvalue_unchanged():
    reject, adjusted = benjamini_hochberg([0.037])
    assert adjusted == pytest.approx([0.037])
    assert list(reject) == [True]
    reject_strict, adjusted_strict = benjamini_hochberg([0.037], alpha=0.02)
    assert adjusted_strict == pytest.approx([0.037])
    assert list(reject_strict) == [False]
    reject_scalar, adjusted_scalar = benjamini_hochberg(0.037)
    assert adjusted_scalar == pytest.approx([0.037])
    assert list(reject_scalar) == [True]


def test_bh_adjusted_monotone_along_ranks():
    p = [0.10, 0.02, 0.30, 0.005, 0.25, 0.15, 0.02]
    _, adjusted = benjamini_hochberg(p, alpha=0.1)
    order = np.argsort(p, kind='stable')
    assert np.all(np.diff(adjusted[order]) >= 0)
    assert np.all(adjusted >= np.asarray(p))
    assert np.all(adjusted <= 1.0)


def test_bh_rejects_invalid_input():
    with pytest.raises(ValueError):
        benjamini_hochberg([0.1, 1.5])
    with pytest.raises(ValueError):
        benjamini_hochberg([0.1, -0.01])
    with pytest.raises(ValueError):
        benjamini_hochberg([0.1, np.nan])
    with pytest.raises(ValueError):
        benjamini_hochberg([0.1, 0.2], alpha=0.0)
    with pytest.raises(ValueError):
        benjamini_hochberg([0.1, 0.2], alpha=1.0)


def test_grouped_bootstrap_groups_never_split():
    group_ids = np.array([0, 0, 1, 1, 2, 3, 3, 4])
    indices = grouped_bootstrap_indices(group_ids, n_resamples=40, seed=42)
    assert len(indices) == 40
    for idx in indices:
        counts = np.bincount(idx, minlength=len(group_ids))
        for group in np.unique(group_ids):
            member_counts = counts[group_ids == group]
            assert np.all(member_counts == member_counts[0])


def test_grouped_bootstrap_deterministic():
    group_ids = [0, 0, 1, 1, 2]
    first = grouped_bootstrap_indices(group_ids, n_resamples=20, seed=7)
    second = grouped_bootstrap_indices(group_ids, n_resamples=20, seed=7)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    other = grouped_bootstrap_indices(group_ids, n_resamples=20, seed=8)
    assert any(not np.array_equal(a, b) for a, b in zip(first, other))


def test_grouped_bootstrap_size_varies_and_groups_repeat():
    group_ids = [0, 0, 0, 1, 1, 2]
    indices = grouped_bootstrap_indices(group_ids, n_resamples=30, seed=11)
    sizes = [len(idx) for idx in indices]
    assert len(set(sizes)) >= 2
    assert max(sizes) > len(group_ids)


def test_grouped_bootstrap_single_group_covers_all():
    indices = grouped_bootstrap_indices([5, 5, 5], n_resamples=3, seed=0)
    assert all(np.array_equal(idx, np.array([0, 1, 2])) for idx in indices)


def test_grouped_bootstrap_zero_resamples_and_invalid():
    assert grouped_bootstrap_indices([0, 0, 1], n_resamples=0, seed=0) == []
    with pytest.raises(ValueError):
        grouped_bootstrap_indices([], n_resamples=3, seed=0)
    with pytest.raises(ValueError):
        grouped_bootstrap_indices([0, 1], n_resamples=-1, seed=0)


def test_paired_labels_shape_and_values():
    labels = paired_permutation_labels(n_pairs=12, n_permutations=25, seed=3)
    assert labels.shape == (25, 12)
    assert np.issubdtype(labels.dtype, np.integer)
    assert np.all(np.isin(labels, (-1, 1)))
    assert paired_permutation_labels(0, 5, seed=0).shape == (5, 0)


def test_paired_labels_deterministic():
    first = paired_permutation_labels(50, 30, seed=5)
    second = paired_permutation_labels(50, 30, seed=5)
    assert np.array_equal(first, second)
    other = paired_permutation_labels(50, 30, seed=6)
    assert not np.array_equal(first, other)


def test_paired_labels_fair_and_independent_per_pair():
    labels = paired_permutation_labels(n_pairs=50, n_permutations=200, seed=1)
    share_plus = float(np.mean(labels == 1))
    assert 0.45 <= share_plus <= 0.55
    assert np.all((labels == 1).sum(axis=0) > 0)
    assert np.all((labels == -1).sum(axis=0) > 0)
    assert not np.array_equal(labels[:, 0], labels[:, 1])


def test_paired_labels_invalid():
    with pytest.raises(ValueError):
        paired_permutation_labels(-1, 5, seed=0)
    with pytest.raises(ValueError):
        paired_permutation_labels(5, -2, seed=0)


def test_contrast_distance_expected_sign():
    delta, z = contrast_to_control(1.2, CONTROL_DIST, kind='distance')
    assert delta == pytest.approx(0.2)
    assert z == pytest.approx(0.2 / np.sqrt(0.00625))
    delta_low, z_low = contrast_to_control(0.8, CONTROL_DIST, kind='distance')
    assert delta_low == pytest.approx(-0.2)
    assert z_low < 0.0


def test_contrast_similarity_sign_flipped():
    control = [0.9, 0.8, 0.85]
    delta, z = contrast_to_control(0.6, control, kind='similarity')
    assert delta == pytest.approx(0.25)
    assert z == pytest.approx(5.0)
    delta_better, z_better = contrast_to_control(0.95, control, kind='similarity')
    assert delta_better == pytest.approx(-0.1)
    assert z_better < 0.0


def test_contrast_zero_sigma_eps_guard():
    delta_same, z_same = contrast_to_control(5.0, [5.0, 5.0])
    assert delta_same == 0.0
    assert z_same == 0.0
    delta_up, z_up = contrast_to_control(6.0, [5.0, 5.0])
    assert delta_up == pytest.approx(1.0)
    assert np.isfinite(z_up)
    assert z_up == pytest.approx(1e12)


def test_contrast_vectorized_over_values():
    deltas, zs = contrast_to_control(np.array([1.2, 0.8, 1.0]), CONTROL_DIST)
    assert deltas == pytest.approx([0.2, -0.2, 0.0])
    assert zs[0] > 0.0 and zs[1] < 0.0 and zs[2] == pytest.approx(0.0)
    delta_scalar, _ = contrast_to_control(1.2, CONTROL_DIST)
    assert isinstance(delta_scalar, float)


def test_contrast_nan_observed_propagates():
    deltas, zs = contrast_to_control(np.array([1.2, np.nan]), CONTROL_DIST)
    assert deltas[0] == pytest.approx(0.2)
    assert np.isnan(deltas[1]) and np.isnan(zs[1])


def test_contrast_invalid_input():
    with pytest.raises(ValueError):
        contrast_to_control(1.0, [1.0, np.nan])
    with pytest.raises(ValueError):
        contrast_to_control(1.0, [1.0])
    with pytest.raises(ValueError):
        contrast_to_control(1.0, CONTROL_DIST, kind='wrong')
    with pytest.raises(ValueError):
        contrast_to_control(np.zeros((2, 2)), CONTROL_DIST)


STEPS = [0, 1, 2, 3, 4]
CI_LOW = [-0.3, -0.2, 0.1, 0.5, 0.8]
CI_HIGH = [0.5, 0.6, 0.9, 1.3, 1.6]


def test_first_detected_known_switchpoint():
    deltas = [0.1, 0.2, 0.5, 0.9, 1.2]
    result = first_detected_step(STEPS, deltas, CI_LOW, CI_HIGH)
    assert result['index'] == 2
    assert result['step'] == 2
    assert result['delta'] == pytest.approx(0.5)
    assert result['direction_share'] is None
    assert result['alpha'] is None


def test_first_detected_never():
    result = first_detected_step(STEPS, [0.1] * 5, [-0.5] * 5, [0.5] * 5)
    assert result == {'step': None, 'index': None, 'delta': None,
                      'alpha': None, 'direction_share': None}


def test_first_detected_wrong_direction_not_counted():
    result = first_detected_step(
        [0, 1, 2], [0.1, 0.1, 0.3],
        ci_low=[-1.2, -0.9, 0.05], ci_high=[-0.4, -0.2, 0.9],
    )
    assert result['index'] == 2
    all_negative = first_detected_step(
        [0, 1], [-1.0, -1.0], ci_low=[-1.5, -1.4], ci_high=[-0.5, -0.4],
    )
    assert all_negative['step'] is None
    assert all_negative['index'] is None


def test_first_detected_repeats_direction_share_and_alpha():
    repeats = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.1, 0.2, 0.5, 0.7, 0.9],
        [-0.1, 0.0, -0.1, 0.2, 0.3],
        [0.2, 0.3, 0.7, 0.8, 1.0],
        [0.0, 0.1, 0.2, 0.3, 0.4],
    ])
    alpha_grid = [0.0, 0.05, 0.10, 0.15, 0.20]
    result = first_detected_step(STEPS, repeats, CI_LOW, CI_HIGH, alpha_grid=alpha_grid)
    assert result['index'] == 2
    assert result['alpha'] == pytest.approx(0.10)
    assert repeats[:, 2] == pytest.approx([0.3, 0.5, -0.1, 0.7, 0.2])
    assert result['delta'] == pytest.approx(0.32)
    assert result['direction_share'] == pytest.approx(0.8)


def test_first_detected_input_validation():
    with pytest.raises(ValueError):
        first_detected_step([0, 1], [0.1], [0.0, 0.1], [0.2, 0.3])
    with pytest.raises(ValueError):
        first_detected_step([0, 1], [0.1, 0.2], [0.1], [0.2, 0.3])
    with pytest.raises(ValueError):
        first_detected_step([0, 1], [0.1, 0.2], [0.1, 0.2], [0.3, 0.4], alpha_grid=[0.0])
    with pytest.raises(ValueError):
        first_detected_step([0, 1], np.zeros((2, 2, 2)), [0.1, 0.2], [0.3, 0.4])


def _contrast_frame():
    return pd.DataFrame({
        'is_control': [True, True, True, False, False],
        'mmd': [1.0, 1.2, 0.8, 1.5, 0.9],
        'precision@3': [0.9, 0.8, 0.85, 0.6, 0.86],
    })


def test_metric_contrast_table_hand_checked():
    frame = _contrast_frame()
    table = metric_contrast_table(frame, frame['is_control'], ['mmd', 'precision@3'])
    assert list(table.index) == [3, 4]
    assert list(table.columns) == ['mmd_delta', 'mmd_z', 'precision@3_delta', 'precision@3_z']
    assert table.loc[3, 'mmd_delta'] == pytest.approx(0.5)
    assert table.loc[3, 'mmd_z'] == pytest.approx(2.5)
    assert table.loc[3, 'precision@3_delta'] == pytest.approx(0.25)
    assert table.loc[3, 'precision@3_z'] == pytest.approx(5.0)
    assert table.loc[4, 'mmd_delta'] == pytest.approx(-0.1)
    assert table.loc[4, 'precision@3_delta'] == pytest.approx(-0.01)


def test_metric_contrast_table_kind_map_override():
    frame = _contrast_frame()
    table = metric_contrast_table(
        frame, frame['is_control'], ['mmd', 'precision@3'],
        kind_map={'mmd': 'similarity', 'precision@3': 'distance'},
    )
    assert table.loc[3, 'mmd_delta'] == pytest.approx(-0.5)
    assert table.loc[3, 'precision@3_delta'] == pytest.approx(-0.25)


def test_metric_contrast_table_nan_control_isolated():
    frame = pd.DataFrame({
        'is_control': [True, True, True, False],
        'mmd': [1.0, 1.2, 0.8, 1.5],
        'js': [np.nan, np.nan, np.nan, 0.4],
    })
    table = metric_contrast_table(frame, frame['is_control'], ['mmd', 'js'])
    assert table.loc[3, 'mmd_delta'] == pytest.approx(0.5)
    assert np.isnan(table.loc[3, 'js_delta'])
    assert np.isnan(table.loc[3, 'js_z'])


def test_metric_contrast_table_nan_observed():
    frame = pd.DataFrame({
        'is_control': [True, True, True, False, False],
        'mmd': [1.0, 1.2, 0.8, np.nan, 1.5],
    })
    table = metric_contrast_table(frame, frame['is_control'], ['mmd'])
    assert np.isnan(table.loc[3, 'mmd_delta'])
    assert np.isnan(table.loc[3, 'mmd_z'])
    assert table.loc[4, 'mmd_delta'] == pytest.approx(0.5)


def test_metric_contrast_table_errors():
    frame = _contrast_frame()
    with pytest.raises(ValueError):
        metric_contrast_table(frame, [False] * 5, ['mmd'])
    with pytest.raises(ValueError):
        metric_contrast_table(frame, frame['is_control'], ['rtd'])
    with pytest.raises(ValueError):
        metric_contrast_table(frame, frame['is_control'], ['mmd'], kind_map={'mmd': 'bad'})
    with pytest.raises(ValueError):
        metric_contrast_table(frame, [True, True, False], ['mmd'])
