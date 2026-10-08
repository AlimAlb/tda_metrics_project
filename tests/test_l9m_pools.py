"""Инварианты length-matched пулов L9M: бины длины, квоты, половины промптов, вложенные alpha-замены."""
import numpy as np
import pytest

from tda_metrics.l9m_pools import (
    _bin_assignment,
    _level_fill,
    build_length_matched_packet,
    length_bin_edges,
    sample_length_matched_pools,
)

POOL_KEYS_ARRAY = (
    'P', 'Q_correct', 'H', 'P_positions', 'Q_correct_positions',
    'H_positions', 'P_prompts', 'Q_prompts', 'H_prompts', 'p_lengths',
    'q_lengths', 'h_lengths', 'bin_edges',
)
POOL_KEYS_LIST = (
    'P_items', 'Q_correct_items', 'H_items', 'bin_quotas',
    'prompt_half_A', 'prompt_half_B',
)


def _make_pool_args(n_prompts=60, hall_shift=0.0, seed=11):
    """Синтетические пулы фолда: по одному correct и hallucinated пункту на промпт.

    Длины correct — целые (clip нормального шума к >= 1), hallucinated —
    те же длины со сдвигом ``hall_shift``; строки перемешаны, матрицы
    случайные (n_prompts, 3). Возвращает кортеж позиционных аргументов
    ``sample_length_matched_pools``.
    """
    rng = np.random.default_rng(seed)
    prompts = np.array([f'prompt_{i:02d}' for i in range(n_prompts)])
    base_lengths = np.maximum(np.round(rng.normal(10.0, 3.0, size=n_prompts)), 1.0)
    order_correct = rng.permutation(n_prompts)
    order_hall = rng.permutation(n_prompts)
    correct_matrix = rng.normal(size=(n_prompts, 3))
    hall_matrix = rng.normal(size=(n_prompts, 3))
    return (
        correct_matrix[order_correct],
        np.arange(n_prompts, dtype=np.int64)[order_correct],
        base_lengths[order_correct],
        prompts[order_correct],
        hall_matrix[order_hall],
        np.arange(n_prompts, 2 * n_prompts, dtype=np.int64)[order_hall],
        (base_lengths + hall_shift)[order_hall],
        prompts[order_hall],
    )


def _standard_pools(n_cloud=12, n_bins=10):
    return sample_length_matched_pools(
        *_make_pool_args(), n_cloud=n_cloud, seed_half=101,
        seed_sample=202, n_bins=n_bins)


def test_level_fill_exact_expected():
    np.testing.assert_array_equal(_level_fill(np.array([1, 5, 5]), 7), [1, 3, 3])
    np.testing.assert_array_equal(_level_fill(np.array([0, 2, 4]), 5), [0, 2, 3])
    np.testing.assert_array_equal(_level_fill(np.array([2, 3]), 0), [0, 0])
    np.testing.assert_array_equal(_level_fill(np.array([2, 3]), 5), [2, 3])


def test_level_fill_sum_and_caps():
    np.testing.assert_array_equal(
        _level_fill(np.array([3, 0, 2, 5, 1]), 8), [3, 0, 2, 2, 1])
    rng = np.random.default_rng(5)
    for _ in range(50):
        caps = rng.integers(0, 7, size=6)
        target = int(rng.integers(0, int(caps.sum()) + 1))
        quota = _level_fill(caps, target)
        assert int(quota.sum()) == target
        assert ((quota >= 0) & (quota <= caps)).all()
        uncapped = quota < caps
        if uncapped.any():
            assert int(quota[uncapped].max() - quota[uncapped].min()) <= 1


def test_level_fill_validation():
    with pytest.raises(ValueError):
        _level_fill(np.array([1, 2]), 4)
    with pytest.raises(ValueError):
        _level_fill(np.array([1, 2]), -1)


def test_bin_edges_unique_and_inf():
    lengths = [1.0] * 5 + [2.0] * 5 + [3.0] * 5
    edges = length_bin_edges(lengths, 3)
    assert edges[0] == -np.inf
    assert edges[-1] == np.inf
    assert len(edges) >= 2
    internal = edges[1:-1]
    assert np.isfinite(internal).all()
    if len(internal):
        assert (np.diff(internal) > 0).all()
    coarse = length_bin_edges(lengths, 10)
    assert len(coarse) - 1 < 10
    assert coarse[0] == -np.inf and coarse[-1] == np.inf


def test_bin_edges_degenerate_single_bin():
    np.testing.assert_array_equal(
        length_bin_edges([5.0] * 7, 4), np.array([-np.inf, np.inf]))
    np.testing.assert_array_equal(
        length_bin_edges([1.0, 2.0, 3.0], 1), np.array([-np.inf, np.inf]))


def test_determinism_same_seed():
    first = _standard_pools()
    second = _standard_pools()
    for key in POOL_KEYS_ARRAY:
        assert np.array_equal(first[key], second[key]), key
    for key in POOL_KEYS_LIST:
        assert first[key] == second[key], key
    assert first['n_bins_actual'] == second['n_bins_actual']
    assert first['n_cloud_actual'] == second['n_cloud_actual']
    assert first['diagnostics'] == second['diagnostics']


def test_different_seed_sample_changes_composition():
    first = _standard_pools()
    second = sample_length_matched_pools(
        *_make_pool_args(), n_cloud=12, seed_half=101, seed_sample=203)
    assert first['prompt_half_A'] == second['prompt_half_A']
    assert not np.array_equal(first['P_positions'], second['P_positions'])
    assert not np.array_equal(first['P'], second['P'])


def test_different_seed_half_changes_halves():
    first = _standard_pools()
    second = sample_length_matched_pools(
        *_make_pool_args(), n_cloud=12, seed_half=102, seed_sample=202)
    assert first['prompt_half_A'] != second['prompt_half_A']
    assert first['prompt_half_B'] != second['prompt_half_B']


def test_disjoint_items_and_prompts():
    pools = _standard_pools()
    assert set(pools['P_items']).isdisjoint(pools['Q_correct_items'])
    assert set(pools['P_prompts'].tolist()).isdisjoint(pools['Q_prompts'].tolist())


def test_H_from_half_B():
    pools = _standard_pools()
    h_prompts = set(pools['H_prompts'].tolist())
    assert h_prompts <= set(pools['prompt_half_B'])
    assert h_prompts.isdisjoint(pools['prompt_half_A'])


def test_exact_bin_quotas():
    pools = _standard_pools()
    edges = pools['bin_edges']
    ids = {
        'p': _bin_assignment(pools['p_lengths'], edges),
        'q': _bin_assignment(pools['q_lengths'], edges),
        'h': _bin_assignment(pools['h_lengths'], edges),
    }
    for b in range(pools['n_bins_actual']):
        counts = {int((values == b).sum()) for values in ids.values()}
        assert counts == {pools['bin_quotas'][b]}
    assert sum(pools['bin_quotas']) == pools['n_cloud_actual']
    n = pools['n_cloud_actual']
    assert pools['P'].shape == (n, 3)
    assert pools['Q_correct'].shape == (n, 3)
    assert pools['H'].shape == (n, 3)


def test_n_cloud_respected():
    pools = _standard_pools()
    assert pools['n_cloud_actual'] == 12


def test_ladder_fallback():
    args = _make_pool_args(hall_shift=40.0)
    pools = sample_length_matched_pools(*args, n_cloud=20, seed_half=101, seed_sample=202)
    assert pools['n_cloud_actual'] < 20
    assert (pools['n_bins_actual'] < 10) or (pools['n_cloud_actual'] < 20)
    assert sum(pools['bin_quotas']) == pools['n_cloud_actual']
    assert pools['P'].shape == (pools['n_cloud_actual'], 3)

    half_a = pools['prompt_half_A']
    half_b = pools['prompt_half_B']
    lengths_ac = args[2][np.isin(args[3], half_a)]
    lengths_qb = args[2][np.isin(args[3], half_b)]
    lengths_h = args[6][np.isin(args[7], half_b)]
    pooled = np.concatenate([lengths_ac, lengths_qb, lengths_h])
    totals = []
    for k in (10, 8, 6, 5, 4, 3, 2):
        edges_k = length_bin_edges(pooled, k)
        n_bins_k = len(edges_k) - 1
        counts = [
            np.bincount(_bin_assignment(side, edges_k), minlength=n_bins_k)
            for side in (lengths_ac, lengths_qb, lengths_h)
        ]
        caps_k = np.minimum(np.minimum(counts[0], counts[1]), counts[2])
        totals.append(int(caps_k.sum()))
    assert pools['n_cloud_actual'] == max(totals)

    edges = pools['bin_edges']
    ids = {
        'p': _bin_assignment(pools['p_lengths'], edges),
        'q': _bin_assignment(pools['q_lengths'], edges),
        'h': _bin_assignment(pools['h_lengths'], edges),
    }
    for b in range(pools['n_bins_actual']):
        counts = {int((values == b).sum()) for values in ids.values()}
        assert counts == {pools['bin_quotas'][b]}
    assert pools['diagnostics']['p_mean'] == pytest.approx(
        float(np.mean(pools['p_lengths'])))
    assert pools['diagnostics']['h_mean'] == pytest.approx(
        float(np.mean(pools['h_lengths'])))


def test_packet_alpha_zero():
    pools = _standard_pools()
    packet = build_length_matched_packet(pools, 0.0, 123)
    assert np.array_equal(packet['Q'], pools['Q_correct'])
    assert packet['n_hall'] == 0
    assert all(kind == 'correct' for kind, _ in packet['Q_items'])
    assert not np.array_equal(packet['Q'], packet['P'])
    np.testing.assert_array_equal(packet['q_lengths'], pools['q_lengths'])
    assert packet['P_items'] == pools['P_items']
    np.testing.assert_array_equal(packet['p_lengths'], pools['p_lengths'])
    assert packet['alpha'] == 0.0
    assert packet['n_cloud_actual'] == pools['n_cloud_actual']


def test_packet_alpha_one():
    pools = _standard_pools()
    packet = build_length_matched_packet(pools, 1.0, 123)
    assert np.array_equal(packet['Q'], pools['H'])
    assert packet['n_hall'] == pools['n_cloud_actual']
    assert all(kind == 'hallucinated' for kind, _ in packet['Q_items'])
    assert packet['Q_items'] == pools['H_items']
    np.testing.assert_array_equal(packet['q_lengths'], pools['h_lengths'])


def test_packet_nested_alphas():
    pools = _standard_pools()
    small = build_length_matched_packet(pools, 0.25, 77)
    large = build_length_matched_packet(pools, 0.75, 77)
    assert small['n_hall'] <= large['n_hall']
    for position, item in enumerate(small['Q_items']):
        if item[0] == 'hallucinated':
            assert large['Q_items'][position] == item
    hall_small = {index for kind, index in small['Q_items'] if kind == 'hallucinated'}
    hall_large = {index for kind, index in large['Q_items'] if kind == 'hallucinated'}
    assert hall_small <= hall_large


def test_packet_alpha_validation():
    pools = _standard_pools()
    with pytest.raises(ValueError):
        build_length_matched_packet(pools, -0.1, 123)
    with pytest.raises(ValueError):
        build_length_matched_packet(pools, 1.1, 123)


def test_pools_validation():
    args = _make_pool_args()
    correct_matrix, correct_index, correct_lengths, correct_prompts, hall_matrix, hall_index, hall_lengths, hall_prompts = args
    bad_prompts = hall_prompts.copy()
    bad_prompts[0] = 'prompt_missing'
    with pytest.raises(ValueError):
        sample_length_matched_pools(
            correct_matrix, correct_index, correct_lengths, correct_prompts,
            hall_matrix, hall_index, hall_lengths, bad_prompts,
            n_cloud=12, seed_half=101, seed_sample=202)
    with pytest.raises(ValueError):
        sample_length_matched_pools(
            correct_matrix, correct_index[:-1], correct_lengths, correct_prompts,
            hall_matrix, hall_index, hall_lengths, hall_prompts,
            n_cloud=12, seed_half=101, seed_sample=202)
    with pytest.raises(ValueError):
        sample_length_matched_pools(
            correct_matrix, correct_index, correct_lengths, correct_prompts,
            hall_matrix, hall_index, hall_lengths, hall_prompts,
            n_cloud=1, seed_half=101, seed_sample=202)
    with pytest.raises(ValueError):
        sample_length_matched_pools(
            correct_matrix, correct_index, correct_lengths, correct_prompts,
            hall_matrix[:, :2], hall_index, hall_lengths, hall_prompts,
            n_cloud=12, seed_half=101, seed_sample=202)


def test_no_repeats_within_side():
    pools = _standard_pools()
    for key in ('P_positions', 'Q_correct_positions', 'H_positions'):
        positions = pools[key]
        assert len(set(positions.tolist())) == len(positions)


def test_bin_assignment_edges():
    edges = np.array([-np.inf, 2.0, 5.0, np.inf])
    values = np.array([1.0, 2.0, 2.5, 5.0, 7.0, -100.0, 100.0])
    ids = _bin_assignment(values, edges)
    assert ids.dtype == np.int64
    np.testing.assert_array_equal(ids, [0, 1, 1, 2, 2, 0, 2])
