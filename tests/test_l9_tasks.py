"""Инварианты l9_tasks: фолды, пакеты, confidence-признаки, задачи A/B.

Всё на numpy/pandas/sklearn/scipy без GPU и сети. Пакеты — на «маркерных»
матрицах (строка кодирует свой глобальный item-индекс), поэтому состав
P_items/Q_items проверяется поэлементно против самих матриц. Задачи A/B
— на синтетике с признаком, идеально восстанавливающим alpha: доля
пакетов с alpha=0 соблюдена по фолдам (GroupKFold видит оба класса в
каждом train-разрезе), проверяются крайние значения метрик и структура
результатов.
"""
import numpy as np
import pandas as pd
import pytest

from tda_metrics.l9_tasks import (
    FEATURE_GROUPS,
    build_packet,
    evaluate_detection,
    evaluate_regression,
    packet_confidence_features,
    prompt_folds,
)

N_POOL = 12
D = 3
N_CLOUD = 10
FOLD_ALPHAS = (0.0, 0.0, 0.05, 0.1, 0.15, 0.25, 0.5, 0.5, 0.75, 1.0)


def marker_pools():
    """Пулы фолда: строка i = маркер глобального индекса (correct: i, hallucinated: 100+i)."""
    correct_rows = np.tile(np.arange(N_POOL, dtype=float).reshape(-1, 1), (1, D))
    hall_rows = np.tile((100.0 + np.arange(N_POOL, dtype=float)).reshape(-1, 1), (1, D))
    correct_index = np.arange(N_POOL)
    hall_index = 100 + np.arange(N_POOL)
    return correct_rows, correct_index, hall_rows, hall_index


# ---------- фолды ----------

def test_prompt_folds_partition_determinism_order():
    ids = [f'p{i:02d}' for i in range(25)]
    folds = prompt_folds(ids, n_folds=5, seed=500)
    assert len(folds) == 5
    assert all(len(fold) == 5 for fold in folds)
    assert sorted(pid for fold in folds for pid in fold) == sorted(ids)
    assert prompt_folds(ids, n_folds=5, seed=500) == folds
    assert all(fold == sorted(fold) for fold in folds)
    assert prompt_folds(ids, n_folds=5, seed=501) != folds


def test_prompt_folds_uneven_split_covers_all():
    ids = [f'p{i:02d}' for i in range(23)]
    folds = prompt_folds(ids, n_folds=5, seed=500)
    assert sorted(len(fold) for fold in folds) == [4, 4, 5, 5, 5]
    assert sorted(pid for fold in folds for pid in fold) == sorted(ids)


def test_prompt_folds_deduplicates_and_validates():
    folds = prompt_folds(['a', 'a', 'b', 'b', 'c', 'c'], n_folds=2)
    assert sorted(pid for fold in folds for pid in fold) == ['a', 'b', 'c']
    assert all(len(fold) >= 1 for fold in folds)
    with pytest.raises(ValueError):
        prompt_folds(['a', 'b'], n_folds=1)
    with pytest.raises(ValueError):
        prompt_folds(['a', 'b', 'c'], n_folds=5)


# ---------- пакеты ----------

def test_build_packet_alpha_zero_is_pure_correct():
    packet = build_packet(*marker_pools(), alpha=0.0, n_cloud=N_CLOUD,
                          seed_p=11, seed_h=22, seed_mix=33)
    assert packet['n_hall'] == 0
    assert np.array_equal(packet['Q'], packet['P'])
    assert all(kind == 'correct' for kind, _ in packet['Q_items'])
    assert packet['P'].shape == (N_CLOUD, D) and packet['Q'].shape == (N_CLOUD, D)
    assert len(packet['P_items']) == len(packet['Q_items']) == N_CLOUD


def test_build_packet_alpha_one_is_pure_hallucinated():
    packet = build_packet(*marker_pools(), alpha=1.0, n_cloud=N_CLOUD,
                          seed_p=11, seed_h=22, seed_mix=33)
    assert packet['n_hall'] == N_CLOUD
    assert all(kind == 'hallucinated' for kind, _ in packet['Q_items'])
    assert all(kind == 'correct' for kind, _ in packet['P_items'])


def test_build_packet_replacement_count_and_items_match_matrices():
    packet = build_packet(*marker_pools(), alpha=0.6, n_cloud=N_CLOUD,
                          seed_p=11, seed_h=22, seed_mix=33)
    assert packet['n_hall'] == 6
    hall_positions = [position for position, (kind, _) in enumerate(packet['Q_items'])
                      if kind == 'hallucinated']
    assert len(hall_positions) == 6
    assert packet['Q'][hall_positions].min() >= 100.0
    for position, (kind, index) in enumerate(packet['P_items']):
        assert kind == 'correct'
        assert packet['P'][position, 0] == float(index)
        assert 0 <= index < N_POOL
    for position, (kind, index) in enumerate(packet['Q_items']):
        assert packet['Q'][position, 0] == float(index)
        if kind == 'correct':
            assert 0 <= index < N_POOL
        else:
            assert 100 <= index < 100 + N_POOL


def test_build_packet_nested_alphas_share_basis():
    p03 = build_packet(*marker_pools(), alpha=0.3, n_cloud=N_CLOUD,
                       seed_p=11, seed_h=22, seed_mix=33)
    p06 = build_packet(*marker_pools(), alpha=0.6, n_cloud=N_CLOUD,
                       seed_p=11, seed_h=22, seed_mix=33)
    assert p03['n_hall'] == 3 and p06['n_hall'] == 6
    assert np.array_equal(p03['P'], p06['P'])
    replaced03 = {position for position, (kind, _) in enumerate(p03['Q_items'])
                 if kind == 'hallucinated'}
    replaced06 = {position for position, (kind, _) in enumerate(p06['Q_items'])
                 if kind == 'hallucinated'}
    assert replaced03 <= replaced06
    for position in replaced03:
        assert p03['Q_items'][position] == p06['Q_items'][position]
        assert np.array_equal(p03['Q'][position], p06['Q'][position])


def test_build_packet_pool_too_small_and_index_mismatch_raise():
    correct_rows, correct_index, hall_rows, hall_index = marker_pools()
    with pytest.raises(ValueError):
        build_packet(correct_rows, correct_index, hall_rows, hall_index,
                     alpha=0.5, n_cloud=N_POOL + 1, seed_p=1, seed_h=2, seed_mix=3)
    with pytest.raises(ValueError):
        build_packet(correct_rows, correct_index[:5], hall_rows, hall_index,
                     alpha=0.5, n_cloud=4, seed_p=1, seed_h=2, seed_mix=3)


# ---------- confidence-признаки ----------

def test_packet_confidence_features_manual_aggregates():
    n_items = 12
    token = np.arange(n_items, dtype=float)
    item_scores = pd.DataFrame({
        'nll': token * 0.5,
        'ppl': np.exp(token * 0.5),
        'entropy': 2.0 + token * 0.25,
        'answer_length': 10.0 + token,
    })
    packet = {
        'P_items': [('correct', index) for index in (0, 3, 6, 9)],
        'Q_items': [('correct', 1), ('hallucinated', 5), ('correct', 7),
                    ('hallucinated', 11)],
    }
    features = packet_confidence_features(item_scores, packet)
    q_positions = [1, 5, 7, 11]
    p_positions = [0, 3, 6, 9]
    assert features['q_nll_mean'] == pytest.approx(float((token[q_positions] * 0.5).mean()))
    assert features['q_nll_std'] == pytest.approx(float((token[q_positions] * 0.5).std(ddof=0)))
    assert features['q_entropy_mean'] == pytest.approx(
        float((2.0 + token[q_positions] * 0.25).mean()))
    assert features['q_len_mean'] == pytest.approx(float((10.0 + token[q_positions]).mean()))
    assert features['p_nll_mean'] == pytest.approx(float((token[p_positions] * 0.5).mean()))
    assert features['p_entropy_mean'] == pytest.approx(
        float((2.0 + token[p_positions] * 0.25).mean()))
    assert features['p_len_mean'] == pytest.approx(float((10.0 + token[p_positions]).mean()))


# ---------- задачи A/B ----------

def make_packet_table(seed=0):
    """50 пакетов: 5 фолдов x 10 (2 alpha=0 + 8 alpha>0), признаки f_alpha/f_noise."""
    rng = np.random.default_rng(seed)
    rows = []
    counter = 0
    for fold in range(5):
        for alpha in FOLD_ALPHAS:
            rows.append({
                'packet_id': f'pk{counter}',
                'alpha': alpha,
                'fold': fold,
                'f_alpha': alpha,
                'f_noise': float(rng.normal()),
            })
            counter += 1
    return pd.DataFrame(rows)


def test_evaluate_detection_perfect_and_random_features():
    table = make_packet_table()
    groups = {'topology': ['f_alpha'], 'llm_confidence': ['f_noise'],
              'combined': ['f_alpha']}
    result = evaluate_detection(table, feature_groups=groups)
    assert set(result) == {'topology', 'llm_confidence', 'combined',
                          'per_alpha_accuracy', 'alpha0_false_positive_rate'}
    assert set(result['topology']) == {'auroc', 'auprc', 'balanced_accuracy'}
    assert result['topology']['auroc'] == pytest.approx(1.0)
    assert result['topology']['auprc'] == pytest.approx(1.0)
    assert result['combined']['auroc'] == pytest.approx(1.0)
    assert 0.0 <= result['llm_confidence']['auroc'] <= 1.0
    assert set(result['per_alpha_accuracy']) == {0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0}
    assert all(0.0 <= rate <= 1.0 for rate in result['per_alpha_accuracy'].values())
    assert 0.0 <= result['alpha0_false_positive_rate'] <= 1.0


def test_evaluate_regression_recovers_alpha():
    table = make_packet_table()
    groups = {'topology': ['f_alpha'], 'combined': ['f_alpha']}
    result = evaluate_regression(table, feature_groups=groups)
    assert set(result) == {'topology', 'combined'}
    assert set(result['topology']) == {'mae', 'r2', 'spearman'}
    assert result['combined']['mae'] < 0.01
    assert result['combined']['r2'] > 0.99
    assert result['combined']['spearman'] == pytest.approx(1.0)
    assert set(result['combined']['per_alpha_mae']) == set(table['alpha'].unique())
    assert all(0.0 <= mae <= 0.05 for mae in result['combined']['per_alpha_mae'].values())


def test_evaluate_default_groups_full_table_structure():
    table = make_packet_table()
    rng = np.random.default_rng(7)
    for feature in FEATURE_GROUPS['combined']:
        table[feature] = table['alpha'] + rng.normal(scale=0.01, size=len(table))
    detection = evaluate_detection(table)
    assert set(detection) == set(FEATURE_GROUPS) | {'per_alpha_accuracy',
                                                   'alpha0_false_positive_rate'}
    for name in FEATURE_GROUPS:
        if name == 'combined':
            continue
        assert set(detection[name]) == {'auroc', 'auprc', 'balanced_accuracy'}
        assert 0.0 <= detection[name]['auroc'] <= 1.0
    assert set(detection['combined']) == {'auroc', 'auprc', 'balanced_accuracy'}
    regression = evaluate_regression(table)
    assert set(regression) == set(FEATURE_GROUPS)
    for name in FEATURE_GROUPS:
        if name == 'combined':
            continue
        assert set(regression[name]) == {'mae', 'r2', 'spearman'}
    assert set(regression['combined']) == {'mae', 'r2', 'spearman', 'per_alpha_mae'}


# ---------- группы признаков ----------

def test_feature_groups_union_and_disjointness():
    group_names = ['topology', 'coverage', 'statistics', 'llm_confidence']
    all_features = [feature for name in group_names for feature in FEATURE_GROUPS[name]]
    assert len(all_features) == len(set(all_features))
    assert set(all_features) == set(FEATURE_GROUPS['combined'])
    assert set(FEATURE_GROUPS) == set(group_names + ['combined'])
