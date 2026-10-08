"""Пакетная задача L9: пакеты ответов с долей галлюцинаций и признаки групп (мастер-план §L9).

Конструкция пакетов: dev prompt_id режутся на фолды, из пула каждого фолда
детерминированными сидами выбираются две непересекающиеся correct-выборки
P и Q_correct, а также hallucinated-строки (их глобальные item-индексы
кэша сохраняются в составе пакета). Q строится вложенной заменой точек
``Q_correct`` через ``make_alpha_replacement`` — пакеты с общими сидами и
разными alpha имеют общий базис и вложенные замены. Поэтому alpha=0 —
честный same-law контроль, а не тривиальное сравнение P с самой собой.
Признаки пакета —
метрики ``compute_all`` (группы topology/coverage/statistics) и агрегаты
скоринга уверенности (группа llm_confidence). Задачи: A — логистическая
регрессия обнаружения ``alpha > 0`` (AUROC, AUPRC, balanced accuracy),
B — ridge-оценка доли ``alpha`` (MAE, R2, Spearman); обе — GroupKFold по
фолдам, сложные модели запрещены (оценивается информативность признаков,
а не мощность классификатора).
"""
import numpy as np
import pandas as pd

from tda_metrics.mixtures import make_alpha_replacement

__all__ = [
    'FEATURE_GROUPS',
    'prompt_folds',
    'build_packet',
    'packet_confidence_features',
    'evaluate_detection',
    'evaluate_regression',
]

N_SPLITS = 5

FEATURE_GROUPS = {
    'topology': ['mtd_PQ', 'mtd_QP', 'ntd_PQ', 'ntd_QP', 'rtd'],
    'coverage': ['precision@1', 'recall@1', 'precision@3', 'recall@3',
                 'precision@10', 'recall@10'],
    'statistics': ['mmd', 'frechet', 'js'],
    'llm_confidence': ['q_nll_mean', 'q_nll_std', 'q_entropy_mean', 'q_len_mean',
                       'p_nll_mean', 'p_entropy_mean', 'p_len_mean'],
    'combined': [],
}
FEATURE_GROUPS['combined'] = (
    FEATURE_GROUPS['topology'] + FEATURE_GROUPS['coverage']
    + FEATURE_GROUPS['statistics'] + FEATURE_GROUPS['llm_confidence']
)


def prompt_folds(prompt_ids, n_folds=5, seed=500):
    """Детерминированное разрезание prompt_id на n_folds почти равных фолдов.

    Уникальные id переставляются ``rng(seed).permutation`` и режутся на
    ``n_folds`` частей (первые получают на один элемент больше); каждый
    фолд сортирован изнутри. Возвращает список отсортированных списков;
    один и тот же набор и seed дают одно и то же разбиение. ValueError
    при ``n_folds < 2`` или меньшем числе уникальных id.
    """
    unique = sorted(set(prompt_ids))
    if n_folds < 2:
        raise ValueError(f'n_folds должен быть >= 2: {n_folds}')
    if len(unique) < n_folds:
        raise ValueError(
            f'уникальных prompt_id ({len(unique)}) меньше n_folds ({n_folds})')
    order = np.random.default_rng(seed).permutation(len(unique))
    shuffled = [unique[position] for position in order]
    return [sorted(part) for part in np.array_split(shuffled, n_folds)]


def build_packet(correct_matrix, correct_index, hall_matrix, hall_index,
                 alpha, n_cloud, seed_p, seed_h, seed_mix):
    """Пакет (P, Q_alpha) фолда: выборки пула, вложенная замена, состав.

    ``correct_matrix``/``hall_matrix`` — матрицы пула фолда (n_pool, d);
    ``correct_index``/``hall_index`` — глобальные item-индексы этих строк
    (позиции пунктов в полном кэше, 0..n_items-1) той же длины. Выборки:
    ``idx_p`` и ``idx_q`` последовательно выбираются без возвращения из
    correct-пула одним генератором ``rng(seed_p)``; поэтому P и Q_correct
    непересекающиеся независимые подвыборки одного распределения.
    ``idx_h`` выбирается от ``seed_h``; ``permutation =
    rng(seed_mix).permutation(n_cloud)``; ``Q = make_alpha_replacement(
    Q_correct, H, alpha, permutation)`` — при общих сидах замены соседних
    alpha вложены. Состав: ``P_items``/``Q_items`` — построчные кортежи
    (kind, глобальный item-индекс); позиция pos получает
    ('hallucinated', hall_index[idx_h[pos]]), если pos входит в первые
    ``round(alpha*n_cloud)`` позиций permutation, иначе ('correct',
    correct_index[idx_q[pos]]); ``n_hall = round(alpha*n_cloud)``. При
    alpha=0 Q — независимая чистая correct-выборка (same-law), при
    alpha=1 — hallucinated. ValueError при correct-пуле меньше
    ``2 * n_cloud``, hallucinated-пуле меньше ``n_cloud``, несовпадении
    длин матриц с их индексами или alpha вне [0, 1].
    """
    correct_matrix = np.asarray(correct_matrix)
    hall_matrix = np.asarray(hall_matrix)
    correct_index = np.asarray(correct_index, dtype=np.int64)
    hall_index = np.asarray(hall_index, dtype=np.int64)
    if len(correct_matrix) != len(correct_index):
        raise ValueError(
            f'correct_matrix ({len(correct_matrix)}) и correct_index '
            f'({len(correct_index)}) разной длины')
    if len(hall_matrix) != len(hall_index):
        raise ValueError(
            f'hall_matrix ({len(hall_matrix)}) и hall_index '
            f'({len(hall_index)}) разной длины')
    if len(correct_matrix) < 2 * n_cloud:
        raise ValueError(
            f'пул correct ({len(correct_matrix)}) меньше 2*n_cloud={2 * n_cloud}; '
            'нужны непересекающиеся P и Q_correct')
    if len(hall_matrix) < n_cloud:
        raise ValueError(
            f'пул hallucinated ({len(hall_matrix)}) меньше n_cloud={n_cloud}')
    correct_rng = np.random.default_rng(seed_p)
    idx_p = correct_rng.choice(len(correct_matrix), size=n_cloud, replace=False)
    remaining_correct = np.setdiff1d(
        np.arange(len(correct_matrix), dtype=np.int64), idx_p, assume_unique=False)
    idx_q = correct_rng.choice(remaining_correct, size=n_cloud, replace=False)
    idx_h = np.random.default_rng(seed_h).choice(
        len(hall_matrix), size=n_cloud, replace=False)
    permutation = np.random.default_rng(seed_mix).permutation(n_cloud)
    p_cloud = correct_matrix[idx_p]
    q_correct_cloud = correct_matrix[idx_q]
    h_cloud = hall_matrix[idx_h]
    q_cloud = make_alpha_replacement(q_correct_cloud, h_cloud, alpha, permutation)
    n_hall = int(round(alpha * n_cloud))
    replaced = np.zeros(n_cloud, dtype=bool)
    replaced[permutation[:n_hall]] = True
    p_items = [('correct', int(correct_index[position])) for position in idx_p]
    q_items = []
    for position in range(n_cloud):
        if replaced[position]:
            q_items.append(('hallucinated', int(hall_index[idx_h[position]])))
        else:
            q_items.append(('correct', int(correct_index[idx_q[position]])))
    return {
        'P': p_cloud,
        'Q': q_cloud,
        'P_items': p_items,
        'Q_items': q_items,
        'n_hall': n_hall,
    }


def packet_confidence_features(item_scores, packet):
    """Агрегаты скоринга уверенности по составу пакета (группа llm_confidence).

    ``item_scores`` — DataFrame, выровненный с полным кэшем: ``.iloc[i]``
    — строка пункта с глобальным item-индексом i; столбцы
    ``nll``/``ppl``/``entropy``/``answer_length`` (пункты вне скоренного
    сплита — NaN). По Q-составу пакета (все ``Q_items``): среднее и
    смещённое СКО (ddof=0) NLL, средние энтропии и длины; по P-составу
    (``P_items``): средние NLL, энтропии и длины — baseline тех же
    correct-пунктов. Возвращает dict с ключами ``q_nll_mean, q_nll_std,
    q_entropy_mean, q_len_mean, p_nll_mean, p_entropy_mean, p_len_mean``.
    """
    q_positions = [index for _, index in packet['Q_items']]
    p_positions = [index for _, index in packet['P_items']]
    q_rows = item_scores.iloc[q_positions]
    p_rows = item_scores.iloc[p_positions]
    return {
        'q_nll_mean': float(q_rows['nll'].mean()),
        'q_nll_std': float(q_rows['nll'].std(ddof=0)),
        'q_entropy_mean': float(q_rows['entropy'].mean()),
        'q_len_mean': float(q_rows['answer_length'].mean()),
        'p_nll_mean': float(p_rows['nll'].mean()),
        'p_entropy_mean': float(p_rows['entropy'].mean()),
        'p_len_mean': float(p_rows['answer_length'].mean()),
    }


def evaluate_detection(packet_table, feature_groups=None):
    """Задача A: обнаружение alpha > 0 по группам признаков (мастер-план §L9).

    ``packet_table`` — DataFrame со столбцами ``alpha``, ``fold`` и
    признаками групп; ``feature_groups=None`` означает ``FEATURE_GROUPS``.
    Для каждой группы строки с NaN в её признаках отбрасываются, далее
    pipeline StandardScaler + LogisticRegression(max_iter=1000, C=1.0);
    разбиение — GroupKFold(N_SPLITS) по столбцу ``fold`` (пакеты с общими
    prompt_id не попадают одновременно в train и test); out-of-fold
    вероятности дают AUROC (roc_auc_score), AUPRC (average_precision_score)
    и balanced accuracy (порог 0.5, balanced_accuracy_score) при
    ``y = 1[alpha > 0]``. Возвращает ``{группа: {auroc, auprc,
    balanced_accuracy}}``; при наличии группы ``'combined'`` дополнительно
    ``'per_alpha_accuracy'`` — dict alpha -> доля y_pred=1 в out-of-fold
    предсказаниях по каждому alpha > 0 (для alpha>0 это доля верных
    предсказаний) — и ``'alpha0_false_positive_rate'`` — та же доля при
    alpha=0 (ложные срабатывания на same-law пакетах; NaN при отсутствии
    пакетов alpha=0).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        roc_auc_score,
    )
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    groups = FEATURE_GROUPS if feature_groups is None else feature_groups
    y_all = (packet_table['alpha'] > 0).astype(int)
    results = {}
    for group_name, features in groups.items():
        valid = packet_table[features].notna().all(axis=1).to_numpy()
        x_group = packet_table.loc[valid, features]
        y_group = y_all[valid]
        folds_group = packet_table.loc[valid, 'fold']
        alpha_group = packet_table.loc[valid, 'alpha']
        proba = np.full(len(x_group), np.nan)
        splitter = GroupKFold(n_splits=N_SPLITS)
        for train_pos, test_pos in splitter.split(x_group, y_group, groups=folds_group):
            model = make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
            model.fit(x_group.iloc[train_pos], y_group.iloc[train_pos])
            proba[test_pos] = model.predict_proba(x_group.iloc[test_pos])[:, 1]
        results[group_name] = {
            'auroc': float(roc_auc_score(y_group, proba)),
            'auprc': float(average_precision_score(y_group, proba)),
            'balanced_accuracy': float(
                balanced_accuracy_score(y_group, (proba >= 0.5).astype(int))),
        }
        if group_name == 'combined':
            predictions = (proba >= 0.5).astype(int)
            alpha_values = alpha_group.to_numpy()
            results['per_alpha_accuracy'] = {
                float(alpha): float(np.mean(predictions[alpha_values == alpha]))
                for alpha in sorted(set(alpha_values.tolist())) if alpha > 0
            }
            zero_mask = alpha_values == 0.0
            results['alpha0_false_positive_rate'] = (
                float(np.mean(predictions[zero_mask])) if zero_mask.any()
                else float('nan'))
    return results


def evaluate_regression(packet_table, feature_groups=None):
    """Задача B: ridge-оценка доли alpha по группам признаков (мастер-план §L9).

    ``packet_table`` — как в ``evaluate_detection``; target = alpha;
    pipeline StandardScaler + Ridge(alpha=1.0); разбиение —
    GroupKFold(N_SPLITS) по столбцу ``fold``; на out-of-fold предсказаниях:
    MAE (mean_absolute_error), R2 (r2_score), Spearman
    (scipy.stats.spearmanr, ``.statistic``). Для группы ``'combined'``
    дополнительно ``'per_alpha_mae'`` — dict alpha -> mean(|pred - alpha|)
    по каждому alpha таблицы. Предсказания не возвращаются: таблица
    предсказаний для калибровки собирается ран-скриптом отдельно (тем же
    CV-разбиением). Возвращает ``{группа: {mae, r2, spearman}}``.
    """
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    groups = FEATURE_GROUPS if feature_groups is None else feature_groups
    target_all = packet_table['alpha'].to_numpy(dtype=float)
    results = {}
    for group_name, features in groups.items():
        valid = packet_table[features].notna().all(axis=1).to_numpy()
        x_group = packet_table.loc[valid, features]
        target = target_all[valid]
        folds_group = packet_table.loc[valid, 'fold']
        prediction = np.full(len(x_group), np.nan)
        splitter = GroupKFold(n_splits=N_SPLITS)
        for train_pos, test_pos in splitter.split(x_group, target, groups=folds_group):
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(x_group.iloc[train_pos], target[train_pos])
            prediction[test_pos] = model.predict(x_group.iloc[test_pos])
        entry = {
            'mae': float(mean_absolute_error(target, prediction)),
            'r2': float(r2_score(target, prediction)),
            'spearman': float(spearmanr(target, prediction).statistic),
        }
        if group_name == 'combined':
            entry['per_alpha_mae'] = {
                float(alpha): float(np.mean(np.abs(prediction - target)[target == alpha]))
                for alpha in sorted(set(target.tolist()))
            }
        results[group_name] = entry
    return results
