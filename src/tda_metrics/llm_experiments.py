"""Эксперименты над кэшем LLM-эмбеддингов: сплиты, облака, смеси, контроли.

Опирается на кэш, созданный через ``embedding_cache.save_cache``: meta — DataFrame
с одной строкой на пару (пункт, слой) в порядке «пункт-мажор, слой-минор»
(порядок fractions), vectors — массив (n_items, n_layers, hidden). Отсюда
инвариант индексации: строка meta с номером r соответствует пункту r // n_layers
и позиции слоя r % n_layers.

Режимы (appendix §26): A — независимые распределения (MTD/NTD/PR/MMD/Fréchet),
B — парные представления одних объектов (RTD + 1-CKA). Семейства конфигов
разводят режимы явно: l3/l3d/l3e/l4a — режим A, l4b — режим B. Обязательные
контроли D-006: l3d — length-matched (усл. 1/2), l3e — shuffled-label.
"""
import hashlib

import numpy as np
import pandas as pd

from tda_metrics.experiment_schema import ExperimentConfig
from tda_metrics.mixtures import make_alpha_replacement
from tda_metrics.reduction import pca_fit, pca_transform

__all__ = [
    'prompt_id_split',
    'prompt_ids_hash',
    'layer_clouds',
    'pca_reduce',
    'independent_mixture',
    'paired_clouds',
    'length_matched_pair',
    'sample_length_matched_pair',
    'shuffled_label_pair',
    'linear_cka',
    'build_l3_configs',
    'parse_experiment_id',
]

_SPLIT_NAMES = ('dev', 'val', 'test')
_LENGTH_BINS = 10
_FAMILIES = ('l3', 'l3d', 'l3e', 'l4a', 'l4b')


def prompt_id_split(meta, seed=0, fractions=(0.6, 0.2, 0.2)):
    """Групповой сплит по prompt_id (мастер-план §L2, D-06-усл.5).

    Один prompt_id попадает ровно в одну часть; пары correct/hallucinated
    не разрываются (группировка до разреза). Перестановка уникальных
    prompt_id детерминирована seed; итоговые списки отсортированы.
    """
    if len(fractions) != 3 or not all(f > 0 for f in fractions):
        raise ValueError(f'fractions должен быть тремя положительными долями: {fractions}')
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f'fractions должны суммироваться в 1.0: {fractions} (сумма {sum(fractions)})')
    unique = sorted(pd.unique(meta['prompt_id']).tolist())
    if len(unique) < 3:
        raise ValueError(f'нужно минимум 3 уникальных prompt_id, есть {len(unique)}')
    order = np.random.default_rng(seed).permutation(len(unique))
    total = sum(fractions)
    n_dev = int(round(len(unique) * fractions[0] / total))
    n_val = int(round(len(unique) * fractions[1] / total))
    if n_dev < 1 or n_val < 1 or n_dev + n_val >= len(unique):
        raise ValueError(f'доли {fractions} дают пустую или вырожденную часть сплита')
    shuffled = [unique[i] for i in order]
    parts = {
        'dev': sorted(shuffled[:n_dev]),
        'val': sorted(shuffled[n_dev:n_dev + n_val]),
        'test': sorted(shuffled[n_dev + n_val:]),
    }
    return parts


def prompt_ids_hash(ids):
    """sha256 отсортированного списка prompt_id (D-06-усл.5)."""
    canonical = '\n'.join(sorted(ids))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def layer_clouds(meta, vectors, prompt_ids, answer_kind, layer_index):
    """Матрица эмбеддингов (n, hidden) одного слоя и одного answer_kind.

    meta — ПОЛНЫЙ кэш-meta (строка на (пункт, слой), пункт-мажор), vectors —
    (n_items, n_layers, hidden). Выборка — по answer_kind, layer_index и
    принадлежности prompt_id списку; порядок строк — порядок meta.
    """
    vectors = np.asarray(vectors)
    n_layers = vectors.shape[1]
    allowed = set(prompt_ids)
    mask = (
        (meta['answer_kind'] == answer_kind)
        & (meta['layer_index'] == layer_index)
        & meta['prompt_id'].isin(allowed)
    )
    rows = np.flatnonzero(mask.to_numpy())
    if len(rows) == 0:
        raise ValueError(
            f'пустая выборка: answer_kind={answer_kind!r}, layer={layer_index}, '
            f'{len(allowed)} prompt_id — проверь фильтры и сплит'
        )
    items = rows // n_layers
    layer_pos = rows % n_layers
    if len(set(layer_pos.tolist())) != 1:
        raise ValueError('строки meta spanят несколько позиций слоёв — нарушен порядок кэша')
    return vectors[items, layer_pos[0]]


def pca_reduce(reference_matrix, target_matrix, n_components=16):
    """Проекция target той же PCA-моделью, обученной на reference (RISKS №3)."""
    model = pca_fit(np.asarray(reference_matrix), n_components=n_components)
    projection, _ = pca_transform(np.asarray(target_matrix), model=model)
    return projection


def independent_mixture(correct_matrix, hallucinated_matrix, alpha, permutation):
    """Q_alpha независимой смеси (режим A): замены по общей перестановке.

    Вложенность: смеси большего alpha содержат замены меньшего
    (mixtures.make_alpha_replacement). P — correct, Q_alpha — смесь.
    """
    return make_alpha_replacement(
        np.asarray(correct_matrix), np.asarray(hallucinated_matrix),
        alpha, permutation,
    )


def paired_clouds(correct_matrix, hallucinated_matrix, alpha, permutation):
    """Парная смесь (режим B): те же замены, но строки — соответствующие объекты.

    Численно совпадает с independent_mixture (та же функция замены), но
    интерпретация парная: строка i P и строка i Q — один и тот же объект
    (prompt); для RTD/1-CKA соответствие индексов обязательно.
    """
    correct = np.asarray(correct_matrix)
    hallucinated = np.asarray(hallucinated_matrix)
    if correct.shape != hallucinated.shape:
        raise ValueError(
            f'paired-режим требует равных форм: {correct.shape} != {hallucinated.shape}'
        )
    return make_alpha_replacement(correct, hallucinated, alpha, permutation)


def length_matched_pair(meta, vectors, prompt_ids, layer_index,
                        length_col='n_pooled', n_max=None, seed=0):
    """Length-matched пара correct/hallucinated (D-06-усл.1/2, контроль L3d).

    Длины (length_col; n_pooled включает константный хвост шаблона — для
    сопоставления по рангам это смещение не важно) обеих групп режутся на
    _LENGTH_BINS бинов по эмпирическим квантилям объединённой длины; в
    каждом бине берётся min(count_correct, count_hall) элементов с каждой
    стороны (детерминированно: сортировка по (длина, prompt_id)). Пары
    соответствия НЕТ — режим A; выравниваются распределения длин.
    Возвращает (correct_matrix, hallucinated_matrix, meta_correct, meta_hall).
    """
    correct = layer_clouds(meta, vectors, prompt_ids, 'correct', layer_index)
    hallucinated = layer_clouds(meta, vectors, prompt_ids, 'hallucinated', layer_index)
    layer_mask = meta['layer_index'] == layer_index
    allowed = set(prompt_ids)
    meta_c = meta[layer_mask & (meta['answer_kind'] == 'correct')
                  & meta['prompt_id'].isin(allowed)].reset_index(drop=True)
    meta_h = meta[layer_mask & (meta['answer_kind'] == 'hallucinated')
                  & meta['prompt_id'].isin(allowed)].reset_index(drop=True)
    if len(meta_c) != len(correct) or len(meta_h) != len(hallucinated):
        raise ValueError('meta и облака рассинхронизированы — проверь порядок кэша')
    if len(meta_c) == 0 or len(meta_h) == 0:
        raise ValueError('одна из сторон length-matched пары пуста')

    pooled = np.concatenate([
        np.asarray(meta_c[length_col], dtype=float),
        np.asarray(meta_h[length_col], dtype=float),
    ])
    edges = np.quantile(pooled, np.linspace(0.0, 1.0, _LENGTH_BINS + 1))
    edges[-1] += 1e-9

    def bin_indices(frame):
        return np.clip(np.digitize(np.asarray(frame[length_col], dtype=float), edges) - 1,
                       0, _LENGTH_BINS - 1)

    meta_c = meta_c.copy()
    meta_h = meta_h.copy()
    meta_c['__bin'] = bin_indices(meta_c)
    meta_h['__bin'] = bin_indices(meta_h)
    sizes_c = meta_c.groupby('__bin').size()
    sizes_h = meta_h.groupby('__bin').size()
    quota = {b: int(min(sizes_c.get(b, 0), sizes_h.get(b, 0)))
             for b in range(_LENGTH_BINS)}

    def matched(frame, matrix):
        frame_sorted = frame.sort_values([length_col, 'prompt_id'])
        matrix_sorted = matrix[frame_sorted.index.to_numpy()]
        ranks = frame_sorted.groupby('__bin', sort=False).cumcount()
        limit = frame_sorted['__bin'].map(quota)
        chosen = ranks.to_numpy() < limit.to_numpy()
        return frame_sorted[chosen].reset_index(drop=True), matrix_sorted[chosen]

    meta_c_m, correct_m = matched(meta_c, correct)
    meta_h_m, hallucinated_m = matched(meta_h, hallucinated)
    if n_max is not None:
        n_take = min(n_max, len(meta_c_m), len(meta_h_m))
        if n_take < len(meta_c_m):
            meta_c_m, correct_m = meta_c_m.iloc[:n_take], correct_m[:n_take]
        if n_take < len(meta_h_m):
            meta_h_m, hallucinated_m = meta_h_m.iloc[:n_take], hallucinated_m[:n_take]
    return correct_m, hallucinated_m, meta_c_m.drop(columns='__bin'), meta_h_m.drop(columns='__bin')


def shuffled_label_pair(joint_matrix, labels01, seed=0):
    """Контроль L3e: перемешанные метки на объединённом облаке.

    labels01 (0 = correct, 1 = hallucinated) переставляются детерминированно;
    P/Q — новые «классы» при неизменных распределениях. Возвращает (P, Q).
    """
    matrix = np.asarray(joint_matrix)
    labels = np.asarray(labels01)
    if len(labels) != len(matrix):
        raise ValueError(f'длины не совпадают: {len(labels)} меток, {len(matrix)} строк')
    if not set(np.unique(labels).tolist()) <= {0, 1}:
        raise ValueError('labels01 должен состоять из 0/1')
    permuted = labels[np.random.default_rng(seed).permutation(len(labels))]
    return matrix[permuted == 0], matrix[permuted == 1]


def linear_cka(X, Y):
    """Линейная CKA (Kornblith et al.); 1 - CKA — расстояние представления.

    Центрируются обе матрицы; строки (объекты) — общие, размерности могут
    различаться. Требует равного числа строк.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f'CKA требует равного числа строк: {X.shape[0]} != {Y.shape[0]}')
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    numerator = np.linalg.norm(Yc.T @ Xc, ord='fro') ** 2
    denominator = np.linalg.norm(Xc.T @ Xc, ord='fro') * np.linalg.norm(Yc.T @ Yc, ord='fro')
    if denominator == 0.0:
        raise ValueError('CKA не определена: одна из матриц вырождена')
    return float(numerator / denominator)


def _l3_kwargs(layer, k, n_cloud, alpha=None):
    return dict(
        dataset='halueval_qa',
        model='qwen2.5-3b-instruct',
        representation='hidden',
        layer=layer,
        pooling='mean_answer',
        pca_dim=None,
        n_P=n_cloud,
        n_Q=n_cloud,
        alpha=alpha,
        seed=k,
        split_id='dev',
        prompt_ids_hash=None,
        metric_params=None,
    )


def build_l3_configs(layer_indices, alphas, repeats, n_cloud=500):
    """Конфиги семейств L3/L4: direct, length-matched, shuffled, смеси A/B.

    experiment_id: l3/{layer}/direct/r{k}; l3d/{layer}/r{k}; l3e/{layer}/r{k};
    l4a/{layer}/a{alpha}/r{k}; l4b/{layer}/a{alpha}/r{k}. Семейства l4b —
    парный режим B (RTD/CKA), остальные — режим A.
    """
    configs = []
    for layer in layer_indices:
        for k in range(repeats):
            configs.append(ExperimentConfig(
                experiment_id=f'l3/{layer}/direct/r{k}',
                experiment_family='l3_direct',
                **_l3_kwargs(layer, k, n_cloud),
            ))
            configs.append(ExperimentConfig(
                experiment_id=f'l3d/{layer}/r{k}',
                experiment_family='l3d_length_matched',
                **_l3_kwargs(layer, k, n_cloud),
            ))
            configs.append(ExperimentConfig(
                experiment_id=f'l3e/{layer}/r{k}',
                experiment_family='l3e_shuffled_label',
                **_l3_kwargs(layer, k, n_cloud),
            ))
        for alpha in alphas:
            for k in range(repeats):
                configs.append(ExperimentConfig(
                    experiment_id=f'l4a/{layer}/a{alpha:g}/r{k}',
                    experiment_family='l4a_independent',
                    **_l3_kwargs(layer, k, n_cloud, alpha=alpha),
                ))
                configs.append(ExperimentConfig(
                    experiment_id=f'l4b/{layer}/a{alpha:g}/r{k}',
                    experiment_family='l4b_paired',
                    **_l3_kwargs(layer, k, n_cloud, alpha=alpha),
                ))
    return configs


def parse_experiment_id(eid):
    """Разбор experiment_id семейств L3/L4 в dict; мусор -> ValueError."""
    parts = eid.split('/')
    if not parts or parts[0] not in _FAMILIES:
        raise ValueError(f'неизвестный experiment_id: {eid}')
    family = parts[0]
    try:
        if family in ('l3', 'l3d', 'l3e'):
            if family == 'l3' and len(parts) == 4:
                layer, mode, repeat = int(parts[1]), parts[2], parts[3]
                if mode != 'direct':
                    raise ValueError(f'l3 поддерживает только direct: {eid}')
            elif family != 'l3' and len(parts) == 3:
                layer, repeat = int(parts[1]), parts[2]
                mode = {'l3d': 'length_matched', 'l3e': 'shuffled'}[family]
            else:
                raise ValueError(f'неверная структура: {eid}')
            if not repeat.startswith('r') or not repeat[1:].isdigit():
                raise ValueError(f'неверный repeat: {eid}')
            return {'family': family, 'layer': layer, 'mode': mode,
                    'repeat': int(repeat[1:])}
        if len(parts) == 4:
            layer, alpha_token, repeat = parts[1], parts[2], parts[3]
            if not alpha_token.startswith('a') or not repeat.startswith('r'):
                raise ValueError(f'неверные токены alpha/repeat: {eid}')
            return {'family': family, 'layer': int(layer),
                    'alpha': float(alpha_token[1:]), 'repeat': int(repeat[1:])}
        raise ValueError(f'неверная структура: {eid}')
    except ValueError as exc:
        raise ValueError(f'неизвестный experiment_id: {eid} ({exc})') from None


def sample_length_matched_pair(meta, vectors, prompt_ids, layer_index, seed,
                               n_cloud, bins=10, length_col='n_pooled'):
    """Независимое стратифицированное length-matched подмножество пары correct/hallucinated (L3R).

    Отличие от детерминированного length_matched_pair: внутри каждого бина
    длины элементы выбираются случайно БЕЗ возвращения (rng от seed), а не
    «первыми после сортировки» — повторные вызовы с разными seed дают
    независимые повторения. Отбор двухсторонний: в каждом бине равная квота
    min(count_correct, count_hallucinated); если суммарная квота больше
    n_cloud, квоты сокращаются пропорционально (floor + наибольшие остатки
    по убыванию дробной части, затем по индексу бина — детерминированно).

    Соглашения: meta — DataFrame с колонками layer_index, answer_kind,
    prompt_id и length_col; vectors — массив (len(meta), hidden), строки
    соответствуют meta по порядку. Возвращает (correct_matrix,
    hallucinated_matrix, meta_correct, meta_hallucinated, info): info —
    dict с keys: bin_edges, quotas (список после сокращения),
    n_correct_pool, n_hallucinated_pool, n_selected, length_diagnostics
    (dict: mean_correct, mean_hallucinated, median_correct,
    median_hallucinated, ks_stat, ks_p), selected_correct_prompt_ids,
    selected_hallucinated_prompt_ids.
    """
    if not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError(f'bins должен быть целым >= 1: {bins!r}')
    if not isinstance(n_cloud, (int, np.integer)) or n_cloud < 2:
        raise ValueError(f'n_cloud должен быть целым >= 2: {n_cloud!r}')
    if not isinstance(seed, (int, np.integer)):
        raise ValueError(f'seed должен быть int: {seed!r}')
    allowed_ids = [] if prompt_ids is None else list(prompt_ids)
    if not allowed_ids:
        raise ValueError('prompt_ids пуст — нечего выбирать')

    vectors = np.asarray(vectors)
    if vectors.ndim != 2 or len(vectors) != len(meta):
        raise ValueError(
            f'vectors должен быть массивом (len(meta), hidden) в порядке meta: '
            f'shape {vectors.shape}, строк meta {len(meta)}'
        )
    layer_mask = meta['layer_index'] == layer_index
    allowed = set(allowed_ids)
    mask_c = layer_mask & (meta['answer_kind'] == 'correct') & meta['prompt_id'].isin(allowed)
    mask_h = layer_mask & (meta['answer_kind'] == 'hallucinated') & meta['prompt_id'].isin(allowed)
    idx_c = np.flatnonzero(mask_c.to_numpy())
    idx_h = np.flatnonzero(mask_h.to_numpy())
    meta_c = meta.iloc[idx_c].reset_index(drop=True)
    meta_h = meta.iloc[idx_h].reset_index(drop=True)
    correct = vectors[idx_c]
    hallucinated = vectors[idx_h]
    if len(meta_c) != len(correct) or len(meta_h) != len(hallucinated):
        raise ValueError('meta и облака рассинхронизированы — проверь порядок кэша')
    if len(meta_c) == 0 or len(meta_h) == 0:
        raise ValueError('одна из сторон length-matched пары пуста')

    lengths_c_pool = np.asarray(meta_c[length_col], dtype=float)
    lengths_h_pool = np.asarray(meta_h[length_col], dtype=float)
    pooled = np.concatenate([lengths_c_pool, lengths_h_pool])
    edges = np.quantile(pooled, np.linspace(0.0, 1.0, bins + 1))
    edges[-1] += 1e-9
    bin_c = np.clip(np.digitize(lengths_c_pool, edges) - 1, 0, bins - 1)
    bin_h = np.clip(np.digitize(lengths_h_pool, edges) - 1, 0, bins - 1)
    quota = np.minimum(np.bincount(bin_c, minlength=bins),
                       np.bincount(bin_h, minlength=bins)).astype(np.int64)
    total_quota = int(quota.sum())
    if total_quota == 0:
        raise ValueError('нет ни одного бина длины с обеими сторонами — выбор пуст')
    if total_quota > n_cloud:
        scaled = quota * (n_cloud / total_quota)
        quota = np.floor(scaled).astype(np.int64)
        remainder = int(n_cloud) - int(quota.sum())
        if remainder > 0:
            frac = scaled - np.floor(scaled)
            quota[np.argsort(-frac, kind='stable')[:remainder]] += 1

    rng = np.random.default_rng(seed)

    def draw(bin_ids):
        picked = [
            np.sort(rng.choice(np.flatnonzero(bin_ids == b),
                               size=int(quota[b]), replace=False))
            for b in range(bins) if quota[b] > 0
        ]
        return np.concatenate(picked)

    sel_c = draw(bin_c)
    sel_h = draw(bin_h)
    correct_matrix = correct[sel_c]
    hallucinated_matrix = hallucinated[sel_h]
    meta_correct = meta_c.iloc[sel_c].reset_index(drop=True)
    meta_hallucinated = meta_h.iloc[sel_h].reset_index(drop=True)

    from scipy.stats import ks_2samp
    lengths_c = np.asarray(meta_correct[length_col], dtype=float)
    lengths_h = np.asarray(meta_hallucinated[length_col], dtype=float)
    ks = ks_2samp(lengths_c, lengths_h)
    info = {
        'bin_edges': [float(x) for x in edges],
        'quotas': [int(q) for q in quota],
        'n_correct_pool': int(len(meta_c)),
        'n_hallucinated_pool': int(len(meta_h)),
        'n_selected': int(len(sel_c)),
        'length_diagnostics': {
            'mean_correct': float(np.mean(lengths_c)),
            'mean_hallucinated': float(np.mean(lengths_h)),
            'median_correct': float(np.median(lengths_c)),
            'median_hallucinated': float(np.median(lengths_h)),
            'ks_stat': float(ks.statistic),
            'ks_p': float(ks.pvalue),
        },
        'selected_correct_prompt_ids': np.unique(np.asarray(meta_correct['prompt_id'])),
        'selected_hallucinated_prompt_ids': np.unique(np.asarray(meta_hallucinated['prompt_id'])),
    }
    return correct_matrix, hallucinated_matrix, meta_correct, meta_hallucinated, info
