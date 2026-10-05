"""Статистический протокол (мастер-план §5): контрасты, bootstrap, permutation, BH.

Чистые функции над готовыми массивами и таблицами экспериментов — сами метрики
не считаются, heavy-стек не нужен. Контраст с same-law контролем и
z-стандартизация — §5.1; bootstrap целыми группами с сохранением пар — §5.2 и
appendix §29.3 (для RTD общий idx применяется к обеим сторонам); парная
перестановка меток внутри prompt — §5.3; Benjamini–Hochberg для заранее
определённых семейств — §5.4; первый обнаруженный шаг траектории — S4.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    'benjamini_hochberg',
    'grouped_bootstrap_indices',
    'paired_permutation_labels',
    'contrast_to_control',
    'first_detected_step',
    'metric_contrast_table',
]


def benjamini_hochberg(pvalues, alpha=0.05):
    """Поправка Benjamini–Hochberg (step-up) для семейства p-value (§5.4).

    Принимает одномерный массив p-value из [0, 1] (скаляр допускается) и
    уровень alpha из (0, 1). Возвращает пару (reject_mask, adjusted_pvalues):
    adjusted p-value — p_(i) * m / i по возрастанию p с «подъёмом» минимумом
    справа (монотонность по рангу) и обрезкой единицей; H0 отклоняется там,
    где adjusted <= alpha. Проверочный пример с ручной арифметикой:
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205] ->
    adjusted = [0.008, 0.032, 0.0672, 0.0672, 0.0672, 0.08, 0.0846, 0.205],
    при alpha=0.05 отклоняются первые две гипотезы.
    """
    p = np.atleast_1d(np.asarray(pvalues, dtype=float))
    if not 0.0 < alpha < 1.0:
        raise ValueError(f'alpha должен быть в (0, 1): {alpha!r}')
    if p.size and (np.any(~np.isfinite(p)) or np.any(p < 0.0) or np.any(p > 1.0)):
        raise ValueError('p-value должны быть конечными и лежать в [0, 1]')
    m = p.size
    adjusted = np.empty(m, dtype=float)
    if m == 0:
        return np.zeros(0, dtype=bool), adjusted
    order = np.argsort(p, kind='stable')
    scaled = p[order] * m / np.arange(1, m + 1, dtype=float)
    adjusted_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted <= alpha, adjusted


def grouped_bootstrap_indices(group_ids, n_resamples, seed):
    """Индексы группового bootstrap: группы ресэмплируются целиком (§5.2, §29.3).

    В каждом resample рисуется столько же групп, сколько уникальных во входе,
    но с возвращением: группа входит целиком (возможно несколько раз — тогда
    её индексы повторены в idx) либо не входит. Парность не разрушается:
    общий idx применяется к обеим сторонам пары, для RTD —
    P_boot = P[idx], Q_boot = Q[idx]. Суммарная длина resample варьирует,
    если группы разного размера. Возвращает список из n_resamples
    отсортированных массивов индексов строк входных данных; результат
    детерминирован при фиксированном seed.
    """
    ids = np.asarray(group_ids)
    if ids.ndim != 1:
        raise ValueError(f'group_ids должны быть одномерными: shape={ids.shape}')
    if ids.size == 0:
        raise ValueError('group_ids пусты: нет групп для bootstrap')
    n_resamples = int(n_resamples)
    if n_resamples < 0:
        raise ValueError(f'n_resamples должен быть целым >= 0: {n_resamples!r}')
    unique, inverse = np.unique(ids, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(unique.size)]
    rng = np.random.default_rng(seed)
    indices = []
    for _ in range(n_resamples):
        drawn = rng.integers(0, unique.size, size=unique.size)
        indices.append(np.sort(np.concatenate([members[g] for g in drawn])))
    return indices


def paired_permutation_labels(n_pairs, n_permutations, seed):
    """Матрица перестановок меток correct/hallucinated внутри пар (§5.3).

    Для каждой пары в каждой перестановке — независимый честный бросок:
    +1 значит «метки в паре как в данных», -1 — «обмен местами внутри пары».
    Межпарных обменов нет по построению: состав пар не участвует, меняется
    только знак каждой пары независимо. Возвращает целочисленную матрицу
    формы (n_permutations, n_pairs) из +-1; детерминирована при фиксированном
    seed.
    """
    n_pairs = int(n_pairs)
    n_permutations = int(n_permutations)
    if n_pairs < 0:
        raise ValueError(f'n_pairs должен быть целым >= 0: {n_pairs!r}')
    if n_permutations < 0:
        raise ValueError(f'n_permutations должен быть целым >= 0: {n_permutations!r}')
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(n_permutations, n_pairs)) * 2 - 1


def contrast_to_control(values, control_values, kind='distance', eps=1e-12):
    """Контраст метрики с same-law контролем и z-оценка (§5.1).

    mu0/sigma0 — среднее и выборочное std (ddof=1) контрольных same-law
    значений. Для distance-метрик delta = m - mu0; для similarity-метрик
    (kind='similarity') знак развёрнут: delta = mu0 - m — положительный
    контраст всегда «хуже контроля». z = delta / (sigma0 + eps); eps защищает
    от деления на ноль при вырожденном контроле (sigma0 = 0).

    values — скаляр или одномерный массив наблюдений (NaN в наблюдениях
    проходит на выход как NaN); control_values — не менее двух конечных
    значений (меньше двух или non-finite в контроле — ValueError).
    Возвращает (delta, z): пару float при скалярном входе, пару массивов
    при массиве.
    """
    if kind not in ('distance', 'similarity'):
        raise ValueError(f"kind должен быть 'distance' или 'similarity': {kind!r}")
    control = np.asarray(control_values, dtype=float)
    if control.ndim != 1 or control.size < 2:
        raise ValueError('control_values: одномерные same-law значения, минимум два')
    if np.any(~np.isfinite(control)):
        raise ValueError('control_values содержат NaN/Inf: same-law контроль испорчен')
    arr = np.asarray(values, dtype=float)
    scalar = arr.ndim == 0
    observed = np.atleast_1d(arr)
    if observed.ndim != 1:
        raise ValueError(f'values: скаляр или одномерный массив, shape={observed.shape}')
    mu0 = float(np.mean(control))
    sigma0 = float(np.std(control, ddof=1))
    delta = observed - mu0
    if kind == 'similarity':
        delta = -delta
    z = delta / (sigma0 + eps)
    if scalar:
        return float(delta[0]), float(z[0])
    return delta, z


def first_detected_step(steps, deltas, ci_low, ci_high, alpha_grid=None):
    """Первый шаг траектории со значимым контрастом (S4, часть I).

    t* — минимальный шаг, на котором 95% CI контраста лежит строго выше нуля
    (ci_low > 0: положительный контраст = ухудшение относительно same-law;
    правильное направление — рост контраста). CI, накрывающий ноль, и тем
    более значимый отрицательный CI («неправильное направление»)
    обнаружением не считаются; NaN в CI тоже не обнаруживается.

    Входы — готовые массивы из таблиц эксперимента: steps (n_steps,);
    ci_low/ci_high (n_steps,); deltas (n_steps,) либо (n_repeats, n_steps) —
    во втором случае delta берётся как среднее повторов, а в выдаче есть
    доля повторов с положительным контрастом в точке t* (NaN в повторе
    считается неправильным направлением). alpha_grid — необязательные
    значения параметра деформации для шагов (например, доли dropping из
    S3): при передаче в выдаче есть alpha в точке обнаружения (§12:
    «минимальная обнаружимая alpha»).

    Возвращает dict с ключами step, index, delta, alpha, direction_share;
    значения None, если обнаружения не произошло или величина неприменима
    (1D deltas без повторов, не передан alpha_grid).
    """
    steps_arr = np.asarray(steps)
    if steps_arr.ndim != 1:
        raise ValueError(f'steps должны быть одномерными: shape={steps_arr.shape}')
    n_steps = steps_arr.size
    deltas_arr = np.asarray(deltas, dtype=float)
    if deltas_arr.ndim not in (1, 2):
        raise ValueError(f'deltas: (n_steps,) или (n_repeats, n_steps): shape={deltas_arr.shape}')
    if deltas_arr.ndim == 1 and deltas_arr.shape != (n_steps,):
        raise ValueError(f'deltas должны иметь длину n_steps={n_steps}: {deltas_arr.shape}')
    if deltas_arr.ndim == 2 and deltas_arr.shape[1] != n_steps:
        raise ValueError(f'deltas должны иметь ширину n_steps={n_steps}: {deltas_arr.shape}')
    ci_low_arr = np.asarray(ci_low, dtype=float)
    ci_high_arr = np.asarray(ci_high, dtype=float)
    if ci_low_arr.shape != (n_steps,) or ci_high_arr.shape != (n_steps,):
        raise ValueError(f'ci_low/ci_high должны иметь форму ({n_steps},)')
    alpha_arr = None
    if alpha_grid is not None:
        alpha_arr = np.asarray(alpha_grid)
        if alpha_arr.shape != (n_steps,):
            raise ValueError(f'alpha_grid должен иметь форму ({n_steps},)')
    result = {'step': None, 'index': None, 'delta': None,
               'alpha': None, 'direction_share': None}
    repeats = deltas_arr if deltas_arr.ndim == 2 else None
    mean_deltas = deltas_arr if repeats is None else repeats.mean(axis=0)
    detected = np.flatnonzero(ci_low_arr > 0.0)
    if detected.size == 0:
        return result
    t = int(detected[0])
    result['index'] = t
    result['step'] = steps_arr[t].item()
    result['delta'] = float(mean_deltas[t])
    if alpha_arr is not None:
        result['alpha'] = alpha_arr[t].item()
    if repeats is not None:
        result['direction_share'] = float(np.mean(repeats[:, t] > 0.0))
    return result


def _default_kind(metric):
    if str(metric).startswith(('precision', 'recall')):
        return 'similarity'
    return 'distance'


def metric_contrast_table(rows_frame, control_mask, metric_cols, kind_map=None):
    """Таблица контрастов delta/z всех строк относительно same-law контроля.

    Для каждой метрики из metric_cols mu0/sigma0 оцениваются по control-строкам
    (control_mask — булев массив по позициям строк rows_frame), затем для
    каждой не-control строки вычисляются delta и z (contrast_to_control).
    kind по умолчанию: 'similarity' для precision*/recall* (больше = ближе к
    данным), 'distance' для остальных; kind_map (метрика -> kind) переопределяет
    умолчание, ключи вне metric_cols игнорируются (неверное значение kind —
    ValueError).

    NaN/Inf в контроле метрики исключаются до оценки mu0/sigma0; если
    конечных контрольных значений меньше двух, обе колонки метрики
    заполняются NaN — ошибка одной метрики не уничтожает остальные
    результаты (§9.6). NaN в наблюдениях проходит в delta/z как NaN.
    Возвращает DataFrame с индексом не-control строк и колонками
    '<метрика>_delta' и '<метрика>_z'.
    """
    mask = np.asarray(control_mask).astype(bool)
    if mask.ndim != 1 or mask.size != len(rows_frame):
        raise ValueError(f'control_mask: булев массив длины {len(rows_frame)}')
    if not mask.any():
        raise ValueError('control_mask не выделяет ни одной same-law строки')
    metric_cols = list(metric_cols)
    if not metric_cols:
        raise ValueError('metric_cols пусты: нечего сводить')
    missing = [column for column in metric_cols if column not in rows_frame.columns]
    if missing:
        raise ValueError(f'в rows_frame нет колонок метрик: {missing}')
    kinds = dict(kind_map) if kind_map is not None else {}
    for kind in kinds.values():
        if kind not in ('distance', 'similarity'):
            raise ValueError(f"kind должен быть 'distance' или 'similarity': {kind!r}")
    target = ~mask
    out = pd.DataFrame(index=rows_frame.index[target])
    for metric in metric_cols:
        kind = kinds.get(metric)
        if kind is None:
            kind = _default_kind(metric)
        control_values = pd.to_numeric(rows_frame.loc[mask, metric], errors='coerce')
        control_values = control_values.to_numpy(dtype=float)
        control_values = control_values[np.isfinite(control_values)]
        observed = pd.to_numeric(rows_frame.loc[target, metric], errors='coerce')
        observed = observed.to_numpy(dtype=float)
        if control_values.size < 2:
            out[f'{metric}_delta'] = np.full(observed.size, np.nan)
            out[f'{metric}_z'] = np.full(observed.size, np.nan)
            continue
        delta, z = contrast_to_control(observed, control_values, kind=kind)
        out[f'{metric}_delta'] = delta
        out[f'{metric}_z'] = z
    return out
