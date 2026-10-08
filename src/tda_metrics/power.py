"""Эмпирический Monte Carlo анализ мощности метрик (мастер-план §5.6).

Вопрос: при каком n метрика обнаруживает заданный эффект с мощностью >= 0.8
при данном alpha. Дизайн project-consistent: порог обнаружения — не
аналитический критерий, а эмпирический квантиль собственного same-law
контроля конкретного прогона (контраст с same-law контролем — тот же
принцип, что и в statistics.contrast_to_control, но здесь порог прикладной —
доля эффект-наблюдений за порогом, а не z-оценка одного наблюдения).

Модуль не считает метрики сам: effect_sampler/control_sampler и metrics_fn —
внешние зависимости (инъекция), как make_runner в synthetic_grid.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    'power_curve',
    'empirical_power',
    'aggregate_power',
    'select_min_n',
    'default_alternative_map',
]

_ALTERNATIVES = ('greater', 'less', 'twosided')

_POWER_CURVE_COLUMNS = (
    'alpha', 'power', 'threshold', 'n_control', 'n_effect',
    'mean_control', 'sd_control', 'mean_effect', 'z_mean_effect',
)

_VALUES_FRAME_COLUMNS = ('n', 'kind', 'repeat', 'metric', 'value')


def _validate_alternative(alternative):
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative должен быть одним из {_ALTERNATIVES}: {alternative!r}")


def _validate_alpha_grid(alpha_grid):
    alphas = list(alpha_grid)
    if not alphas:
        raise ValueError('alpha_grid пуст: нечего оценивать')
    for alpha in alphas:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f'alpha должен быть в (0, 1): {alpha!r}')
    return alphas


def default_alternative_map(metrics):
    """Карта metric -> alternative по конвенции проекта (§5.6, риск R-PWR-1).

    Similarity-метрики (имена с префиксом precision*/recall* — тот же
    предикат, что statistics._default_kind) детектируются уходом ВНИЗ от
    same-law распределения: их «ухудшение» — уменьшение значения, поэтому
    им соответствует alternative='less' (порог — нижний квантиль контроля,
    детект — строго ниже порога). Остальные метрики — distance-подобные,
    им соответствует alternative='greater'. Возвращает dict по iterable
    имён метрик; используется в aggregate_power(kind_map=...).
    """
    return {
        metric: ('less' if str(metric).startswith(('precision', 'recall')) else 'greater')
        for metric in metrics
    }


def power_curve(effect_values, control_values, alpha_grid=(0.05, 0.10, 0.20), alternative='greater'):
    """Эмпирическая мощность метрики при фиксированной паре выборок (§5.6).

    Порог для alternative='greater' — эмпирический квантиль контроля уровня
    1 - alpha (np.quantile(control_values, 1 - alpha, method='higher');
    'higher' гарантирует, что порог — реально наблюдавшееся в контроле
    значение, без оптимистичной интерполяции между соседними точками).
    power(alpha) = доля effect_values строго выше порога.

    Для alternative='less' — зеркальная механика для similarity-метрик,
    у которых «ухудшение» — уменьшение значения: порог — нижний квантиль
    np.quantile(control_values, alpha, method='lower') ('lower' — реально
    наблюдавшееся значение, консервативно: фактический уровень не выше
    номинала), power(alpha) = доля effect_values строго ниже порога.
    Вырождение симметрично 'greater': при alpha < 1/len(control) порог —
    минимум контроля, фактический уровень 1/(n_control+1).

    Для alternative='twosided' строится пара квантилей
    [alpha/2, 1 - alpha/2] (оба — method='higher'); power(alpha) = доля
    effect_values вне этого интервала (строго ниже нижнего или строго выше
    верхнего порога). В этом режиме колонка threshold хранит кортеж
    (lower, upper) вместо скаляра — единственное отступление от плоской
    схемы (single нету в постановке) ради сохранения имени колонки
    неизменным между режимами. Оговорка сериализации: кортеж не переживает
    to_parquet/to_csv round-trip (pyarrow не сериализует кортежи); для
    персистенции twosided-сводки нужны две отдельные колонки
    (threshold_lower/threshold_upper) — до того twosided использовать
    только в памяти.

    control_values — одномерный массив same-law наблюдений, не менее двух
    конечных значений (как в contrast_to_control; меньше двух или NaN/Inf —
    ValueError). effect_values — скаляр или одномерный массив наблюдений
    эффекта; NaN допускается и в power не засчитывается (сравнение с
    порогом даёт False), Inf допускается и засчитывается детектом (Inf
    строго выше любого конечного порога; для twosided -Inf — строго ниже
    нижнего; на практике Inf даёт, например, Fréchet на вырожденной
    ковариации).

    mean_control/sd_control — среднее и выборочное std (ddof=1) контроля.
    z_mean_effect — среднее по effect_values поэлементных
    z = (v - mean_control) / sd_control; при sd_control == 0 z = 0 для
    v == mean_control и z = +inf (без знака — отступление от буквальной
    двусторонней интуиции контраста, принятое ради того, чтобы среднее по
    смеси знаков не вырождалось в NaN из-за +inf + -inf) для v != mean_control.

    Вырождение порога при малом контроле: порог равен максимуму контроля
    при alpha < 1/(len(control_values) - 1) — ceil-индекс method='higher'
    при таких alpha доезжает до последнего порядка статистик; при
    n_control=10 условие alpha < 1/9 ≈ 0.111 покрывает оба ярлыка 0.05 и
    0.10: их пороги идентичны, строки power совпадают. Фактический уровень
    теста «наблюдение строго выше максимума контроля» для независимого
    same-law наблюдения равен 1/(n_control+1) (при n_control=10 это
    1/11 ≈ 0.0909): ярлык alpha=0.05 в вырожденной точке либерален
    (фактический уровень выше номинала), а не консервативен. Для честной
    различимости строк alpha-сетки рекомендуется n_control >= 20 (при 20:
    порог alpha=0.05 — максимум, фактический уровень 1/21 ≈ 0.048;
    alpha=0.10 — второе сверху значение, 2/21 ≈ 0.095; alpha=0.20 —
    четвёртое сверху, 4/21 ≈ 0.190).

    Возвращает DataFrame с одной строкой на alpha из alpha_grid и колонками
    (alpha, power, threshold, n_control, n_effect, mean_control, sd_control,
    mean_effect, z_mean_effect); alpha-независимые колонки повторяются на
    каждой строке ради плоской схемы.
    """
    _validate_alternative(alternative)
    alphas = _validate_alpha_grid(alpha_grid)

    control = np.asarray(control_values, dtype=float)
    if control.ndim != 1 or control.size < 2:
        raise ValueError('control_values: одномерные same-law значения, минимум два')
    if np.any(~np.isfinite(control)):
        raise ValueError('control_values содержат NaN/Inf: same-law контроль испорчен')

    effect = np.atleast_1d(np.asarray(effect_values, dtype=float))
    if effect.ndim != 1:
        raise ValueError(f'effect_values: скаляр или одномерный массив, shape={effect.shape}')

    n_control = int(control.size)
    n_effect = int(effect.size)
    mean_control = float(np.mean(control))
    sd_control = float(np.std(control, ddof=1))
    mean_effect = float(np.nanmean(effect)) if n_effect else float('nan')

    diff = effect - mean_control
    if sd_control == 0.0:
        z_values = np.where(diff == 0.0, 0.0, np.inf)
    else:
        z_values = diff / sd_control
    z_mean_effect = float(np.nanmean(z_values)) if n_effect else float('nan')

    rows = []
    for alpha in alphas:
        if alternative == 'greater':
            threshold = float(np.quantile(control, 1.0 - alpha, method='higher'))
            power = float(np.mean(effect > threshold)) if n_effect else float('nan')
        elif alternative == 'less':
            threshold = float(np.quantile(control, alpha, method='lower'))
            power = float(np.mean(effect < threshold)) if n_effect else float('nan')
        else:
            lower = float(np.quantile(control, alpha / 2.0, method='higher'))
            upper = float(np.quantile(control, 1.0 - alpha / 2.0, method='higher'))
            threshold = (lower, upper)
            power = float(np.mean((effect < lower) | (effect > upper))) if n_effect else float('nan')
        rows.append({
            'alpha': float(alpha),
            'power': power,
            'threshold': threshold,
            'n_control': n_control,
            'n_effect': n_effect,
            'mean_control': mean_control,
            'sd_control': sd_control,
            'mean_effect': mean_effect,
            'z_mean_effect': z_mean_effect,
        })
    return pd.DataFrame(rows, columns=list(_POWER_CURVE_COLUMNS))


def empirical_power(
    effect_sampler,
    control_sampler,
    metrics_fn,
    n_grid=(100, 250, 500, 1000),
    n_control=10,
    n_effect=10,
    seed=0,
    alpha_grid=(0.05, 0.10, 0.20),
    alternative='greater',
):
    """Сырые метрики Monte Carlo прогона: control/effect пары на сетке n (§5.6).

    effect_sampler и control_sampler — callable(n, k) -> (P, Q); сами
    отвечают за собственные seed'ы (k — локальный индекс повтора,
    смещённый базой: фактический аргумент, передаваемый в сэмплер —
    seed + i, где i — порядковый номер повтора 0..n_control-1 (или
    0..n_effect-1 для effect); это даёт детерминизм (тот же seed -> те же
    вызовы сэмплеров -> те же наблюдения) и позволяет запускать несколько
    независимых серий без пересечения повторов, меняя только seed.

    На каждый (n, i) из control-цикла — одна control-пара ->
    metrics_fn(P, Q) (словарь метрик, например compute_all); аналогично
    независимый effect-цикл длины n_effect. Число пар в каждом цикле не
    зависит от другого (n_control и n_effect могут различаться).

    alpha_grid/alternative здесь не используются для вычислений (мощность
    агрегируется потом power_curve/aggregate_power над длинной таблицей) —
    принимаются только для ранней валидации несовместимых параметров перед
    потенциально дорогим циклом сэмплирования (fail fast), ради единообразия
    с остальным API модуля.

    Возвращает long-DataFrame с колонками (n, kind, repeat, metric, value);
    kind принимает значения 'control'/'effect', repeat — локальный индекс
    повтора (0-based, без смещения seed).
    """
    _validate_alternative(alternative)
    _validate_alpha_grid(alpha_grid)
    n_control = int(n_control)
    n_effect = int(n_effect)
    if n_control < 1:
        raise ValueError(f'n_control должен быть целым >= 1: {n_control!r}')
    if n_effect < 1:
        raise ValueError(f'n_effect должен быть целым >= 1: {n_effect!r}')

    records = []
    for n in n_grid:
        n = int(n)
        for i in range(n_control):
            p, q = control_sampler(n, seed + i)
            metrics = metrics_fn(p, q)
            for metric, value in metrics.items():
                records.append({
                    'n': n, 'kind': 'control', 'repeat': i,
                    'metric': metric, 'value': float(value),
                })
        for i in range(n_effect):
            p, q = effect_sampler(n, seed + i)
            metrics = metrics_fn(p, q)
            for metric, value in metrics.items():
                records.append({
                    'n': n, 'kind': 'effect', 'repeat': i,
                    'metric': metric, 'value': float(value),
                })
    return pd.DataFrame.from_records(records, columns=list(_VALUES_FRAME_COLUMNS))


def aggregate_power(values_frame, alpha_grid=(0.05, 0.10, 0.20), alternative='greater',
                    kind_map=None):
    """power_curve по каждой (n, metric)-группе длинной таблицы (§5.6).

    values_frame — формат empirical_power (колонки n, kind, repeat, metric,
    value); kind='control' строки группы идут в control_values, kind='effect'
    — в effect_values. Колонка target_met = power >= 0.8 (целевая доля
    мастер-плана «минимальное n с мощностью 0.8»). Отсутствие обязательных
    колонок — ValueError; недостаточно control-наблюдений в группе (< 2) —
    ValueError из power_curve (ошибка не глушится: либо данные корректны,
    либо прогон не достроен).

    kind_map — словарь metric -> alternative, применяемый по группам:
    группы с метрикой из kind_map агрегируются с указанным направлением,
    остальные — с alternative. None соответствует прежнему поведению
    (единый alternative на все метрики). Для direction-aware прогона —
    default_alternative_map(metrics): similarity-метрики (precision*/recall*)
    получают 'less', иначе односторонний 'greater' пропускает их эффект
    только потому, что он направлен вниз (R-PWR-1: дефект v1 power-таблиц).

    Возвращает long-DataFrame: колонки n, metric, затем колонки
    power_curve (alpha, power, threshold, ...), затем target_met; строки
    упорядочены по (n, metric, alpha) — groupby с sort=True.
    """
    missing = set(_VALUES_FRAME_COLUMNS) - set(values_frame.columns)
    if missing:
        raise ValueError(f'values_frame: не хватает колонок {sorted(missing)}')

    frames = []
    for (n, metric), group in values_frame.groupby(['n', 'metric'], sort=True):
        control_values = group.loc[group['kind'] == 'control', 'value'].to_numpy(dtype=float)
        effect_values = group.loc[group['kind'] == 'effect', 'value'].to_numpy(dtype=float)
        group_alternative = alternative if kind_map is None else kind_map.get(metric, alternative)
        curve = power_curve(effect_values, control_values, alpha_grid=alpha_grid,
                            alternative=group_alternative)
        curve.insert(0, 'metric', metric)
        curve.insert(0, 'n', n)
        frames.append(curve)

    if not frames:
        return pd.DataFrame(columns=['n', 'metric'] + list(_POWER_CURVE_COLUMNS) + ['target_met'])

    result = pd.concat(frames, ignore_index=True)
    result['target_met'] = result['power'] >= 0.8
    return result


def select_min_n(power_frame, target_power=0.8, alpha=0.05):
    """Минимальное n с power >= target_power при фиксированном alpha, на метрику (§5.6).

    power_frame — выход aggregate_power (или совместимая таблица с
    колонками n, metric, alpha, power). Для каждой метрики, встречающейся в
    power_frame, возвращается минимальное n среди строк с alpha == alpha
    (сравнение np.isclose — защита от float-шума в alpha_grid) и
    power >= target_power; если таких строк нет (в т.ч. если для метрики
    нет строк с нужным alpha вовсе) — значение None. Метрика без
    обнаруженного n явно присутствует в словаре со значением None — не
    замалчивается отсутствием ключа.

    Возвращает dict metric -> int | None.
    """
    metrics = sorted(pd.unique(power_frame['metric']))
    subset = power_frame[np.isclose(power_frame['alpha'].astype(float), float(alpha))]
    result = {}
    for metric in metrics:
        rows = subset[subset['metric'] == metric]
        meeting = rows[rows['power'] >= target_power]
        result[metric] = int(meeting['n'].min()) if not meeting.empty else None
    return result
