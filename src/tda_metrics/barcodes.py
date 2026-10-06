"""Извлечение и отрисовка баркодов: H0/H1-анатомия одного облака и пары.

Три источника баркодов: single_linkage_bars — H0 одного облака merge-дистанциями
single-linkage (scipy-механика mtd0 из metrics); persistence_diagrams — полные
диаграммы одного облака через ripser++; cross_bars — кросс-баркод пары строго
через MTopDiv (mtd.calc_cross_barcodes, как TopologyMetrics.mtd_homology) —
собственных пере-реализаций кросс-матрицы нет. Тяжёлые зависимости (ripser++,
MTopDiv, matplotlib) импортируются лениво внутри функций; сводки и пороги
работают на чистом numpy/scipy.
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

__all__ = [
    'bar_counts',
    'cross_bars',
    'longest_bar_length',
    'persistence_diagrams',
    'plot_barcode',
    'single_linkage_bars',
    'summarize_bars',
    'top_bars',
]

_LEVEL_COLORS = {0: 'tab:blue', 1: 'tab:green'}


def single_linkage_bars(X):
    """H0-баркод одного облака: массив (n, 2) баров (birth, death).

    Смерти — высоты слияния single-linkage по конденсированной матрице
    попарных расстояний (механика mtd0 из metrics: scipy linkage,
    method='single') в порядке linkage: n-1 конечных баров (0.0, merge) плюс
    последний существенный (0.0, inf). Детерминирован; пустое облако ->
    (0, 2), одна точка -> единственный бар (0.0, inf).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f'ожидается 2D-массив (n, d), получена {X.ndim}D')
    n = len(X)
    if n == 0:
        return np.zeros((0, 2))
    bars = np.zeros((n, 2), dtype=np.float64)
    if n > 1:
        bars[:n - 1, 1] = linkage(pdist(X), method='single')[:, 2]
    bars[n - 1, 1] = np.inf
    return bars


def persistence_diagrams(X, maxdim=1):
    """Диаграммы персистентности одного облака через ripser++.

    Python-биндинги ripser++ (README: «The ripserplusplus Python API»):
    ``import ripserplusplus; ripserplusplus.run(arguments, X)`` — первый
    аргумент строка опций CLI (``--format point-cloud --dim k``), второй —
    2D numpy-массив точек; результат — словарь numpy-массивов пар
    (birth, death), индексированный размерностью; массивы могут быть
    structured (поля 'birth'/'death', float32) или плоскими. Совместимость
    с ripser-стилем (результат с ключом 'dgms') и атрибутом .dgms
    сохранена. В отличие от ``single_linkage_bars``, H0-диаграмма ripser++
    может не содержать существенный класс (конечные бары слияний без
    inf-бара). Возвращает список массивов по уровням 0..maxdim; пустой
    уровень — массив (0, 2); нераспознанный формат — RuntimeError с типом.
    """
    import ripserplusplus

    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    maxdim = int(maxdim)
    if maxdim < 0:
        raise ValueError(f'maxdim должен быть >= 0: {maxdim!r}')
    result = ripserplusplus.run(f'--format point-cloud --dim {maxdim}', X)

    def _as_diagram(value):
        if value is None or len(value) == 0:
            return np.zeros((0, 2))
        arr = np.asarray(value)
        if arr.dtype.names is not None:
            fields = [name for name in ('birth', 'death') if name in arr.dtype.names]
            if len(fields) != 2:
                fields = list(arr.dtype.names[:2])
            arr = np.stack([arr[name] for name in fields], axis=1)
        return np.asarray(arr, dtype=np.float64).reshape(-1, 2)

    if isinstance(result, dict) and 'dgms' in result:
        dgms = result['dgms']
        return [_as_diagram(dgms[level]) if level < len(dgms) else np.zeros((0, 2))
                for level in range(maxdim + 1)]
    if isinstance(result, dict):
        return [_as_diagram(result.get(level, result.get(str(level))))
                for level in range(maxdim + 1)]
    dgms = getattr(result, 'dgms', None)
    if dgms is not None:
        return [_as_diagram(dgms[level]) if level < len(dgms) else np.zeros((0, 2))
                for level in range(maxdim + 1)]
    raise RuntimeError(
        f'ripser++ вернул нераспознанный результат (тип {type(result).__name__})')


def cross_bars(P, Q):
    """Кросс-баркод пары (P, Q) через MTopDiv: список массивов по уровням.

    Строго mtd.calc_cross_barcodes с полными батчами (как
    TopologyMetrics.mtd_homology): сумма длин уровня 1 равна mtd(P, Q)
    при полных облаках. Возвращает список np.ndarray (m, 2) [birth, death]
    по уровням 0..dim.
    """
    import mtd

    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if len(P) == 0 or len(Q) == 0:
        raise ValueError('кросс-баркод определён для непустых облаков')
    barcodes = mtd.calc_cross_barcodes(
        P, Q,
        batch_size1=len(P), batch_size2=len(Q),
        pdist_device='cpu', dim=1, is_plot=False,
    )
    return [np.asarray(level, dtype=np.float64).reshape(-1, 2)
            for level in barcodes]


def longest_bar_length(dgm):
    """Самый длинный конечный бар диаграммы (death - birth); inf-смерти игнорируются; пусто -> 0.0."""
    dgm = np.asarray(dgm, dtype=np.float64).reshape(-1, 2)
    if len(dgm) == 0:
        return 0.0
    finite = dgm[np.isfinite(dgm[:, 1])]
    if len(finite) == 0:
        return 0.0
    return float((finite[:, 1] - finite[:, 0]).max())


def bar_counts(dgm, threshold):
    """Число конечных баров длиннее threshold; inf-смерти не считаются."""
    dgm = np.asarray(dgm, dtype=np.float64).reshape(-1, 2)
    if len(dgm) == 0:
        return 0
    finite = dgm[np.isfinite(dgm[:, 1])]
    if len(finite) == 0:
        return 0
    return int(np.sum(finite[:, 1] - finite[:, 0] > float(threshold)))


def summarize_bars(bars):
    """Сводка одного уровня баркода: n_bars, total_length, longest, mean_length, max_death."""
    bars = np.asarray(bars, dtype=np.float64).reshape(-1, 2)
    if len(bars) == 0:
        return {'n_bars': 0, 'total_length': 0.0, 'longest': 0.0,
                'mean_length': 0.0, 'max_death': 0.0}
    lengths = bars[:, 1] - bars[:, 0]
    return {
        'n_bars': int(len(bars)),
        'total_length': float(lengths.sum()),
        'longest': float(lengths.max()),
        'mean_length': float(lengths.mean()),
        'max_death': float(bars[:, 1].max()),
    }


def top_bars(bars, k=5):
    """k самых длинных баров по убыванию длины; (min(k, n), 2)."""
    bars = np.asarray(bars, dtype=np.float64).reshape(-1, 2)
    if len(bars) == 0 or int(k) <= 0:
        return np.zeros((0, 2))
    lengths = bars[:, 1] - bars[:, 0]
    order = np.argsort(lengths)[::-1][:int(k)]
    return bars[order]


def plot_barcode(ax, bars, level, max_bars=60, label=None):
    """Отрисовать баркод на готовом ax горизонтальными отрезками; возвращает ax.

    inf-смерти клипаются на максимальную конечную смерть (1.0 при её
    отсутствии), затем бары сортируются по длине убыванием и первые
    max_bars рисуются отрезками y = 0..k с x = (birth, death); цвет —
    конвенция MTopDiv: уровень 0 'tab:blue', уровень 1 'tab:green'.
    label вешается на первый отрезок (одна запись в легенде); matplotlib
    в модуле не импортируется — ax передаётся готовым.
    """
    bars = np.asarray(bars, dtype=np.float64).reshape(-1, 2)
    if int(max_bars) <= 0 or len(bars) == 0:
        return ax
    clipped = bars.copy()
    finite_deaths = clipped[np.isfinite(clipped[:, 1]), 1]
    limit = float(finite_deaths.max()) if len(finite_deaths) else 1.0
    clipped[~np.isfinite(clipped[:, 1]), 1] = limit
    order = np.argsort(clipped[:, 1] - clipped[:, 0])[::-1][:int(max_bars)]
    color = _LEVEL_COLORS.get(int(level), 'tab:gray')
    for y, j in enumerate(order):
        ax.plot([clipped[j, 0], clipped[j, 1]], [y, y], color=color, lw=1.2,
                label=label if (y == 0 and label is not None) else None)
    ax.set_ylim(-1, len(order))
    return ax
