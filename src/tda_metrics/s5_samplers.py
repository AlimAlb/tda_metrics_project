"""Сэмплеры сценария S5: чистая направленность топологических метрик.

Референс — 8 мод на кольце радиуса 8 (make_mode_reference(n_modes=8,
radius=8.0, spread=1.0)); антипода моды i — мода (i + 4) % 8. S5a удаляет
целиком антисимметричные пары мод: центр активных мод остаётся в нуле,
среднее выборки не сдвигается (mode dropping без дефекта Fréchet). S5b
заменяет точную долю точек внешней симметричной парой мод на оси абсцисс
(mode invention). S5c — моментно-согласованная пара «популяций»: кольцо
из 8 мод против квадрата из 4 мод с совпадающими средним и ковариацией.
Стиль сэмплирования — как в tda_metrics.mixtures.sample_q_dropping_invention:
базовый шум (n, 2) и перестановка присвоения генерируются от одного seed
(метод общих случайных чисел), размеры мод — floor + остаток первым модам.
"""
import numpy as np

from tda_metrics.mixtures import make_mode_reference

__all__ = [
    'sample_s5a',
    'sample_s5b',
    'make_matched_pair_s5c',
    'population_moments',
    'sample_from_population',
    'moment_diagnostics',
]


def _equal_sizes(k, n):
    sizes = np.full(k, n // k, dtype=int)
    sizes[:n - int(sizes.sum())] += 1
    return sizes


def _scatter_modes(noise, assign, centers, spread, sizes):
    points = np.empty_like(noise)
    start = 0
    for center, size in zip(centers, sizes):
        idx = assign[start:start + int(size)]
        points[idx] = center + spread * noise[idx]
        start += int(size)
    return points


def sample_s5a(ref, n, seed, dropped_pairs):
    """Выборка из референса без указанных антисимметричных пар мод.

    Удаляются обе моды каждой пары из dropped_pairs; оставшиеся моды
    сэмплируются с равными весами (n поровну: floor + остаток первым по
    индексу). Антиподность удаляемых пар гарантирует нулевой центр центров
    активных мод, поэтому выборочное среднее остаётся у нуля (свойство из
    MTopDiv-экспериментов). Пара (a, b) антиподна, если
    (b - a) % n_modes == n_modes // 2; моды не должны повторяться между
    парами; должна остаться хотя бы одна мода.

    Возвращает (points, info), где info содержит dropped_pairs, n_modes_kept
    и dropped_mass = 2 * len(dropped_pairs) / ref.n_modes.
    """
    n_modes = ref.n_modes
    half = n_modes // 2
    dropped = []
    for pair in dropped_pairs:
        a, b = pair
        if not (0 <= a < n_modes and 0 <= b < n_modes):
            raise ValueError('индексы мод должны быть в [0, n_modes)')
        if (b - a) % n_modes != half:
            raise ValueError(f'пара {(a, b)} не антиподна')
        if a in dropped or b in dropped:
            raise ValueError('моды не должны повторяться между парами')
        dropped.append(a)
        dropped.append(b)
    kept_idx = [i for i in range(n_modes) if i not in dropped]
    if len(kept_idx) < 1:
        raise ValueError('должна остаться хотя бы одна мода')

    centers = ref.centers[np.asarray(kept_idx, dtype=int)]
    sizes = _equal_sizes(len(kept_idx), n)
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(n, ref.centers.shape[1]))
    assign = rng.permutation(n)
    points = _scatter_modes(noise, assign, centers, ref.spread, sizes)

    info = dict(
        dropped_pairs=[(int(pair[0]), int(pair[1])) for pair in dropped_pairs],
        n_modes_kept=len(kept_idx),
        dropped_mass=2.0 * len(dropped_pairs) / n_modes,
    )
    return points, info


def sample_s5b(ref, n, seed, n_replaced, invent_offset=14.0):
    """Выборка с заменой ровно n_replaced точек внешней симметричной парой мод.

    Замещающие точки берутся из мод с центрами (+invent_offset, 0) и
    (-invent_offset, 0), spread как у референса, поровну (ceil/floor, первая
    внешняя мода получает ceil). Остальные n - n_replaced точек — из всех
    n_modes мод референса поровну (floor + остаток первым модам, базовые
    веса сохранены). Первые n_replaced позиций общей перестановки получают
    внешние точки (make_alpha_replacement-стиль): при изменении n_replaced
    точки пересобираются из того же шума. При n_replaced = 0 выборка
    поточечно совпадает со sample_reference(ref, n, seed).

    Возвращает (points, replaced_mask, info), где replaced_mask — булева
    маска (n,) заменённых точек, info содержит n_replaced,
    alpha_realized = n_replaced / n и invent_offset.
    """
    if not 0 <= n_replaced <= n:
        raise ValueError('n_replaced должен быть в [0, n]')
    if not invent_offset > ref.radius + 3.0 * ref.spread:
        raise ValueError('invent_offset должен быть больше ref.radius + 3 * ref.spread')

    n_ext_plus = int(np.ceil(n_replaced / 2.0))
    n_ext_minus = n_replaced - n_ext_plus
    base_sizes = _equal_sizes(ref.n_modes, n - n_replaced)

    external_centers = np.array([[float(invent_offset), 0.0], [-float(invent_offset), 0.0]])
    centers = np.concatenate([external_centers, ref.centers], axis=0)
    sizes = [n_ext_plus, n_ext_minus] + [int(s) for s in base_sizes]

    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(n, ref.centers.shape[1]))
    assign = rng.permutation(n)
    points = _scatter_modes(noise, assign, centers, ref.spread, sizes)

    replaced_mask = np.zeros(n, dtype=bool)
    replaced_mask[assign[:n_replaced]] = True
    info = dict(
        n_replaced=int(n_replaced),
        alpha_realized=n_replaced / n,
        invent_offset=float(invent_offset),
    )
    return points, replaced_mask, info


def make_matched_pair_s5c():
    """Пара «популяций» с совпадающими средним и ковариацией: кольцо 8 мод vs квадрат 4 мод.

    p: 8 мод на кольце радиуса 8.0, spread=1.0. q: 4 моды в вершинах
    квадрата (±sqrt(32), ±sqrt(32)), spread=1.0. Среднее центров равно нулю
    у обеих; ковариация Σ w c cᵀ + spread² I равна 32 I + I = 33 I у обеих
    (у вершины квадрата (a, a) матрица c cᵀ даёт [[a², a²], [a², a²]], сумма
    четырёх знаковых комбинаций с весом 1/4 равна a² I = 32 I). Возвращает
    (pop_p, pop_q) — словари с ключами mode_centers и spread.
    """
    radius = 8.0
    spread = 1.0
    pop_p = dict(
        mode_centers=make_mode_reference(n_modes=8, radius=radius, spread=spread).centers,
        spread=spread,
    )
    side = np.sqrt(0.5 * radius ** 2)
    signs = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    pop_q = dict(
        mode_centers=side * signs,
        spread=spread,
    )
    mean_p, cov_p = population_moments(pop_p)
    mean_q, cov_q = population_moments(pop_q)
    if not np.allclose(mean_p, mean_q, rtol=1e-9, atol=1e-9):
        raise ValueError('средние популяций p и q должны совпадать (rtol=1e-9)')
    if not np.allclose(cov_p, cov_q, rtol=1e-9):
        raise ValueError('ковариации популяций p и q должны совпадать (rtol=1e-9)')
    return pop_p, pop_q


def population_moments(pop):
    """Среднее и ковариация смеси равновесных мод.

    mean = Σ w_i c_i, cov = Σ w_i c_i c_iᵀ + spread² I, где w_i = 1/k для
    k мод. Возвращает (mean (2,), cov (2, 2)).
    """
    centers = np.asarray(pop['mode_centers'], dtype=float)
    spread = float(pop['spread'])
    weights = np.full(centers.shape[0], 1.0 / centers.shape[0])
    mean = weights @ centers
    cov = np.einsum('i,im,in->mn', weights, centers, centers)
    cov = cov + spread ** 2 * np.eye(centers.shape[1])
    return mean, cov


def sample_from_population(pop, n, seed):
    """Выборка из популяции: равные веса всех мод, стиль sample_s5a.

    Базовый шум (n, 2) и перестановка присвоения от одного seed; размеры
    мод — floor + остаток первым модам.
    """
    centers = np.asarray(pop['mode_centers'], dtype=float)
    spread = float(pop['spread'])
    sizes = _equal_sizes(centers.shape[0], n)
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(n, centers.shape[1]))
    assign = rng.permutation(n)
    return _scatter_modes(noise, assign, centers, spread, sizes)


def moment_diagnostics(P, Q):
    """Выборочные моменты двух облаков и их расхождения.

    mean_p/mean_q — np.mean по точкам, cov_p/cov_q — np.cov(bias=True);
    mean_diff — максимум |mean_p - mean_q| по компонентам, cov_diff —
    максимум |cov_p - cov_q| по элементам.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    mean_p = np.mean(P, axis=0)
    mean_q = np.mean(Q, axis=0)
    cov_p = np.cov(P, rowvar=False, bias=True)
    cov_q = np.cov(Q, rowvar=False, bias=True)
    return dict(
        mean_p=mean_p,
        mean_q=mean_q,
        cov_p=cov_p,
        cov_q=cov_q,
        mean_diff=float(np.max(np.abs(mean_p - mean_q))),
        cov_diff=float(np.max(np.abs(cov_p - cov_q))),
    )
