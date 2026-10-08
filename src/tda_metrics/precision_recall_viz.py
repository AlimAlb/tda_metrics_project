"""Прозрачная 2D-геометрия improved precision/recall@k (Kynkäänniemi et al., 2019).

Числовое ядро — только numpy, без TensorFlow и scipy, поэтому работает
локально и на CPU. knn_radii строит радиусы kNN-шаров так же, как
ManifoldEstimator библиотеки precision_recall: расстояние до k-го
ближайшего «другого» элемента (сама точка стоит на позиции 0
отсортированного ряда своих расстояний). coverage_mask брутфорсом
проверяет попадание query-точек в объединение шаров с закрытой границей.
precision_recall_membership возвращает precision/recall@k, посчитанные из
тех же масок q_covered/p_covered, которые рисует
plot_precision_recall_panel (matplotlib импортируется лениво, внутри
функции). Сэмплеры сценариев mode_drop / mode_invent построены поверх
tda_metrics.mixtures методом общих случайных чисел: при фиксированном
seed шум не меняется, конфигурации различаются только присвоением мод.
"""
import numpy as np

from tda_metrics.mixtures import make_mode_reference, sample_reference

__all__ = [
    'coverage_mask',
    'knn_radii',
    'plot_precision_recall_panel',
    'precision_recall_membership',
    'sample_mode_drop',
    'sample_mode_invent',
]


def knn_radii(points, k):
    """Радиусы kNN-шаров: расстояние от каждой точки до k-го ближайшего ДРУГОГО элемента.

    Сама точка исключается: в полной матрице попарных расстояний её
    собственный ноль стоит на позиции 0, поэтому k-й «другой» — столбец с
    индексом k-1 в матрице без диагонали. Требует points.ndim == 2,
    n >= 2 и 1 <= k < n, иначе ValueError. Для 2D и n <= 1000 полная
    матрица расстояний через broadcasting вычисляется мгновенно.

    points: (n, d) float array; k: int. Возвращает (n,) массив радиусов.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError('points должен быть массивом формы (n, d)')
    n = points.shape[0]
    if n < 2:
        raise ValueError('points должен содержать хотя бы две точки')
    if not 1 <= k < n:
        raise ValueError(f'требуется 1 <= k < n, получено k={k}, n={n}')
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    np.fill_diagonal(dist, np.inf)
    return np.partition(dist, k - 1, axis=1)[:, k - 1]


def coverage_mask(centers, radii, query, atol=1e-12):
    """Маска покрытия query-точек объединением шаров B(centers[i], radii[i]).

    query[j] считается покрытой, если существует i с
    dist(centers[i], query[j]) <= radii[i] + atol; граница шара — покрытая
    точка. Brute-force через broadcasting.

    centers: (nc, d), radii: (nc,), query: (nq, d). Возвращает (nq,) bool.
    """
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    query = np.asarray(query, dtype=float)
    diff = query[:, None, :] - centers[None, :, :]
    dist = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    return np.any(dist <= radii[None, :] + atol, axis=1)


def precision_recall_membership(P, Q, k, atol=1e-12):
    """Принадлежность точек kNN-многообразиям и improved precision/recall@k.

    P (np, d) — референс («данные»), Q (nq, d) — модель. Многообразие
    M_X^(k) — объединение шаров B(x_i, r_i^(k)) с радиусами knn_radii(X, k).
    precision@k = mean_j[q_j ∈ M_P^(k)], recall@k = mean_i[p_i ∈ M_Q^(k)];
    оба значения считаются из тех же масок q_covered / p_covered, которые
    рисует plot_precision_recall_panel. Формулы совпадают с
    knn_precision_recall_features библиотеки precision_recall.

    Возвращает dict с ключами 'k', 'radii_P', 'radii_Q', 'q_covered',
    'p_covered', 'precision', 'recall'.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    radii_P = knn_radii(P, k)
    radii_Q = knn_radii(Q, k)
    q_covered = coverage_mask(P, radii_P, Q, atol=atol)
    p_covered = coverage_mask(Q, radii_Q, P, atol=atol)
    return {
        'k': int(k),
        'radii_P': radii_P,
        'radii_Q': radii_Q,
        'q_covered': q_covered,
        'p_covered': p_covered,
        'precision': float(np.mean(q_covered)),
        'recall': float(np.mean(p_covered)),
    }


def plot_precision_recall_panel(ax, centers, radii, query, covered, k, highlighted_center_index, metric_name, metric_value, axes_limit):
    """Рисует на готовом ax одну панель одного направления improved precision/recall@k.

    Диски B(centers[i], radii[i]) — полупрозрачная заливка (alpha=0.12,
    тонкая граница, один цвет), центры многообразия — мелкие тёмно-серые
    точки, query-точки — зелёные (покрытые) и красные (непокрытые). Ровно
    один выделенный центр (чёрная звезда) соединён отрезком со своим k-м
    ближайшим соседом, отрезок подписан r_i^(k). Заголовок несёт имя
    метрики, k и значение; серая подпись внизу указывает, какое облако
    задаёт радиусы. matplotlib импортируется лениво, внутри функции.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Patch

    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    query = np.asarray(query, dtype=float)
    covered = np.asarray(covered, dtype=bool)

    disk_color = 'tab:blue'
    for center, radius in zip(centers, radii):
        ax.add_patch(Circle((center[0], center[1]), radius, facecolor=disk_color, edgecolor=disk_color, alpha=0.12, linewidth=0.3, zorder=1))
    ax.scatter(centers[:, 0], centers[:, 1], s=8, c='dimgray', zorder=3)
    ax.scatter(query[covered, 0], query[covered, 1], s=14, c='tab:green', zorder=4)
    ax.scatter(query[~covered, 0], query[~covered, 1], s=14, c='tab:red', zorder=4)

    point = centers[highlighted_center_index]
    dist = np.linalg.norm(centers - point, axis=1)
    dist[highlighted_center_index] = np.inf
    neighbor = centers[np.argsort(dist)[k - 1]]
    ax.plot([point[0], neighbor[0]], [point[1], neighbor[1]], color='black', linewidth=1.3, zorder=5)
    ax.scatter([point[0]], [point[1]], marker='*', s=170, c='black', zorder=6)
    segment = neighbor - point
    normal = np.array([-segment[1], segment[0]])
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    label_position = 0.5 * (point + neighbor) + 0.45 * normal
    ax.text(label_position[0], label_position[1], rf'$r_i^{{({k})}}$', fontsize=10, ha='center', va='center', zorder=7)

    direction_label = 'Q in M_P' if metric_name == 'precision' else 'P in M_Q'
    ax.set_title(f'{metric_name}@{k} = {metric_value:.3f} ({direction_label})', fontsize=11)
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)
    ax.set_aspect('equal')

    radii_label = 'radii from P (reference)' if metric_name == 'precision' else 'radii from Q (model)'
    ax.text(0.02, 0.03, radii_label, transform=ax.transAxes, fontsize=7, color='0.4', ha='left', va='bottom')

    handles = [
        Patch(facecolor=disk_color, edgecolor=disk_color, alpha=0.12, linewidth=0.3, label='manifold ball B(x, r)'),
        Line2D([], [], marker='o', linestyle='none', markerfacecolor='tab:green', markeredgecolor='tab:green', markersize=5, label='covered query'),
        Line2D([], [], marker='o', linestyle='none', markerfacecolor='tab:red', markeredgecolor='tab:red', markersize=5, label='uncovered query'),
        Line2D([], [], marker='o', linestyle='none', markerfacecolor='dimgray', markeredgecolor='dimgray', markersize=4, label='manifold center'),
    ]
    ax.legend(handles=handles, fontsize='small', loc='upper right')


def sample_mode_drop(ref, n, seed, dropped_pair=(0, 4)):
    """Q с удалённой симметричной парой мод референса (по умолчанию (0, 4) — противоположные).

    Моды с индексами dropped_pair исключаются, оставшиеся получают равные
    веса: n делится между ними поровну (floor + остаток первым). Общие
    случайные числа: шум (n, d) и перестановка присвоения генерируются от
    одного seed в том же порядке, что в sample_q_dropping_invention, точки
    имеют вид center + spread * normal. При противоположной паре удалённых
    мод центр выборки остаётся около нуля.

    Возвращает (n, d) массив точек.
    """
    dropped = tuple(dropped_pair)
    kept = [i for i in range(ref.n_modes) if i not in dropped]
    centers = ref.centers[kept]
    k = len(kept)
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(n, ref.centers.shape[1]))
    assign = rng.permutation(n)
    sizes = np.full(k, n // k, dtype=int)
    sizes[:n - sizes.sum()] += 1
    q = np.empty_like(noise)
    start = 0
    for j in range(k):
        idx = assign[start:start + sizes[j]]
        q[idx] = centers[j] + ref.spread * noise[idx]
        start += sizes[j]
    return q


def sample_mode_invent(ref, n, seed, n_replaced, invent_offset=14.0):
    """Q с заменой ровно n_replaced точек внешней симметричной парой мод (±invent_offset, 0).

    Заменённые точки делятся между двумя внешними модами поровну
    (ceil/floor), остальные n - n_replaced точек — из мод референса
    поровну (floor + остаток первым). Внешние моды лежат на оси x, их
    центры антисимметричны, поэтому сдвига среднего нет. Возвращает
    (points, replaced_mask): replaced_mask (n,) bool, True у заменённых
    точек; доля замен = mean(replaced_mask).

    ValueError: n_replaced вне [0, n]; при n_replaced > 0 внешние моды
    должны лежать вне референса (invent_offset > ref.radius + 3*spread,
    как в sample_q_dropping_invention).
    """
    if not 0 <= n_replaced <= n:
        raise ValueError(f'требуется 0 <= n_replaced <= n, получено {n_replaced} при n={n}')
    if n_replaced > 0 and invent_offset <= ref.radius + 3.0 * ref.spread:
        raise ValueError('invent_offset слишком мал: внешние моды пересекаются с референсными')
    d = ref.centers.shape[1]
    external = np.zeros((2, d))
    external[0, 0] = invent_offset
    external[1, 0] = -invent_offset
    centers = np.concatenate([ref.centers, external], axis=0)
    k_ref = ref.n_modes
    n_kept = n - n_replaced
    sizes = np.full(k_ref, n_kept // k_ref, dtype=int)
    sizes[:n_kept - sizes.sum()] += 1
    n_external = int(np.ceil(n_replaced / 2))
    sizes = np.concatenate([sizes, np.array([n_external, n_replaced - n_external], dtype=int)])
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(n, d))
    assign = rng.permutation(n)
    points = np.empty_like(noise)
    replaced_mask = np.zeros(n, dtype=bool)
    start = 0
    for j, size in enumerate(sizes):
        idx = assign[start:start + size]
        points[idx] = centers[j] + ref.spread * noise[idx]
        if j >= k_ref:
            replaced_mask[idx] = True
        start += size
    return points, replaced_mask
