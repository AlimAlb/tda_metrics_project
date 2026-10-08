"""Length-matched пулы пакетной задачи L9M: бины длины с общими квотами по трём сторонам.

Промпты фолда режутся перестановкой ``rng(seed_half)`` на две половины:
half_A даёт correct-строки P, half_B — correct-строки Q_correct и
hallucinated-строки H; P и Q_correct не пересекаются ни по строкам, ни
по промптам. Квантильные бины длины строятся по объединению длин трёх
пулов, ёмкость бина — минимум счётчиков сторон; при нехватке суммарной
ёмкости под n_cloud число бинов огрубляется по лестнице ``bin_ladder``.
``sample_length_matched_pools`` выдаёт выровненные по длине пулы,
``build_length_matched_packet`` собирает из них пакет (P, Q_alpha)
вложенной заменой Q_correct -> H (``make_alpha_replacement``): пакеты с
общими сидами и разными alpha имеют общий базис и вложенные замены.
"""
import numpy as np

from tda_metrics.mixtures import make_alpha_replacement

__all__ = [
    'length_bin_edges',
    'sample_length_matched_pools',
    'build_length_matched_packet',
]


def length_bin_edges(lengths, n_bins):
    """Квантильные границы бинов длины с бесконечными концами.

    ``edges = unique(quantile(lengths, linspace(0, 1, n_bins + 1)))``;
    первый и последний элементы заменяются на -inf/+inf — крайние
    значения попадают в крайние бины. Дубликаты квантилей при дискретных
    длинах схлопываются: фактических бинов ``len(edges) - 1`` может быть
    меньше ``n_bins``. Если все длины одинаковы, возвращается
    ``[-inf, +inf]`` — один общий бин. ValueError при ``n_bins < 1``.
    """
    if not isinstance(n_bins, (int, np.integer)) or int(n_bins) < 1:
        raise ValueError(f'n_bins должен быть целым >= 1: {n_bins!r}')
    lengths = np.asarray(lengths, dtype=float)
    edges = np.unique(np.quantile(lengths, np.linspace(0.0, 1.0, int(n_bins) + 1)))
    if len(edges) < 2:
        return np.array([-np.inf, np.inf])
    edges = edges.astype(float).copy()
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _bin_assignment(lengths, edges):
    """Номер бина для каждой длины: searchsorted side='right', клип в [0, len(edges) - 2].

    Значение ровно на внутренней границе уходит в бин правее; значения за
    пределами крайних бинов клипятся в крайние. Возвращает int64.
    """
    edges = np.asarray(edges)
    bin_ids = np.searchsorted(edges, np.asarray(lengths, dtype=float), side='right') - 1
    return np.clip(bin_ids, 0, len(edges) - 2).astype(np.int64)


def _level_fill(caps, n_target):
    """Максимально-равное заполнение ёмкостей: quota[b] <= caps[b], sum(quota) == n_target.

    Раунды: в каждом проходе добавляется 1 каждой ненасыщенной ёмкости
    (quota < cap) в порядке возрастания индекса; заполнение
    останавливается при достижении ``n_target`` — разброс квот
    ненасыщенных бинов не превышает 1. ValueError при ``n_target`` вне
    [0, sum(caps)].
    """
    caps = np.asarray(caps, dtype=np.int64)
    if not isinstance(n_target, (int, np.integer)):
        raise ValueError(f'n_target должен быть целым: {n_target!r}')
    n_target = int(n_target)
    if n_target < 0:
        raise ValueError(f'n_target должен быть >= 0: {n_target}')
    total_capacity = int(caps.sum())
    if n_target > total_capacity:
        raise ValueError(
            f'n_target ({n_target}) больше суммарной ёмкости ({total_capacity})')
    quota = np.zeros(len(caps), dtype=np.int64)
    total = 0
    while total < n_target:
        for b in range(len(caps)):
            if quota[b] < caps[b]:
                quota[b] += 1
                total += 1
                if total == n_target:
                    break
    return quota


def sample_length_matched_pools(
        correct_matrix, correct_index, correct_lengths, correct_prompts,
        hall_matrix, hall_index, hall_lengths, hall_prompts,
        n_cloud, seed_half, seed_sample, n_bins=10,
        bin_ladder=(10, 8, 6, 5, 4, 3, 2)):
    """Length-matched пулы фолда для пакетной задачи L9M: P, Q_correct и H с равными квотами по бинам длины.

    Промпты фолда (``set(correct_prompts) == set(hall_prompts)``) режутся
    перестановкой ``rng(seed_half)`` на половины: half_A — первые
    ``len(unique) // 2`` уникальных промптов в порядке перестановки,
    half_B — остальные (при нечётном числе промптов half_B на один
    больше). Пулы: A-correct — correct-строки промптов half_A (дают P),
    B-correct — correct-строки half_B (дают Q_correct), H —
    hallucinated-строки half_B; P и Q_correct не пересекаются ни по
    строкам, ни по промптам.

    Бины длины — квантильные по объединению длин трёх пулов
    (``length_bin_edges``); ёмкость бина — минимум счётчиков сторон.
    Кандидаты числа бинов: ``n_bins``, затем значения ``bin_ladder``
    меньше ``n_bins`` в порядке убывания (лестница огрубления). Первый
    кандидат с суммарной ёмкостью >= ``n_cloud`` берётся с квотами
    максимально-равного заполнения (``_level_fill``); если таких нет —
    кандидат с наибольшей ёмкостью целиком (``n_cloud_actual`` меньше
    запрошенного; ValueError при нулевой ёмкости). Выборка без
    возвращения одним генератором ``rng(seed_sample)``: бины по
    возрастанию, внутри бина стороны в порядке A-correct, B-correct, H;
    нулевая квота генератор не трогает. Квоты равны по трём сторонам в
    каждом бине, поэтому длины P, Q_correct и H совпадают по составу
    бинов.

    Возвращает dict с ключами: ``P``/``Q_correct``/``H`` — матрицы
    (n_cloud_actual, d); ``P_items``/``Q_correct_items``/``H_items`` —
    построчные кортежи (kind, глобальный item-индекс);
    ``P_positions``/``Q_correct_positions``/``H_positions`` — позиции
    строк в исходных матрицах;
    ``P_prompts``/``Q_prompts``/``H_prompts``;
    ``p_lengths``/``q_lengths``/``h_lengths``; ``bin_edges`` (±inf на
    концах); ``bin_quotas`` (list int); ``n_bins_actual``;
    ``n_cloud_actual``; ``diagnostics`` (mean/median/std длин сторон,
    ``smd_pq`` — стандартизованная разность средних P и Q_correct,
    ``prompt_half_sizes``); ``prompt_half_A``/``prompt_half_B``.
    ValueError при некорректных формах, несовпадении множеств промптов,
    n_cloud < 2, нецелых сидах или пустом пуле какой-либо стороны.
    """
    correct_matrix = np.asarray(correct_matrix)
    hall_matrix = np.asarray(hall_matrix)
    if correct_matrix.ndim != 2 or hall_matrix.ndim != 2:
        raise ValueError(
            f'матрицы должны быть 2D: correct {correct_matrix.shape}, '
            f'hall {hall_matrix.shape}')
    if correct_matrix.shape[1] != hall_matrix.shape[1]:
        raise ValueError(
            f'ширины матриц не совпадают: correct {correct_matrix.shape[1]}, '
            f'hall {hall_matrix.shape[1]}')
    correct_index = np.asarray(correct_index, dtype=np.int64)
    correct_lengths = np.asarray(correct_lengths, dtype=float)
    correct_prompts = np.asarray(correct_prompts)
    hall_index = np.asarray(hall_index, dtype=np.int64)
    hall_lengths = np.asarray(hall_lengths, dtype=float)
    hall_prompts = np.asarray(hall_prompts)
    if correct_index.ndim != 1 or len(correct_index) != len(correct_matrix):
        raise ValueError(
            f'correct_index должен быть 1D длиной len(correct_matrix): '
            f'{correct_index.shape} против {len(correct_matrix)}')
    if correct_lengths.ndim != 1 or len(correct_lengths) != len(correct_matrix):
        raise ValueError(
            f'correct_lengths должен быть 1D длиной len(correct_matrix): '
            f'{correct_lengths.shape} против {len(correct_matrix)}')
    if correct_prompts.ndim != 1 or len(correct_prompts) != len(correct_matrix):
        raise ValueError(
            f'correct_prompts должен быть 1D длиной len(correct_matrix): '
            f'{correct_prompts.shape} против {len(correct_matrix)}')
    if hall_index.ndim != 1 or len(hall_index) != len(hall_matrix):
        raise ValueError(
            f'hall_index должен быть 1D длиной len(hall_matrix): '
            f'{hall_index.shape} против {len(hall_matrix)}')
    if hall_lengths.ndim != 1 or len(hall_lengths) != len(hall_matrix):
        raise ValueError(
            f'hall_lengths должен быть 1D длиной len(hall_matrix): '
            f'{hall_lengths.shape} против {len(hall_matrix)}')
    if hall_prompts.ndim != 1 or len(hall_prompts) != len(hall_matrix):
        raise ValueError(
            f'hall_prompts должен быть 1D длиной len(hall_matrix): '
            f'{hall_prompts.shape} против {len(hall_matrix)}')
    if not isinstance(n_cloud, (int, np.integer)) or int(n_cloud) < 2:
        raise ValueError(f'n_cloud должен быть целым >= 2: {n_cloud!r}')
    n_cloud = int(n_cloud)
    if not isinstance(seed_half, (int, np.integer)) or not isinstance(
            seed_sample, (int, np.integer)):
        raise ValueError(
            f'seed_half и seed_sample должны быть int: {seed_half!r}, {seed_sample!r}')
    if not isinstance(n_bins, (int, np.integer)) or int(n_bins) < 1:
        raise ValueError(f'n_bins должен быть целым >= 1: {n_bins!r}')
    n_bins = int(n_bins)
    if set(correct_prompts.tolist()) != set(hall_prompts.tolist()):
        raise ValueError(
            'множества промптов correct и hallucinated не совпадают — '
            'пулы фолда должны быть выровнены по промптам')

    unique_prompts = sorted(set(correct_prompts.tolist()))
    order = np.random.default_rng(seed_half).permutation(len(unique_prompts))
    shuffled = [unique_prompts[position] for position in order]
    n_half = len(unique_prompts) // 2
    half_a = sorted(shuffled[:n_half])
    half_b = sorted(shuffled[n_half:])

    pos_ac = np.flatnonzero(np.isin(correct_prompts, half_a)).astype(np.int64)
    pos_qb = np.flatnonzero(np.isin(correct_prompts, half_b)).astype(np.int64)
    pos_h = np.flatnonzero(np.isin(hall_prompts, half_b)).astype(np.int64)
    if len(pos_ac) == 0 or len(pos_qb) == 0 or len(pos_h) == 0:
        raise ValueError(
            f'один из пулов пуст: A-correct {len(pos_ac)}, B-correct '
            f'{len(pos_qb)}, H {len(pos_h)} — проверьте половины промптов')

    lengths_ac = correct_lengths[pos_ac]
    lengths_qb = correct_lengths[pos_qb]
    lengths_h = hall_lengths[pos_h]
    all_lengths = np.concatenate([lengths_ac, lengths_qb, lengths_h])

    candidates = [n_bins]
    for value in bin_ladder:
        if value < n_bins and value not in candidates:
            candidates.append(value)

    chosen = None
    evaluated = []
    for k in candidates:
        edges_k = length_bin_edges(all_lengths, k)
        n_bins_k = len(edges_k) - 1
        counts_ac = np.bincount(_bin_assignment(lengths_ac, edges_k), minlength=n_bins_k)
        counts_qb = np.bincount(_bin_assignment(lengths_qb, edges_k), minlength=n_bins_k)
        counts_h = np.bincount(_bin_assignment(lengths_h, edges_k), minlength=n_bins_k)
        caps_k = np.minimum(np.minimum(counts_ac, counts_qb), counts_h).astype(np.int64)
        total_k = int(caps_k.sum())
        if total_k >= n_cloud:
            chosen = (edges_k, caps_k, n_cloud)
            break
        evaluated.append((total_k, caps_k, edges_k))
    if chosen is None:
        best = evaluated[0]
        for item in evaluated[1:]:
            if item[0] > best[0]:
                best = item
        if best[0] == 0:
            raise ValueError(
                'ни один вариант биннинга не даёт ни одной общей позиции '
                'трёх пулов — длины сторон не пересекаются')
        chosen = (best[2], best[1], best[0])
    edges, caps, n_cloud_actual = chosen
    if n_cloud_actual == n_cloud:
        quotas = _level_fill(caps, n_cloud)
    else:
        quotas = caps.astype(np.int64).copy()
    n_bins_actual = len(edges) - 1

    ids_ac = _bin_assignment(lengths_ac, edges)
    ids_qb = _bin_assignment(lengths_qb, edges)
    ids_h = _bin_assignment(lengths_h, edges)

    rng = np.random.default_rng(seed_sample)
    sides = ((pos_ac, ids_ac), (pos_qb, ids_qb), (pos_h, ids_h))
    picked_positions = [[], [], []]
    picked_relative = [[], [], []]
    for b in range(n_bins_actual):
        for side, (positions, ids) in enumerate(sides):
            if quotas[b] > 0:
                relative_b = np.flatnonzero(ids == b)
                positions_b = positions[relative_b]
                drawn = rng.choice(len(positions_b), size=int(quotas[b]), replace=False)
                picked_positions[side].append(positions_b[drawn])
                picked_relative[side].append(relative_b[drawn])
    p_positions = np.concatenate(picked_positions[0])
    qb_positions = np.concatenate(picked_positions[1])
    h_positions = np.concatenate(picked_positions[2])
    p_relative = np.concatenate(picked_relative[0])
    qb_relative = np.concatenate(picked_relative[1])
    h_relative = np.concatenate(picked_relative[2])

    assert len(np.unique(p_positions)) == len(p_positions)
    assert len(np.unique(qb_positions)) == len(qb_positions)
    assert len(np.unique(h_positions)) == len(h_positions)
    for ids, relative in (
            (ids_ac, p_relative), (ids_qb, qb_relative), (ids_h, h_relative)):
        assert np.array_equal(
            np.bincount(ids[relative], minlength=n_bins_actual), quotas)

    p_items = [('correct', int(correct_index[position])) for position in p_positions]
    q_correct_items = [
        ('correct', int(correct_index[position])) for position in qb_positions]
    h_items = [
        ('hallucinated', int(hall_index[position])) for position in h_positions]
    assert set(p_items).isdisjoint(q_correct_items)

    p_lengths = correct_lengths[p_positions]
    q_lengths = correct_lengths[qb_positions]
    h_lengths_selected = hall_lengths[h_positions]
    p_prompts = correct_prompts[p_positions]
    q_prompts = correct_prompts[qb_positions]
    h_prompts_selected = hall_prompts[h_positions]

    mean_p = float(np.mean(p_lengths))
    mean_q = float(np.mean(q_lengths))
    var_p = float(np.var(p_lengths))
    var_q = float(np.var(q_lengths))
    denom = (var_p + var_q) / 2.0
    smd_pq = (mean_q - mean_p) / np.sqrt(denom) if denom > 0.0 else 0.0
    diagnostics = {
        'p_mean': mean_p,
        'p_median': float(np.median(p_lengths)),
        'p_std': float(np.std(p_lengths)),
        'q_mean': mean_q,
        'q_median': float(np.median(q_lengths)),
        'q_std': float(np.std(q_lengths)),
        'h_mean': float(np.mean(h_lengths_selected)),
        'h_median': float(np.median(h_lengths_selected)),
        'h_std': float(np.std(h_lengths_selected)),
        'smd_pq': float(smd_pq),
        'prompt_half_sizes': {'A': len(half_a), 'B': len(half_b)},
    }
    return {
        'P': correct_matrix[p_positions],
        'Q_correct': correct_matrix[qb_positions],
        'H': hall_matrix[h_positions],
        'P_items': p_items,
        'Q_correct_items': q_correct_items,
        'H_items': h_items,
        'P_positions': p_positions,
        'Q_correct_positions': qb_positions,
        'H_positions': h_positions,
        'P_prompts': p_prompts,
        'Q_prompts': q_prompts,
        'H_prompts': h_prompts_selected,
        'p_lengths': p_lengths,
        'q_lengths': q_lengths,
        'h_lengths': h_lengths_selected,
        'bin_edges': edges,
        'bin_quotas': [int(value) for value in quotas],
        'n_bins_actual': int(n_bins_actual),
        'n_cloud_actual': int(n_cloud_actual),
        'diagnostics': diagnostics,
        'prompt_half_A': half_a,
        'prompt_half_B': half_b,
    }


def build_length_matched_packet(pools, alpha, seed_mix):
    """Пакет (P, Q_alpha) из length-matched пулов: вложенная замена Q_correct -> H.

    ``pools`` — результат ``sample_length_matched_pools``; ``Q =
    make_alpha_replacement(Q_correct, H, alpha, permutation)`` с
    перестановкой ``rng(seed_mix).permutation(n_cloud_actual)`` — при
    общих сидах замены соседних alpha вложены (мастер-план §L4). Состав
    ``Q_items``: позиция pos получает ``H_items[pos]``, если pos входит
    в первые ``round(alpha*n)`` позиций перестановки, иначе
    ``Q_correct_items[pos]``; ``q_lengths`` выбирается из ``h_lengths``/
    ``q_lengths`` пула тем же правилом. При alpha=0 Q — чистый Q_correct
    (независимая от P same-law выборка), при alpha=1 — H. ValueError при
    alpha вне [0, 1].
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f'alpha должен быть в [0, 1]: {alpha!r}')
    n = int(pools['n_cloud_actual'])
    permutation = np.random.default_rng(seed_mix).permutation(n)
    n_hall = int(round(alpha * n))
    q_cloud = make_alpha_replacement(pools['Q_correct'], pools['H'], alpha, permutation)
    replaced = np.zeros(n, dtype=bool)
    replaced[permutation[:n_hall]] = True
    q_items = []
    q_lengths = []
    for position in range(n):
        if replaced[position]:
            q_items.append(pools['H_items'][position])
            q_lengths.append(float(pools['h_lengths'][position]))
        else:
            q_items.append(pools['Q_correct_items'][position])
            q_lengths.append(float(pools['q_lengths'][position]))
    return {
        'P': pools['P'],
        'Q': q_cloud,
        'P_items': pools['P_items'],
        'Q_items': q_items,
        'n_hall': n_hall,
        'n_cloud_actual': n,
        'alpha': float(alpha),
        'q_lengths': np.asarray(q_lengths, dtype=float),
        'p_lengths': pools['p_lengths'],
    }
