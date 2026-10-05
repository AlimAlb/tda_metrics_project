"""Строители смесей для экспериментов mode dropping × invention (мастер-план §Часть I/S3, §L4).

Синтетический референс — несколько хорошо разделимых мод, центры равномерно
по кругу; внешние (invented) моды добавляются строго симметричными парами,
чтобы не создавать случайный сдвиг среднего. Метод общих случайных чисел:
базовый шум и перестановка присвоения генерируются от одного seed — при
изменении долей drop/invent точки пересобираются из того же шума, меняется
только мода-присвоение. `make_alpha_replacement` — каркас L4-смесей
(вложенные замены по общей перестановке test-ID).
"""
from dataclasses import dataclass

import numpy as np

__all__ = [
    'ModeReference',
    'make_mode_reference',
    'sample_reference',
    'sample_q_dropping_invention',
    'make_alpha_replacement',
    'nested_alphas',
]


@dataclass(frozen=True)
class ModeReference:
    """Описание референсного распределения: центры мод, разброс, радиус кольца центров."""

    centers: np.ndarray
    spread: float
    radius: float

    @property
    def n_modes(self):
        return int(self.centers.shape[0])


def make_mode_reference(n_modes=8, radius=8.0, spread=1.0):
    """Центры n_modes мод равномерно по кругу радиуса radius (среднее центров = 0).

    n_modes рекомендуется кратным 4 — сетка drop/invent {0, .25, .5, .75} даёт
    целые числа удалённых/добавленных мод.
    """
    if n_modes < 2:
        raise ValueError('n_modes должен быть >= 2')
    if radius <= 0 or spread <= 0:
        raise ValueError('radius и spread должны быть положительными')
    angles = np.linspace(0.0, 2.0 * np.pi, n_modes, endpoint=False)
    centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    return ModeReference(centers=centers, spread=float(spread), radius=float(radius))


def sample_q_dropping_invention(
    ref,
    n,
    seed,
    drop_fraction=0.0,
    invent_fraction=0.0,
    invent_offset=14.0,
):
    """Выборка Q(d, i) из референса: drop — доля удаляемых последних мод, invent — пары внешних мод.

    Удаляются последние по индексу моды, веса оставшихся перенормируются
    поровну. Внешние моды — строго антисимметричные пары на радиусе
    invent_offset (среднее центров внешних мод равно нулю). Размер выборки
    всегда n. Общие случайные числа: базовый шум (n, 2) и перестановка
    присвоения берутся от одного seed и не зависят от (d, i) — соседние
    конфигурации различаются только мода-присвоением.
    """
    if not 0.0 <= drop_fraction < 1.0:
        raise ValueError('drop_fraction должен быть в [0, 1)')
    if not 0.0 <= invent_fraction <= 1.0:
        raise ValueError('invent_fraction должен быть в [0, 1]')
    n_modes = ref.n_modes
    n_dropped = int(round(drop_fraction * n_modes))
    n_dropped = min(n_dropped, n_modes - 1)
    n_pairs = int(round(invent_fraction * n_modes / 2.0))
    if n_pairs > 0 and invent_offset <= ref.radius + 3.0 * ref.spread:
        raise ValueError(
            'invent_offset слишком мал: внешние моды пересекаются с референсными'
        )

    kept = ref.centers[:n_modes - n_dropped]
    pair_angles = np.pi * (np.arange(n_pairs) + 0.5) / max(n_pairs, 1) + np.pi / 4.0
    directions = np.stack([np.cos(pair_angles), np.sin(pair_angles)], axis=1)
    invented = invent_offset * np.concatenate([directions, -directions], axis=0)
    active = np.concatenate([kept, invented], axis=0) if n_pairs > 0 else kept
    k = active.shape[0]

    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(n, ref.centers.shape[1]))
    assign = rng.permutation(n)

    sizes = np.full(k, n // k, dtype=int)
    sizes[:n - sizes.sum()] += 1
    q = np.empty_like(noise)
    start = 0
    for j in range(k):
        idx = assign[start:start + sizes[j]]
        q[idx] = active[j] + ref.spread * noise[idx]
        start += sizes[j]
    return q


def sample_reference(ref, n, seed):
    """Выборка из референса без деформаций — эквивалент Q(drop=0, invent=0)."""
    return sample_q_dropping_invention(ref, n, seed)


def make_alpha_replacement(correct, hallucinated, alpha, permutation):
    """Вложенная замена: первые round(alpha·n) позиций permutation получают hallucinated, остальные correct.

    Общая перестановка делает замены соседних alpha вложенными: замены при
    большем alpha содержат замены при меньшем (мастер-план §L4).
    """
    correct = np.asarray(correct)
    hallucinated = np.asarray(hallucinated)
    if correct.shape != hallucinated.shape:
        raise ValueError('correct и hallucinated должны иметь одинаковую форму')
    permutation = np.asarray(permutation)
    n = correct.shape[0]
    if permutation.shape != (n,) or set(permutation.tolist()) != set(range(n)):
        raise ValueError('permutation должен быть перестановкой range(n)')
    if not 0.0 <= alpha <= 1.0:
        raise ValueError('alpha должен быть в [0, 1]')
    k = int(round(alpha * n))
    out = correct.copy()
    idx = permutation[:k]
    out[idx] = hallucinated[idx]
    return out


def nested_alphas(permutation, alphas):
    """Маски замен для списка alpha по общей перестановке (вложенные prefix-множества)."""
    permutation = np.asarray(permutation)
    n = permutation.shape[0]
    if set(permutation.tolist()) != set(range(n)):
        raise ValueError('permutation должен быть перестановкой range(n)')
    masks = {}
    for alpha in alphas:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError('alpha должен быть в [0, 1]')
        mask = np.zeros(n, dtype=bool)
        mask[permutation[:int(round(alpha * n))]] = True
        masks[alpha] = mask
    return masks