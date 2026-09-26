"""Снижение размерности признаков (PCA через SVD).

Мотивация: в 512-мерных CLIP-эмбеддингах расстояния концентрируются и
топологические метрики дискриминируют слабо; сжатие до ~16 главных компонент
снимает часть проблемы и заодно ускоряет RTD/MTD.
"""
import numpy as np

__all__ = ['pca_fit', 'pca_transform']


def pca_fit(X, n_components=16):
    """Обучает PCA на X; возвращает (mean, components, explained_ratio).

    explained_ratio — доли дисперсии всех компонент (по убыванию), первые
    n_components штук идут в components.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError('ожидается 2D-массив (n, d)')
    max_components = min(len(X), X.shape[1])
    if not 1 <= n_components <= max_components:
        raise ValueError(f'n_components={n_components} вне диапазона [1, {max_components}]')
    mean = X.mean(axis=0)
    _, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    variances = (S ** 2) / max(len(X) - 1, 1)
    ratio = variances / variances.sum()
    return mean, Vt[:n_components], ratio


def pca_transform(X, model=None, n_components=16):
    """Сжатие X до n_components; модель обучается на X.

    Если передана модель из `pca_fit` (mean, components, ratio), X проецируется
    на неё — так «модель» Q трансформируется той же матрицей, что и «данные» P.
    Возвращает (проекция, {'explained': первые k долей, 'total': их сумма}).
    """
    if model is None:
        model = pca_fit(X, n_components=n_components)
    mean, components, ratio = model
    projection = (np.asarray(X, dtype=float) - mean) @ components.T
    k = components.shape[0]
    info = {'explained': ratio[:k], 'total': float(np.sum(ratio[:k]))}
    return projection, info