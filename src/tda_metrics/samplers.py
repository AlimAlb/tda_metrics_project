"""Сэмплеры облаков точек для управляемых 2D-экспериментов.

Все функции детерминированы через `seed` (`np.random.default_rng`) и
возвращают массив формы (n, dim).
"""
import numpy as np

__all__ = [
    'sample_gaussian',
    'sample_uniform_cube',
    'sample_ball',
    'sample_ring',
    'sample_gaussian_mixture',
    'translate_cloud',
    'rotate_cloud',
    'reflect_cloud',
    'scale_cloud',
    'stretch_cloud',
]


def sample_gaussian(n, dim=2, mean=None, cov=None, seed=None):
    """Точки из N(mean, cov)."""
    rng = np.random.default_rng(seed)
    mean = np.zeros(dim) if mean is None else np.asarray(mean, dtype=float)
    cov = np.eye(dim) if cov is None else np.asarray(cov, dtype=float)
    return rng.multivariate_normal(mean, cov, size=n)


def sample_uniform_cube(n, dim=2, side=1.0, seed=None):
    """Равномерно в кубе [-side/2, side/2]^dim (dim=2, side=1 — единичный квадрат)."""
    rng = np.random.default_rng(seed)
    return (rng.random((n, dim)) - 0.5) * side


def sample_ball(n, dim=2, radius=1.0, center=None, seed=None):
    """Равномерно по объему шара радиуса radius (в 2D — диск).

    Радиус сэмплируется как r = R * U^(1/dim): тогда P(r' < r) = (r/R)^dim,
    т.е. точки заполняют шар равномерно, а не сгустаются у центра.
    """
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = radius * rng.random(n) ** (1.0 / dim)
    points = directions * radii[:, None]
    if center is not None:
        points = points + np.asarray(center, dtype=float)
    return points


def sample_ring(n, dim=2, radius=1.0, thickness=0.0, center=None, seed=None):
    """Толстое кольцо: направление равномерно на сфере, радиус ~ N(radius, thickness/4).

    thickness задает эффективную толщину ±2σ (правило 4σ); при thickness=0 — окружность.
    """
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = np.maximum(rng.normal(radius, thickness / 4.0, size=n), 0.0)
    points = directions * radii[:, None]
    if center is not None:
        points = points + np.asarray(center, dtype=float)
    return points


def sample_gaussian_mixture(n, radius=0.0, n_components=4, mean=(0.0, 0.0), cov=None, seed=None):
    """Смесь равновесых 2D-гауссиан с сохранением глобальных моментов.

    Центры компонент лежат в whitened-пространстве на окружности радиуса `radius`
    (равномерно по углу), дисперсия каждой компоненты sigma_c^2 = 1 - radius^2 / 2
    подобрана так, что итоговое распределение имеет те же mean и cov, что и
    N(mean, cov). При radius=0 вырождается в обычную гауссиану.

    Ограничения: radius^2 < 2; n_components = 1 или >= 3 — при двух компонентах
    изотропность ковариации нарушается.
    """
    rng = np.random.default_rng(seed)
    mean = np.asarray(mean, dtype=float)
    cov = np.eye(2) if cov is None else np.asarray(cov, dtype=float)
    if mean.shape != (2,) or cov.shape != (2, 2):
        raise ValueError('смесь определена только в 2D (mean (2,), cov (2, 2))')
    if n_components < 1:
        raise ValueError('n_components должен быть >= 1')
    if n_components == 1:
        if radius != 0.0:
            raise ValueError('n_components=1 поддерживается только при radius=0')
        return rng.multivariate_normal(mean, cov, size=n)
    if n_components == 2:
        raise ValueError('n_components=2 не поддерживается: нарушается изотропность ковариации')
    if radius ** 2 >= 2.0:
        raise ValueError('radius^2 должен быть < 2')

    angles = np.linspace(0.0, 2.0 * np.pi, n_components, endpoint=False)
    centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    std_component = np.sqrt(1.0 - 0.5 * radius ** 2)

    comp_idx = rng.integers(0, n_components, size=n)
    z = centers[comp_idx] + rng.normal(size=(n, 2)) * std_component

    transform = np.linalg.cholesky(cov)
    return mean + z @ transform.T


def translate_cloud(X, shift):
    """Перенос облака на вектор shift (изометрия: внутренние расстояния не меняются)."""
    X = np.asarray(X, dtype=float)
    shift = np.asarray(shift, dtype=float)
    if shift.shape != X.shape[1:]:
        raise ValueError('shift должен иметь форму (dim,)')
    return X + shift


def rotate_cloud(X, angle_deg):
    """Поворот облака вокруг нуля на angle_deg градусов (изометрия)."""
    X = np.asarray(X, dtype=float)
    angle = np.deg2rad(angle_deg)
    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    return X @ rotation.T


def reflect_cloud(X, axis='x'):
    """Отражение облака относительно оси: 'x' — знак y-координаты, 'y' — знак x-координаты."""
    X = np.asarray(X, dtype=float)
    if axis not in ('x', 'y'):
        raise ValueError("axis должен быть 'x' или 'y'")
    reflected = X.copy()
    if axis == 'x':
        reflected[:, 1] = -reflected[:, 1]
    else:
        reflected[:, 0] = -reflected[:, 0]
    return reflected


def scale_cloud(X, factor):
    """Равномерное масштабирование облака вокруг нуля (внутренние расстояния умножаются на factor)."""
    X = np.asarray(X, dtype=float)
    if factor <= 0:
        raise ValueError('factor должен быть положительным')
    return X * float(factor)


def stretch_cloud(X, factor_x, factor_y=1.0):
    """Анизотропное растяжение облака по осям (не изометрия: меняет внутренние расстояния)."""
    X = np.asarray(X, dtype=float)
    scale = np.array([float(factor_x), float(factor_y)])
    if np.any(scale <= 0):
        raise ValueError('факторы растяжения должны быть положительными')
    return X * scale