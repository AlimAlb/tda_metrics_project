"""Параметрические датасеты на основе MNIST: вращение и масштаб.

Вариация вдоль управляемого параметра (угол / масштаб) порождает семейство
картинок; полный проход по углу 0..360 замыкает цикл в пространстве признаков,
тогда как масштабный свип топологически эквивалентен отрезку (негативный
контроль). torchvision подгружается лениво — только внутри `load_mnist`.
"""
import numpy as np
from PIL import Image

__all__ = [
    'load_mnist',
    'select_class',
    'rotate_images',
    'zoom_images',
    'sweep_images',
    'angles_grid',
    'scale_grid',
]

_RESAMPLE = getattr(Image, 'Resampling', Image).BICUBIC


def load_mnist(root='data', train=True, download=True):
    """MNIST как массивы картинок (n, 28, 28) uint8 и меток (n,)."""
    from torchvision import datasets
    mnist = datasets.MNIST(root, train=train, download=download)
    return mnist.data.numpy(), mnist.targets.numpy()


def select_class(images, labels, digit, n, seed):
    """n картинок заданного класса, выбранных детерминированно через seed."""
    idx = np.flatnonzero(labels == digit)
    rng = np.random.default_rng(seed)
    return images[rng.choice(idx, size=min(n, len(idx)), replace=False)]


def angles_grid(step_deg=10.0, start=0.0, end=360.0):
    """Равномерная сетка углов [start, end) с шагом step_deg."""
    return np.arange(start, end, step_deg)


def scale_grid(lo=0.6, hi=1.4, n=9):
    """Равномерная сетка масштабов [lo, hi] из n значений."""
    return np.linspace(lo, hi, n)


def rotate_images(images, angle_deg):
    """Поворот каждой картинки на angle_deg (PIL, bicubic, фон = 0).

    Детерминирован; поворот на 0 — тождественное преобразование.
    """
    out = np.empty_like(images)
    for i, img in enumerate(images):
        pil = Image.fromarray(img)
        out[i] = np.asarray(pil.rotate(angle_deg, resample=_RESAMPLE, fillcolor=0))
    return out


def zoom_images(images, scale):
    """Масштабирование с сохранением холста 28x28 (при scale>1 — центральный кроп)."""
    out = np.empty_like(images)
    h, w = images.shape[1:3]
    for i, img in enumerate(images):
        pil = Image.fromarray(img)
        if scale >= 1.0:
            big = pil.resize((int(w * scale), int(h * scale)), resample=_RESAMPLE)
            left = (big.width - w) // 2
            top = (big.height - h) // 2
            out[i] = np.asarray(big.crop((left, top, left + w, top + h)))
        else:
            small = pil.resize((max(1, int(w * scale)), max(1, int(h * scale))), resample=_RESAMPLE)
            canvas = Image.new('L', (w, h), 0)
            canvas.paste(small, ((w - small.width) // 2, (h - small.height) // 2))
            out[i] = np.asarray(canvas)
    return out


def sweep_images(images, values, transform):
    """Декартово произведение: каждая картинка × каждое значение параметра.

    transform(images, value) -> тот же массив картинок после преобразования.
    Порядок строк — n-major: строка i * len(values) + j соответствует картинке i
    при значении values[j]; форма результата (n * k, 28, 28).
    """
    swept = np.stack([transform(images, value) for value in values], axis=1)
    return swept.reshape(len(images) * len(values), *images.shape[1:])