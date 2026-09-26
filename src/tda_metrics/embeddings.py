"""CLIP-эмбеддинги картинок: извлечение, кэш, сабсэмплинг.

Требует torch и openai-clip (устанавливаются в ноутбуке). Общий `forward`
позволяет прогонять картинки через произвольный бэкбон (CLIP, DINO, автоэнкодер).
"""
import os

import numpy as np
import torch
import clip
from PIL import Image

__all__ = ['to_rgb', 'backbone_embeddings', 'clip_embeddings', 'get_embeddings', 'subsample']


def to_rgb(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    return image.convert('RGB')


@torch.no_grad()
def backbone_embeddings(images, model, preprocess, device, forward=None, batch_size=256, desc='backbone'):
    """L2-нормализованные фичи произвольной модели, shape (n, d).

    forward(model, batch_on_device) -> тензор фич; по умолчанию
    model.encode_image (CLIP). Кэш в `embeddings/` не ведется — см. get_embeddings.
    """
    if forward is None:
        forward = lambda m, batch: m.encode_image(batch)
    features = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack([preprocess(to_rgb(img)) for img in images[i:i + batch_size]])
        features.append(forward(model, batch.to(device)).cpu().float().numpy())
    features = np.concatenate(features)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    return features


def clip_embeddings(images, model, preprocess, device, batch_size=256, desc='CLIP'):
    """L2-нормализованные CLIP-эмбеддинги, shape (n, 512)."""
    return backbone_embeddings(images, model, preprocess, device, batch_size=batch_size, desc=desc)


def get_embeddings(name, images, model, preprocess, device, forward=None, cache_dir='embeddings'):
    """Извлекает и кэширует эмбеддинги (npy в cache_dir); forward — как в backbone_embeddings."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f'{name}.npy')
    if os.path.exists(path):
        return np.load(path)
    features = backbone_embeddings(images, model, preprocess, device, forward=forward, desc=name)
    np.save(path, features)
    return features


def subsample(X, n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    return X[idx]