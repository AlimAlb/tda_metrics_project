"""CLIP-эмбеддинги картинок: извлечение, кэш, сабсэмплинг.

Требует torch и openai-clip (устанавливаются в ноутбуке).
"""
import os

import numpy as np
import torch
import clip
from PIL import Image

__all__ = ['to_rgb', 'clip_embeddings', 'get_embeddings', 'subsample']


def to_rgb(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    return image.convert('RGB')


@torch.no_grad()
def clip_embeddings(images, model, preprocess, device, batch_size=256, desc='CLIP'):
    """L2-нормализованные эмбеддинги, shape (n, 512)."""
    features = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack([preprocess(to_rgb(img)) for img in images[i:i + batch_size]])
        features.append(model.encode_image(batch.to(device)).cpu().float().numpy())
    features = np.concatenate(features)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    return features


def get_embeddings(name, images, model, preprocess, device, cache_dir='embeddings'):
    """Извлекает и кэширует эмбеддинги (npy в cache_dir)."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f'{name}.npy')
    if os.path.exists(path):
        return np.load(path)
    features = clip_embeddings(images, model, preprocess, device, desc=name)
    np.save(path, features)
    return features


def subsample(X, n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    return X[idx]