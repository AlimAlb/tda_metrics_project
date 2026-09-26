"""Инварианты VAE/DCGAN: формы, детерминизм, диапазоны (малые прогоны на GPU/CPU)."""
import numpy as np
import pytest
import torch

from tda_metrics.models import (
    ConvVAE,
    Generator,
    Discriminator,
    train_vae,
    encode_vae,
    train_dcgan,
    sample_images,
    save_model,
    load_model,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture(scope='module')
def images():
    rng = np.random.default_rng(0)
    return (rng.random((256, 28, 28)) * 255).astype(np.uint8)


@pytest.fixture(scope='module')
def vae(images):
    return train_vae(images, latent_dim=16, epochs=1, batch_size=64, seed=42, device=DEVICE)


@pytest.fixture(scope='module')
def gan(images):
    return train_dcgan(images, epochs=1, batch_size=64, seed=7, device=DEVICE)


def test_vae_encode_shape(vae, images):
    latents = encode_vae(vae, images, device=DEVICE)
    assert latents.shape == (256, 16)


def test_vae_encode_deterministic(vae, images):
    first = encode_vae(vae, images, device=DEVICE)
    second = encode_vae(vae, images, device=DEVICE)
    assert np.array_equal(first, second)


def test_vae_decode_shape_and_range(vae, images):
    with torch.no_grad():
        x = torch.as_tensor(images / 255.0, dtype=torch.float32)[:, None].to(DEVICE)
        reconstruction = vae.decode(vae.encode(x)[0])
    assert reconstruction.shape == (256, 1, 28, 28)
    assert reconstruction.min() >= 0.0 and reconstruction.max() <= 1.0


def test_gan_sample_shape_dtype_range(gan):
    samples = sample_images(gan, 32, seed=7, device=DEVICE)
    assert samples.shape == (32, 28, 28)
    assert samples.dtype == np.uint8
    assert samples.min() >= 0 and samples.max() <= 255


def test_gan_sample_deterministic(gan):
    assert np.array_equal(
        sample_images(gan, 16, seed=7, device=DEVICE),
        sample_images(gan, 16, seed=7, device=DEVICE),
    )


def test_gan_different_seeds_differ(gan):
    assert not np.array_equal(
        sample_images(gan, 16, seed=7, device=DEVICE),
        sample_images(gan, 16, seed=43, device=DEVICE),
    )


def test_checkpoint_roundtrip(gan, tmp_path):
    path = str(tmp_path / 'g.pt')
    save_model(gan, path)
    loaded = load_model(Generator(100), path, device=DEVICE)
    assert np.array_equal(
        sample_images(gan, 8, seed=7, device=DEVICE),
        sample_images(loaded, 8, seed=7, device=DEVICE),
    )


def test_shapes_smoke():
    z = torch.randn(4, 100)
    assert Generator(100)(z).shape == (4, 1, 28, 28)
    x = torch.rand(4, 1, 28, 28)
    assert Discriminator()(x).shape == (4, 1)
    x_hat, mu, logvar = ConvVAE(16)(x)
    assert x_hat.shape == (4, 1, 28, 28)
    assert mu.shape == (4, 16) and logvar.shape == (4, 16)