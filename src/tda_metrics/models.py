"""Генеративные модели на MNIST 28x28: conv-VAE и DCGAN.

VAE используется в двух ролях: энкодер — «модельный» эмбеддер (латент 16-d,
сравним с PCA-сжатием CLIP), декодер — перенос латентного цикла в картинки.
DCGAN — генератор сэмплов для парадигмы «данные vs модель».
Требует torch; обучение и инференс детерминированы через seed.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ConvVAE', 'train_vae', 'encode_vae', 'Generator', 'Discriminator',
           'train_dcgan', 'sample_images', 'save_model', 'load_model']


# ---------------- VAE ----------------

class ConvVAE(nn.Module):
    """Conv-VAE для 28x28x1: энкодер -> латент (latent_dim), декодер обратно."""

    def __init__(self, latent_dim=16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 64 * 7 * 7)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid(),
        )
        self.latent_dim = latent_dim

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        return self.dec(self.fc_dec(z).view(-1, 64, 7, 7))

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decode(z), mu, logvar


def _to_tensor(images, device):
    x = torch.as_tensor(np.asarray(images), dtype=torch.float32) / 255.0
    if x.ndim == 3:
        x = x[:, None, :, :]
    return x.to(device)


def train_vae(images, latent_dim=16, epochs=10, lr=1e-3, batch_size=128,
              seed=42, device='cuda', checkpoint=None, log_every=None):
    """Обучает ConvVAE (MSE + KL); возвращает модель в eval-режиме.

    checkpoint=None — без сохранения, иначе путь .pt (при наличии файла модель
    загружается и обучение пропускается — кэш на VM).
    """
    torch.manual_seed(seed)
    if checkpoint is not None and os.path.exists(checkpoint):
        return load_model(ConvVAE(latent_dim), checkpoint, device)
    model = ConvVAE(latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x_all = _to_tensor(images, device)
    n = len(x_all)
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch_size):
            x = x_all[perm[i:i + batch_size]]
            x_hat, mu, logvar = model(x)
            mse = F.mse_loss(x_hat, x, reduction='sum') / len(x)
            kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            loss = mse + kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        if log_every is not None and (epoch + 1) % log_every == 0:
            print(f'VAE epoch {epoch + 1}/{epochs}: loss {total / n:.1f}', flush=True)
    if checkpoint is not None:
        save_model(model, checkpoint)
    return model.eval()


@torch.no_grad()
def encode_vae(model, images, device='cuda'):
    """Латенты (mu) энкодера: (n, latent_dim), numpy."""
    return model.encode(_to_tensor(images, device))[0].cpu().numpy()


# ---------------- DCGAN ----------------

class Generator(nn.Module):
    """z (z_dim) -> 28x28x1, выход в [-1, 1] (tanh)."""

    def __init__(self, z_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 256, 7, 1, 0), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 1, 3, 1, 1), nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z.view(-1, self.net[0].in_channels, 1, 1))


class Discriminator(nn.Module):
    """28x28x1 в [-1, 1] -> логит реальности."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_dcgan(images, z_dim=100, epochs=30, lr=2e-4, batch_size=128,
                 seed=7, device='cuda', checkpoint=None, log_every=None):
    """Обучает DCGAN (BCE с логитами, сглаживание меток); возвращает Generator в eval.

    checkpoint — путь .pt для G (кэш на VM: при наличии файла обучение пропускается).
    """
    torch.manual_seed(seed)
    if checkpoint is not None and os.path.exists(checkpoint):
        return load_model(Generator(z_dim), checkpoint, device)
    G, D = Generator(z_dim).to(device), Discriminator().to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    loss = nn.BCEWithLogitsLoss()
    x_all = _to_tensor(images, device) * 2.0 - 1.0
    n = len(x_all)
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            real = x_all[perm[i:i + batch_size]]
            b = len(real)
            z = torch.randn(b, z_dim, device=device)
            fake = G(z)

            opt_d.zero_grad()
            d_real = D(real).view(-1)
            d_fake = D(fake.detach()).view(-1)
            loss_d = loss(d_real, torch.full((b,), 0.9, device=device)) \
                + loss(d_fake, torch.zeros(b, device=device))
            loss_d.backward()
            opt_d.step()

            opt_g.zero_grad()
            loss_g = loss(D(fake).view(-1), torch.ones(b, device=device))
            loss_g.backward()
            opt_g.step()
        if log_every is not None and (epoch + 1) % log_every == 0:
            print(f'DCGAN epoch {epoch + 1}/{epochs}', flush=True)
    if checkpoint is not None:
        save_model(G, checkpoint)
    return G.eval()


@torch.no_grad()
def sample_images(G, n, z_dim=100, seed=7, device='cuda'):
    """Сэмплы генератора: uint8 (n, 28, 28), детерминированы seed."""
    torch.manual_seed(seed)
    z = torch.randn(n, z_dim, device=device)
    images = G(z).cpu().numpy()
    return ((images + 1.0) * 127.5).clip(0, 255).astype(np.uint8).squeeze(1)


# ---------------- чекпоинты ----------------

def save_model(model, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path, device='cuda'):
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()