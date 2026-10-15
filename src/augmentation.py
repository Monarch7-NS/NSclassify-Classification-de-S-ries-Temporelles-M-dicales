"""Augmentations en ligne, appliquées uniquement aux batchs d'entraînement (tenseurs [B, 1, L])."""
import torch


def add_jitter(x, sigma_range=(0.01, 0.03), p=0.5):
    sigma = torch.empty(x.shape[0], 1, 1, device=x.device).uniform_(*sigma_range)
    mask = (torch.rand(x.shape[0], 1, 1, device=x.device) < p).float()
    return x + mask * sigma * torch.randn_like(x)


def amplitude_scale(x, scale_range=(0.95, 1.05), p=0.5):
    scale = torch.empty(x.shape[0], 1, 1, device=x.device).uniform_(*scale_range)
    mask = torch.rand(x.shape[0], 1, 1, device=x.device) < p
    return torch.where(mask, x * scale, x)


def temporal_shift(x, max_shift=4, p=0.5):
    """Décalage de ±max_shift échantillons, bords répliqués (pas de valeurs inventées)."""
    b, _, length = x.shape
    shift = torch.randint(-max_shift, max_shift + 1, (b, 1), device=x.device)
    shift = shift * (torch.rand(b, 1, device=x.device) < p).long()
    idx = (torch.arange(length, device=x.device)[None, :] + shift).clamp(0, length - 1)
    return torch.gather(x, 2, idx[:, None, :])


def augment(x, sigma_range=(0.01, 0.03), scale_range=(0.95, 1.05), max_shift=4, p=0.5):
    """Variante 'mild' par défaut : bruit gaussien, échelle d'amplitude, décalage temporel."""
    x = add_jitter(x, sigma_range, p)
    x = amplitude_scale(x, scale_range, p)
    return temporal_shift(x, max_shift, p)
