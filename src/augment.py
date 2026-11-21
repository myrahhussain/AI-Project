"""Stain colour augmentation implemented from scratch in PyTorch.

HED augmentation (Tellez et al., 2019): convert RGB to the Haematoxylin /
Eosin / DAB stain space by colour deconvolution, randomly rescale and shift
each stain channel, and convert back to RGB. The conversion uses the same
Ruifrok-Johnston matrix and log transform as scikit-image's rgb2hed / hed2rgb,
which the tests check against.

All functions take float tensors shaped (N, 3, H, W) with values in [0, 1] and
run on whichever device the tensor lives on, so they work on GPU batches.
"""
from __future__ import annotations

import math

import torch

# Rows are the stain optical-density vectors (H, E, DAB), as in scikit-image.
_RGB_FROM_HED = [
    [0.65, 0.70, 0.29],
    [0.07, 0.99, 0.11],
    [0.27, 0.57, 0.78],
]
_EPS = 1e-6


def _mats(x: torch.Tensor):
    rgb_from_hed = torch.tensor(_RGB_FROM_HED, dtype=x.dtype, device=x.device)
    return rgb_from_hed, torch.linalg.inv(rgb_from_hed)


def rgb_to_hed(x: torch.Tensor) -> torch.Tensor:
    """(N,3,H,W) RGB in [0,1] -> (N,H,W,3) stain concentrations (H, E, DAB), >= 0."""
    _, hed_from_rgb = _mats(x)
    x = x.clamp_min(_EPS)
    log_adjust = math.log(_EPS)
    stains = (torch.log(x) / log_adjust).permute(0, 2, 3, 1) @ hed_from_rgb
    return stains.clamp_min(0)


def hed_to_rgb(stains: torch.Tensor) -> torch.Tensor:
    """(N,H,W,3) stain concentrations -> (N,3,H,W) RGB in [0,1]."""
    rgb_from_hed, _ = _mats(stains)
    log_adjust = -math.log(_EPS)
    log_rgb = -(stains * log_adjust) @ rgb_from_hed
    return torch.exp(log_rgb).clamp(0, 1).permute(0, 3, 1, 2)


def hed_augment(x: torch.Tensor, sigma: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Randomly perturb the H, E and DAB channels of each image.

    For every image and every stain channel s:  s' = alpha * s + beta, with
    alpha ~ U(1 - sigma, 1 + sigma) and beta ~ U(-sigma, sigma).
    Tellez et al. call sigma ~0.05 "light" and ~0.2 "strong".
    `generator` must be a CPU generator (random numbers are drawn on CPU, then
    moved to the tensor's device, so seeding behaves the same on CPU and GPU).
    """
    if sigma == 0:
        return x
    n = x.shape[0]
    alpha = 1 + (torch.rand(n, 1, 1, 3, generator=generator) * 2 - 1) * sigma
    beta = (torch.rand(n, 1, 1, 3, generator=generator) * 2 - 1) * sigma
    alpha, beta = alpha.to(x.device, x.dtype), beta.to(x.device, x.dtype)
    stains = rgb_to_hed(x)
    return hed_to_rgb(alpha * stains + beta)


def flip_rot90(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Random member of the 8 flips / 90-degree rotations, chosen per image."""
    n = x.shape[0]
    r = torch.randint(0, 2, (n, 3), generator=generator).bool().to(x.device)
    x = torch.where(r[:, 0].view(n, 1, 1, 1), x.flip(3), x)
    x = torch.where(r[:, 1].view(n, 1, 1, 1), x.flip(2), x)
    x = torch.where(r[:, 2].view(n, 1, 1, 1), x.transpose(2, 3), x)
    return x
