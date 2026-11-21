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
