"""One training / evaluation routine shared by every method (matched budgets).

Methods
  baseline : flips + 90-degree rotations only
  hed      : baseline + HED stain augmentation (strength sigma)
  vahadane : baseline, on stain-normalised patches
  both     : stain-normalised patches + HED augmentation

For 'vahadane' and 'both' the CALLER passes data that is already normalised
(train AND evaluation arrays), so every method sees the same number of
patches, epochs, batch size and optimiser steps.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from . import augment, evaluate
from .models import make_model

METHODS = ("baseline", "hed", "vahadane", "both")


@dataclass
class Config:
    method: str = "baseline"
    seed: int = 0
    epochs: int = 5
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    sigma: float = 0.1            # HED augmentation strength (used by 'hed' and 'both')
    pretrained: bool = True
    amp: bool = True
    threshold_source: str = "id_val"   # 'id_val', 'ood_val' or 'fixed' (0.5): never the test hospital


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(x_uint8: np.ndarray, device) -> torch.Tensor:
    """(N, H, W, 3) uint8 -> (N, 3, H, W) float in [0, 1] on `device`."""
    return torch.from_numpy(x_uint8).to(device).permute(0, 3, 1, 2).float().div_(255.0)
