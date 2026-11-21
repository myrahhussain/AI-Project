"""Data utilities for Camelyon17-WILDS.

Design rules (from the project plan and instructor feedback):
  * Whole slides stay inside one split. We do NOT use the official WILDS
    `id_val` split for model selection, because its patches may share slides
    with the training patches. `check_official_id_val_overlap` measures this.
  * The test hospital is never touched unless `include_test=True`.
  * Subsets are class balanced and spread across slides, sampled with a seed.

Hospital ids follow the WILDS convention (training hospitals 0, 3, 4;
validation hospital 1; test hospital 2). Confirm these against the printout
of `dataset.split_dict` and the metadata when the real data is loaded.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

TRAIN_HOSPITALS = (0, 3, 4)
VAL_HOSPITAL = 1
TEST_HOSPITAL = 2


@dataclass
class Meta:
    """Plain numpy view of the dataset metadata."""

    hospital: np.ndarray
    slide: np.ndarray
    y: np.ndarray
    split: np.ndarray
    split_dict: dict

    @property
    def slide_key(self) -> np.ndarray:
        # Unique id for a slide even if slide numbers repeat across hospitals.
        return self.hospital.astype(np.int64) * 100000 + self.slide.astype(np.int64)

    def __len__(self) -> int:
        return len(self.y)
