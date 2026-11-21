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


def load_meta(dataset) -> Meta:
    """Build a Meta from a WILDS Camelyon17 dataset object."""
    arr = dataset.metadata_array.numpy()
    fields = {n: i for i, n in enumerate(dataset.metadata_fields)}
    return Meta(
        hospital=arr[:, fields["hospital"]].astype(int),
        slide=arr[:, fields["slide"]].astype(int),
        y=arr[:, fields["y"]].astype(int),
        split=np.asarray(dataset.split_array),
        split_dict=dict(dataset.split_dict),
    )


def shared_slides(meta: Meta, idx_a, idx_b) -> set:
    """Slides that appear in both index sets (should be empty for clean splits)."""
    sk = meta.slide_key
    return set(sk[np.asarray(idx_a)]) & set(sk[np.asarray(idx_b)])


def check_official_id_val_overlap(meta: Meta) -> dict:
    """How many slides does the official `id_val` split share with `train`?"""
    tr = np.where(meta.split == meta.split_dict["train"])[0]
    iv = np.where(meta.split == meta.split_dict["id_val"])[0]
    shared = shared_slides(meta, tr, iv)
    return {
        "train_slides": len(set(meta.slide_key[tr])),
        "id_val_slides": len(set(meta.slide_key[iv])),
        "shared_slides": len(shared),
    }
