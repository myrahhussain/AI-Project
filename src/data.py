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


def balanced_subset(meta: Meta, candidates, n: int, rng: np.random.Generator) -> np.ndarray:
    """Class-balanced subset of `candidates`, spread evenly across slides.

    Takes up to n/2 patches per class, drawing round-robin from the slides so
    that no single slide dominates. Returns fewer than n if a class runs out.
    """
    candidates = np.asarray(candidates)
    sk = meta.slide_key
    per_class = n // 2
    chosen = []
    for c in (0, 1):
        cc = candidates[meta.y[candidates] == c]
        groups = {}
        for s in np.unique(sk[cc]):
            g = cc[sk[cc] == s].copy()
            rng.shuffle(g)
            groups[s] = g
        order = list(groups.keys())
        rng.shuffle(order)
        ptr = {s: 0 for s in order}
        picked = []
        while len(picked) < per_class and any(ptr[s] < len(groups[s]) for s in order):
            for s in order:
                if ptr[s] < len(groups[s]):
                    picked.append(groups[s][ptr[s]])
                    ptr[s] += 1
                    if len(picked) >= per_class:
                        break
        chosen.append(np.array(picked, dtype=int))
    out = np.concatenate(chosen)
    rng.shuffle(out)
    return out


def _holdout_slides(meta: Meta, pool, frac: float, rng: np.random.Generator) -> set:
    """Pick whole slides to hold out, at least one per hospital in the pool."""
    pool = np.asarray(pool)
    held = set()
    for h in np.unique(meta.hospital[pool]):
        slides = np.unique(meta.slide_key[pool[meta.hospital[pool] == h]])
        n_hold = max(1, int(round(frac * len(slides))))
        n_hold = min(n_hold, max(len(slides) - 1, 0))
        if n_hold == 0:
            continue
        held.update(rng.choice(slides, size=n_hold, replace=False).tolist())
    return held


def _make_split(meta, pool, ood_pool, n_train, n_id_val, n_ood_val, holdout_frac, rng, test_pool=None, n_test=None):
    pool = np.asarray(pool)
    held = _holdout_slides(meta, pool, holdout_frac, rng)
    is_held = np.isin(meta.slide_key[pool], list(held))
    train_pool, id_pool = pool[~is_held], pool[is_held]
    out = {
        "train": balanced_subset(meta, train_pool, n_train, rng),
        "id_val": balanced_subset(meta, id_pool, n_id_val, rng),
    }
    if ood_pool is not None and len(ood_pool) > 0:
        out["ood_val"] = balanced_subset(meta, ood_pool, n_ood_val, rng)
    if test_pool is not None:
        out["test"] = balanced_subset(meta, test_pool, n_test or n_ood_val, rng)
    return out


def make_official_split(
    meta: Meta,
    n_train: int = 40000,
    n_id_val: int = 8000,
    n_ood_val: int = 8000,
    seed: int = 0,
    holdout_frac: float = 0.2,
    include_test: bool = False,
    n_test: int | None = None,
) -> dict:
    """Official WILDS hospitals with our own slide-level held-out validation.

    train   : slides from hospitals 0,3,4 (official train split), minus held-out slides
    id_val  : held-out slides from the same hospitals ("seen hospital" check)
    ood_val : the official validation hospital (1), used to choose settings
    test    : the official test hospital (2), only returned if include_test=True
    """
    rng = np.random.default_rng(seed)
    pool = np.where(
        np.isin(meta.hospital, TRAIN_HOSPITALS) & (meta.split == meta.split_dict["train"])
    )[0]
    ood_pool = np.where(meta.hospital == VAL_HOSPITAL)[0]
    test_pool = np.where(meta.hospital == TEST_HOSPITAL)[0] if include_test else None
    split = _make_split(meta, pool, ood_pool, n_train, n_id_val, n_ood_val, holdout_frac, rng, test_pool, n_test)
    check_split(meta, split, test_hospitals=(TEST_HOSPITAL,), allow_test=include_test)
    return split
