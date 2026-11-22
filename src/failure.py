"""Failure analysis helpers: which patches are wrong, where do errors concentrate, what do CAMs show."""
from __future__ import annotations

import numpy as np


def error_sets(y, prob, thr: float) -> dict:
    """Indices of false positives (healthy called tumour) and false negatives (missed tumour)."""
    y, prob = np.asarray(y).astype(int), np.asarray(prob)
    pred = (prob >= thr).astype(int)
    return {"fp": np.where((pred == 1) & (y == 0))[0], "fn": np.where((pred == 0) & (y == 1))[0]}


def slide_error_table(y, prob, slide, thr: float) -> list[dict]:
    """Per-slide error counts, worst first. Shows whether errors concentrate in a few slides."""
    y, prob, slide = np.asarray(y).astype(int), np.asarray(prob), np.asarray(slide)
    pred = (prob >= thr).astype(int)
    rows = []
    for s in np.unique(slide):
        m = slide == s
        rows.append({"slide": int(s), "n": int(m.sum()),
                     "fp": int(((pred == 1) & (y == 0) & m).sum()),
                     "fn": int(((pred == 0) & (y == 1) & m).sum()),
                     "error_rate": float((pred[m] != y[m]).mean())})
    return sorted(rows, key=lambda r: -r["error_rate"])


def fixed_fraction(err_a: np.ndarray, err_b: np.ndarray) -> float:
    """Fraction of method A's errors that method B does not make."""
    a, b = set(map(int, err_a)), set(map(int, err_b))
    return float(len(a - b) / len(a)) if a else float("nan")


def cam_figure(x_uint8, idx, cams: dict, probs: dict, out: str, title: str = ""):
    """Rows = patches, columns = original + one CAM overlay per method (with its predicted probability)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .cam import overlay

    names = list(cams)
    fig, ax = plt.subplots(len(idx), 1 + len(names), figsize=(2 * (1 + len(names)), 2 * len(idx)), squeeze=False)
    for r, i in enumerate(idx):
        ax[r, 0].imshow(x_uint8[i]); ax[r, 0].axis("off")
        for c, n in enumerate(names, start=1):
            ax[r, c].imshow(overlay(x_uint8[i], cams[n][r])); ax[r, c].axis("off")
            ax[r, c].set_title(f"{n} p={probs[n][i]:.2f}", fontsize=8)
    ax[0, 0].set_title("original", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
