"""Vahadane-style stain normalization implemented from scratch (NumPy).

Idea (Vahadane et al. style stain separation): an H&E image is a mix of two
stains. In optical-density (OD) space, OD = -log(I / 255), the mixing is
linear:  OD ~= C @ W,  where W (2 x 3) holds the two stain colours and
C (n_pixels x 2) holds the stain concentrations, both non-negative. We find
W and C by non-negative matrix factorisation with an L1 (sparsity) penalty on
C, then re-render the image with the concentrations of the input and the
stain colours of a reference image, so only the colour changes and the tissue
structure is kept.

Notes on this implementation
  * Optimisation uses multiplicative-update NMF with an L1 term, started from
    the standard H&E stain vectors (Ruifrok-Johnston). This is a simplification
    of the dictionary-learning solver in the original paper, and the tests
    check it on images with known stain colours.
  * Background (near white) pixels are left untouched.
  * Patches with too little tissue, or a degenerate fit, fall back to the
    default stain matrix instead of failing.
  * Two modes: "patch" estimates stains per patch (noisy on 96x96 inputs),
    "slide" pools tissue pixels across all patches of a slide (more stable and
    much faster) and is the recommended default.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-8


def _unit_rows(w: np.ndarray) -> np.ndarray:
    return w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), _EPS)


# Default H&E stain OD vectors, rows = (Haematoxylin, Eosin), unit length.
DEFAULT_W = _unit_rows(np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]]))


def rgb_to_od(img: np.ndarray) -> np.ndarray:
    """uint8 RGB (..., 3) -> optical density (..., 3), floored to avoid log(0)."""
    return -np.log(np.maximum(img.astype(np.float64), 1.0) / 255.0)


def od_to_rgb(od: np.ndarray) -> np.ndarray:
    return np.clip(255.0 * np.exp(-od), 0, 255)


def tissue_mask(img: np.ndarray, lum_thresh: float = 0.8) -> np.ndarray:
    """Boolean (H, W) mask of non-background pixels (luminosity below threshold)."""
    f = img.astype(np.float64) / 255.0
    lum = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    return lum < lum_thresh


def _order_stains(w: np.ndarray) -> np.ndarray:
    """Return row order so that row 0 is Haematoxylin (larger red/green ratio)."""
    ratio = w[:, 0] / np.maximum(w[:, 1], _EPS)
    return np.argsort(-ratio)


def estimate_stain_matrix(
    od_pixels: np.ndarray, w0: np.ndarray = DEFAULT_W, lam: float = 0.02, n_iter: int = 300,
    min_cos_to_default: float = 0.85,
):
    """Sparse non-negative factorisation OD ~= C @ W.

    Minimises 0.5 * ||OD - C W||^2 + lam * sum(C) with C >= 0, W >= 0 and unit-length
    rows of W. Returns (W, ok). `ok` is False if the fit looked degenerate, in
    which case W is `w0`.
    """
    v = np.asarray(od_pixels, dtype=np.float64)
    w = w0.copy()
    c = np.maximum(v @ np.linalg.pinv(w), 0) + 1e-3
    for _ in range(n_iter):
        c *= (v @ w.T) / (c @ (w @ w.T) + lam + _EPS)
        w *= (c.T @ v) / (c.T @ c @ w + _EPS)
        norms = np.maximum(np.linalg.norm(w, axis=1, keepdims=True), _EPS)
        w /= norms
        c *= norms.T
    ok = bool(np.isfinite(w).all()) and bool(np.isfinite(c).all())
    if ok:
        ok = float(w[0] @ w[1]) < 0.99  # the two stains must stay distinct
    if ok:
        w = w[_order_stains(w)]
        # Real H&E stain colours stay within roughly 30 degrees of the standard vectors.
        # A fit that drifts further is treated as a failed fit (it usually means the
        # patch has no pure-stain pixels), and we fall back to the default stains.
        ok = bool(np.all(np.sum(w * w0, axis=1) >= min_cos_to_default))
    if not ok:
        return w0.copy(), False
    return w, True
