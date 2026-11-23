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


def solve_concentrations(od_pixels: np.ndarray, w: np.ndarray, n_iter: int = 30) -> np.ndarray:
    """Non-negative stain concentrations for fixed W (multiplicative NNLS)."""
    v = np.asarray(od_pixels, dtype=np.float64)
    c = np.maximum(v @ np.linalg.pinv(w), 0) + 1e-3
    ww = w @ w.T
    vw = v @ w.T
    for _ in range(n_iter):
        c *= vw / (c @ ww + _EPS)
    return c


class StainNormalizer:
    """Fit on one reference patch, then transform patches to its stain colours."""

    def __init__(
        self,
        lam: float = 0.02,
        lum_thresh: float = 0.8,
        min_tissue_px: int = 200,
        max_px: int = 20000,
        n_iter: int = 300,
        percentile: float = 95.0,
        seed: int = 0,
        od_floor: float = 20.0,
    ):
        self.lam, self.lum_thresh, self.min_tissue_px = lam, lum_thresh, min_tissue_px
        self.max_px, self.n_iter, self.percentile, self.seed = max_px, n_iter, percentile, seed
        # estimation ignores pixels darker than this, so saturated near-black
        # pixels cannot inflate the slide's stain-strength estimate
        self.od_floor = od_floor
        self.w_ref: np.ndarray | None = None
        self.maxc_ref: np.ndarray | None = None

    # ---- estimation -------------------------------------------------
    def _tissue_od(self, imgs) -> np.ndarray:
        parts = []
        for img in imgs:
            m = tissue_mask(img, self.lum_thresh)
            if m.any():
                parts.append(-np.log(np.maximum(img[m].astype(np.float64), self.od_floor) / 255.0))
        return np.concatenate(parts) if parts else np.zeros((0, 3))

    def estimate(self, imgs):
        """Estimate (W, maxC, ok) from the pooled tissue pixels of one or more images."""
        od = self._tissue_od(imgs)
        if len(od) < self.min_tissue_px:
            return DEFAULT_W.copy(), None, False
        if len(od) > self.max_px:
            rng = np.random.default_rng(self.seed)
            od = od[rng.choice(len(od), self.max_px, replace=False)]
        w, ok = estimate_stain_matrix(od, lam=self.lam, n_iter=self.n_iter)
        c = solve_concentrations(od, w)
        maxc = np.percentile(c, self.percentile, axis=0)
        if not np.all(maxc > 1e-4):
            return DEFAULT_W.copy(), None, False
        return w, maxc, ok

    def fit(self, reference_img: np.ndarray) -> "StainNormalizer":
        w, maxc, ok = self.estimate([reference_img])
        if maxc is None:
            raise ValueError("reference image has too little tissue to fit stain colours")
        self.w_ref, self.maxc_ref = w, maxc
        return self

    # ---- transformation ---------------------------------------------
    def transform(self, img: np.ndarray, w: np.ndarray | None = None, maxc: np.ndarray | None = None) -> np.ndarray:
        """Normalise one uint8 patch. If (w, maxc) are given (e.g. per slide) they are used
        instead of per-patch estimates."""
        assert self.w_ref is not None, "call fit() first"
        mask = tissue_mask(img, self.lum_thresh)
        if w is None or maxc is None:
            w, maxc, _ = self.estimate([img])
        if maxc is None or mask.sum() < 1:
            return img.copy()  # not enough tissue to estimate: leave unchanged
        od = rgb_to_od(img[mask])
        c = solve_concentrations(od, w)
        c = c * (self.maxc_ref / np.maximum(maxc, _EPS))
        out = img.copy()
        out[mask] = np.rint(od_to_rgb(c @ self.w_ref)).astype(np.uint8)
        return out

    def transform_batch(self, x: np.ndarray, slides: np.ndarray | None = None, mode: str = "slide",
                        patches_per_slide: int = 200, progress: bool = False) -> np.ndarray:
        """Normalise an array (N, H, W, 3) of uint8 patches.

        mode="patch": stains estimated for every patch separately.
        mode="slide": stains estimated once per slide from pooled patches (needs `slides`).
        """
        out = np.empty_like(x)
        iterator = range(len(x))
        if mode == "patch" or slides is None:
            if progress:
                from tqdm import tqdm
                iterator = tqdm(iterator)
            for i in iterator:
                out[i] = self.transform(x[i])
            return out
        rng = np.random.default_rng(self.seed)
        slides = np.asarray(slides)
        groups = np.unique(slides)
        if progress:
            from tqdm import tqdm
            groups = tqdm(groups)
        for s in groups:
            idx = np.where(slides == s)[0]
            pick = idx if len(idx) <= patches_per_slide else rng.choice(idx, patches_per_slide, replace=False)
            w, maxc, _ = self.estimate([x[i] for i in pick])
            for i in idx:
                out[i] = self.transform(x[i], w=w, maxc=maxc)
        return out


def choose_reference(x: np.ndarray, n_candidates: int = 50, seed: int = 0, lum_thresh: float = 0.8) -> int:
    """Index of a *typical* reference patch from the training hospitals.

    Candidates need plenty of tissue and a trustworthy stain estimate. Among them we pick the one
    closest to the median stain colours and median stain intensities, so the reference is neither an
    unusually pale patch nor an extreme one. (An earlier rule that preferred patches with balanced H and E
    picked pale, eosin-heavy patches, which washed out the nuclei in every normalised image.)

    Pass ONLY training-hospital patches. The choice uses no labels and never sees the test hospital.
    """
    rng = np.random.default_rng(seed)
    cand = rng.choice(len(x), min(n_candidates, len(x)), replace=False)
    norm = StainNormalizer()
    found = []
    for i in cand:
        if tissue_mask(x[i], lum_thresh).mean() < 0.6:
            continue
        w, maxc, ok = norm.estimate([x[i]])
        if maxc is None or not ok:
            continue
        found.append((int(i), np.asarray(w, dtype=float), np.log(np.maximum(np.asarray(maxc, dtype=float), 1e-6))))
    if not found:
        return int(cand[0])
    med_w = np.median([f[1] for f in found], axis=0)
    med_c = np.median([f[2] for f in found], axis=0)
    dist = [np.linalg.norm(w - med_w) + np.abs(c - med_c).sum() for _, w, c in found]
    return found[int(np.argmin(dist))][0]
