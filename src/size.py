"""Metastasis size from patch predictions: from "is there tumour?" to "how much tumour?".

CAMELYON17 labels every slide by the size of its largest metastasis (pathologists' criteria):
  macro-metastasis       > 2.0 mm
  micro-metastasis       > 0.2 mm and <= 2.0 mm
  isolated tumour cells  <= 0.2 mm (ITC)
These slide labels decide the patient's pN stage, so size is the clinically meaningful
"strength" of a lymph-node metastasis.

Camelyon17-WILDS tiled each slide with 96x96 patches every 32 pixels at 1/4 resolution, so the
central 32x32 squares (the region each label describes) form a grid that covers the slide, and
tumour patches were kept almost in full. Each grid cell is ~32 px x 4 x 0.24 um ≈ 31 um wide.
Treating every cell called "tumour" as tumour tissue, the largest 8-connected group of tumour
cells approximates the largest metastasis, and its longest extent approximates its diameter.

Simplifications (state them when reporting): size only (the clinical ITC rule also counts cells),
a 31 um grid (one cell of error at each end), and normal patches were subsampled by WILDS, so a
model's false alarms can only appear where a normal patch was kept.
"""
from __future__ import annotations

from functools import reduce
from math import gcd

import numpy as np

CELL_UM = 32 * 4 * 0.243          # grid step in micrometres (level-0 pixel size ~0.23-0.25 um)
CATEGORIES = ("negative", "itc", "micro", "macro")   # in increasing order of size
MICRO_MM, MACRO_MM = 0.2, 2.0


def grid_step(coords) -> int:
    """Common spacing of patch coordinates (works whatever units WILDS stored them in)."""
    c = np.unique(np.asarray(coords, dtype=np.int64))
    d = np.diff(c)
    d = d[d > 0]
    return int(reduce(gcd, d.tolist())) if len(d) else 1


def to_grid(x, y):
    """Integer grid positions (gx, gy) of patches from their coordinates."""
    x, y = np.asarray(x, dtype=np.int64), np.asarray(y, dtype=np.int64)
    step = gcd(grid_step(x), grid_step(y)) or 1
    return (x - x.min()) // step, (y - y.min()) // step


def _diameter_cells(pts: np.ndarray) -> float:
    """Longest distance between cell centres of a group, in cells (plus one cell for the cell widths)."""
    if len(pts) == 1:
        return 1.0
    cand = pts
    if len(pts) > 3:
        try:
            from scipy.spatial import ConvexHull
            cand = pts[ConvexHull(pts).vertices]
        except Exception:            # all points on one line: its two ends are among these extremes
            keys = (pts[:, 0], pts[:, 1], pts.sum(1), pts[:, 0] - pts[:, 1])
            cand = pts[np.unique([f(k) for k in keys for f in (np.argmin, np.argmax)])]
    diff = cand[:, None, :] - cand[None, :, :]
    return float(np.sqrt((diff ** 2).sum(-1)).max()) + 1.0


def lesion_stats(gx, gy, positive, cell_um: float = CELL_UM) -> dict:
    """Largest metastasis among the positive cells of ONE slide.

    Returns largest_mm (longest extent of the largest 8-connected group), largest_cells,
    total_mm2 (area of all positive cells), n_groups and category.
    """
    from scipy import ndimage

    gx, gy, positive = np.asarray(gx), np.asarray(gy), np.asarray(positive, dtype=bool)
    if not positive.any():
        return {"largest_mm": 0.0, "largest_cells": 0, "total_mm2": 0.0, "n_groups": 0, "category": "negative"}
    px, py = gx[positive], gy[positive]
    x0, y0 = px.min(), py.min()
    grid = np.zeros((py.max() - y0 + 1, px.max() - x0 + 1), dtype=bool)
    grid[py - y0, px - x0] = True
    labels, n = ndimage.label(grid, structure=np.ones((3, 3), dtype=int))
    sizes = np.bincount(labels.ravel())[1:]
    biggest = int(np.argmax(sizes)) + 1
    ys, xs = np.nonzero(labels == biggest)
    largest_mm = _diameter_cells(np.stack([xs, ys], 1).astype(float)) * cell_um / 1000.0
    return {
        "largest_mm": largest_mm,
        "largest_cells": int(sizes.max()),
        "total_mm2": float(positive.sum()) * (cell_um / 1000.0) ** 2,
        "n_groups": int(n),
        "category": size_category(largest_mm),
    }


def size_category(largest_mm: float) -> str:
    if largest_mm <= 0:
        return "negative"
    if largest_mm <= MICRO_MM:
        return "itc"
    if largest_mm <= MACRO_MM:
        return "micro"
    return "macro"


def rank(category: str) -> int:
    return CATEGORIES.index(category)


def slide_name(patient, node) -> str:
    """CAMELYON17 file name of a slide, e.g. patient 4 node 4 -> 'patient_004_node_4.tif'."""
    return f"patient_{int(patient):03d}_node_{int(node)}.tif"


def load_stage_labels(path: str) -> dict:
    """{'patient_004_node_4.tif': 'macro', ...} from CAMELYON17's stage_labels.csv (patient rows skipped)."""
    import pandas as pd

    df = pd.read_csv(path)
    name_col = [c for c in df.columns if c.lower() in ("patient", "name", "file")][0]
    stage_col = [c for c in df.columns if c.lower() in ("stage", "label")][0]
    out = {}
    for n, s in zip(df[name_col].astype(str), df[stage_col].astype(str)):
        s = s.strip().lower()
        if n.endswith(".tif") and s in CATEGORIES:
            out[n.strip()] = s
    return out


def per_slide(meta_df, positive, cell_um: float = CELL_UM, official: dict | None = None):
    """One row per slide: size statistics of `positive` cells (+ the official label if given).

    meta_df needs columns patient, node, x_coord, y_coord (WILDS metadata rows, same order as positive).
    """
    import pandas as pd

    meta_df = meta_df.reset_index(drop=True)
    positive = np.asarray(positive, dtype=bool)
    rows = []
    for (patient, node), g in meta_df.groupby(["patient", "node"], sort=True):
        i = g.index.to_numpy()
        gx, gy = to_grid(g["x_coord"].to_numpy(), g["y_coord"].to_numpy())
        st = lesion_stats(gx, gy, positive[i], cell_um)
        name = slide_name(patient, node)
        row = {"slide": name, "n_patches": len(i), **st}
        if official is not None:
            row["official"] = official.get(name, "unknown")
        rows.append(row)
    return pd.DataFrame(rows)


def compare(pred: "pd.DataFrame", truth: "pd.DataFrame") -> dict:
    """Agreement of predicted sizes with the official labels and with ground-truth-derived sizes."""
    m = pred.merge(truth[["slide", "largest_mm", "official"]], on="slide", suffixes=("", "_gt"))
    known = m[m["official"].isin(CATEGORIES)]
    pr = known["category"].map(rank).to_numpy()
    tr = known["official"].map(rank).to_numpy()
    return {
        "slides": int(len(m)),
        "category_agreement": float((pr == tr).mean()) if len(known) else float("nan"),
        "over_staged": int((pr > tr).sum()),
        "under_staged": int((pr < tr).sum()),
        "mae_mm": float(np.abs(m["largest_mm"] - m["largest_mm_gt"]).mean()),
        "mean_error_mm": float((m["largest_mm"] - m["largest_mm_gt"]).mean()),
    }
