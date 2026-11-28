"""Evaluation: per-hospital accuracy, sensitivity, specificity, AUC and generalisation gaps."""
from __future__ import annotations

import glob
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

METRICS = ("acc", "sens", "spec", "auc")


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximising Youden's J = sensitivity + specificity - 1.

    Call this on VALIDATION data only (never on the test hospital).
    """
    y, p = np.asarray(y), np.asarray(p)
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    k = int(np.argmax(j))
    t = float(thr[k])
    return 0.5 if not np.isfinite(t) else t


def metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    out = {
        "n": int(len(y)),
        "acc": (tp + tn) / max(len(y), 1),
        "sens": tp / (tp + fn) if (tp + fn) else float("nan"),
        "spec": tn / (tn + fp) if (tn + fp) else float("nan"),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
    }
    return out


def evaluate_groups(y, p, hospital, thr: float) -> dict:
    """Metrics overall ('all') and for each hospital ('h<id>')."""
    y, p, hospital = np.asarray(y), np.asarray(p), np.asarray(hospital)
    out = {"all": metrics(y, p, thr)}
    for h in np.unique(hospital):
        m = hospital == h
        out[f"h{int(h)}"] = metrics(y[m], p[m], thr)
    return out


def gap(id_metrics: dict, ood_metrics: dict) -> dict:
    """In-distribution minus out-of-distribution for each metric (positive = worse on unseen)."""
    return {k: id_metrics[k] - ood_metrics[k] for k in METRICS}


def aggregate(results: list[dict]) -> dict:
    """Mean and std over seeds of the overall ID / OOD metrics and the gaps."""
    rows = {}
    for split in ("id_val", "ood_val", "test"):
        if all(split in r["groups"] for r in results):
            for k in METRICS:
                v = np.array([r["groups"][split]["all"][k] for r in results], dtype=float)
                rows[f"{split}_{k}"] = (float(np.nanmean(v)), float(np.nanstd(v)))
    for key in ("gap", "gap_test"):
        if results and all(key in r for r in results):
            for k in METRICS:
                v = np.array([r[key][k] for r in results], dtype=float)
                rows[f"{key}_{k}"] = (float(np.nanmean(v)), float(np.nanstd(v)))
    return rows


def load_results(results_dir: str, tag: str | None = None) -> dict:
    """Group saved per-run JSON files by method."""
    by_method: dict[str, list[dict]] = {}
    pattern = f"{tag}_*.json" if tag else "*.json"
    for f in sorted(glob.glob(os.path.join(results_dir, pattern))):
        with open(f) as fh:
            r = json.load(fh)
        if "method" in r and "groups" in r:
            by_method.setdefault(r["method"], []).append(r)
    return by_method


def summary_table(results_dir: str, tag: str | None = None):
    """pandas DataFrame: one row per method, 'mean ± std' over seeds."""
    import pandas as pd

    rows = []
    for method, rs in load_results(results_dir, tag).items():
        agg = aggregate(rs)
        row = {"method": method, "seeds": len(rs)}
        for k, (m, s) in agg.items():
            row[k] = f"{m:.3f} ± {s:.3f}"
        rows.append(row)
    return pd.DataFrame(rows)
