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
