import json

import numpy as np

from src import evaluate


def test_metrics_on_known_predictions():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1, 0.2, 0.7, 0.3, 0.4])
    m = evaluate.metrics(y, p, 0.5)
    assert m["n"] == 8
    assert abs(m["sens"] - 2 / 3) < 1e-9          # 2 of 3 tumours found
    assert abs(m["spec"] - 4 / 5) < 1e-9          # 4 of 5 healthy patches correct
    assert abs(m["acc"] - 6 / 8) < 1e-9
    assert 0.5 < m["auc"] <= 1


def test_single_class_group_gives_nan_not_crash():
    m = evaluate.metrics(np.zeros(5), np.linspace(0, 1, 5), 0.5)
    assert np.isnan(m["sens"]) and np.isnan(m["auc"])


def test_best_threshold_separates_shifted_scores():
    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(500), np.ones(500)])
    p = np.concatenate([rng.normal(0.2, 0.05, 500), rng.normal(0.4, 0.05, 500)]).clip(0, 1)
    thr = evaluate.best_threshold(y, p)
    assert 0.25 < thr < 0.35
    assert evaluate.metrics(y, p, thr)["acc"] > evaluate.metrics(y, p, 0.5)["acc"]


def test_per_hospital_groups_and_gap():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.6, 0.4, 0.7, 0.3])
    h = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    g = evaluate.evaluate_groups(y, p, h, 0.5)
    assert set(g) == {"all", "h0", "h1"}
    assert g["h0"]["acc"] == 1.0 and g["h1"]["acc"] == 0.0
    gp = evaluate.gap(g["h0"], g["h1"])
    assert gp["acc"] == 1.0
