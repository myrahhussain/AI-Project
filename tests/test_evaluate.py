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
