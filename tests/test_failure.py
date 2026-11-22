import numpy as np

from src import failure


def test_error_sets_find_false_positives_and_negatives():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.9, 0.1, 0.2, 0.8, 0.4, 0.6])
    e = failure.error_sets(y, p, 0.5)
    assert list(e["fp"]) == [0] and list(e["fn"]) == [2]


def test_slide_table_ranks_worst_slide_first():
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.9, 0.9, 0.1, 0.1, 0.9, 0.9])
    slide = np.array([7, 7, 8, 8, 8, 8])
    rows = failure.slide_error_table(y, p, slide, 0.5)
    assert rows[0]["slide"] == 7 and rows[0]["fp"] == 2 and rows[0]["error_rate"] == 1.0
    assert rows[1]["error_rate"] == 0.0


def test_fixed_fraction():
    assert failure.fixed_fraction(np.array([1, 2, 3, 4]), np.array([3, 4, 9])) == 0.5
    assert np.isnan(failure.fixed_fraction(np.array([]), np.array([1])))


def test_cam_figure_is_written(tmp_path):
    rng = np.random.default_rng(0)
    x = (rng.random((5, 96, 96, 3)) * 255).astype(np.uint8)
    idx = np.array([1, 3])
    cams = {"baseline": rng.random((2, 96, 96)), "hed": rng.random((2, 96, 96))}
    probs = {"baseline": rng.random(5), "hed": rng.random(5)}
    out = tmp_path / "f.png"
    failure.cam_figure(x, idx, cams, probs, str(out), "test")
    assert out.exists() and out.stat().st_size > 1000
