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


def test_analyze_end_to_end_on_tiny_models(tmp_path):
    import torch
    from src.models import make_model
    rng = np.random.default_rng(0)
    x = (rng.random((40, 96, 96, 3)) * 255).astype(np.uint8)
    y = (rng.random(40) < 0.5).astype(int)
    slide = rng.integers(0, 4, 40)
    torch.manual_seed(0)
    models = {"baseline": make_model(False), "hed": make_model(False)}
    s = failure.analyze(x, y, slide, models, {"baseline": 0.5, "hed": 0.5}, str(tmp_path), n_show=3)
    assert set(s["counts"]) == {"baseline", "hed"}
    assert len(s["worst_slides"]) <= 4 and "hed_fixes_baseline_fp" in s
    assert any(f.name.endswith("_cams.png") for f in tmp_path.iterdir())


def test_analyze_scores_each_method_on_its_own_input(tmp_path, monkeypatch):
    import torch
    from src import failure, cam
    monkeypatch.setattr(cam, "fused_layercam", lambda model, t: torch.zeros(len(t), t.shape[2], t.shape[3]))

    class Mean(torch.nn.Module):          # logit = mean pixel brightness, so the input decides the output
        def forward(self, x):
            return (x.mean(dim=(1, 2, 3)) - 0.5) * 20
    x_raw = np.zeros((6, 8, 8, 3), np.uint8)             # dark -> low probability
    x_norm = np.full((6, 8, 8, 3), 255, np.uint8)        # bright -> high probability
    y = np.ones(6, int); slide = np.zeros(6, int)
    m = {"a": Mean(), "b": Mean()}
    s = failure.analyze(x_raw, y, slide, m, {"a": 0.5, "b": 0.5}, str(tmp_path), n_show=2, inputs={"b": x_norm})
    assert s["counts"]["a"]["fn"] == 6 and s["counts"]["b"]["fn"] == 0
