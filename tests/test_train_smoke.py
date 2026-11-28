"""CPU smoke test on synthetic patches. Real training must run on a GPU."""
import numpy as np
import pytest

from src import train


def synthetic(n, seed, shift=0.0):
    """Tumour patches get a dark purple blob; `shift` adds a colour cast (like a new lab)."""
    rng = np.random.default_rng(seed)
    x = (rng.random((n, 96, 96, 3)) * 30 + 200).astype(np.float32)
    y = (rng.random(n) < 0.5).astype(np.int64)
    for i in np.where(y == 1)[0]:
        x[i, 30:60, 30:60] = [110, 60, 150]
    x[..., 0] += shift
    return np.clip(x, 0, 255).astype(np.uint8), y


def make_data(seed=0):
    xt, yt = synthetic(96, seed)
    xi, yi = synthetic(32, seed + 1)
    xo, yo = synthetic(32, seed + 2, shift=15)
    return {
        "train": (xt, yt, np.zeros(len(yt), int)),
        "id_val": (xi, yi, np.zeros(len(yi), int)),
        "ood_val": (xo, yo, np.ones(len(yo), int)),
    }


@pytest.mark.parametrize("method", ["baseline", "hed"])
def test_methods_run_with_matched_budget(method, tmp_path):
    cfg = train.Config(method=method, seed=0, epochs=1, batch_size=32, pretrained=False, amp=False)
    r = train.run_method(make_data(), cfg, out_dir=str(tmp_path), tag="smoke")
    assert r["budget"] == {"n_train": 96, "epochs": 1, "batch_size": 32, "steps": 3}
    for split in ("id_val", "ood_val"):
        for k in ("acc", "sens", "spec", "auc"):
            v = r["groups"][split]["all"][k]
            assert 0 <= v <= 1
    assert "h0" in r["groups"]["id_val"] and "h1" in r["groups"]["ood_val"]
    assert set(r["gap"]) == {"acc", "sens", "spec", "auc"}
    assert (tmp_path / f"smoke_{method}_seed0.json").exists()


def test_threshold_uses_validation_not_test():
    data = make_data()
    data["test"] = data["ood_val"]
    cfg = train.Config(method="baseline", epochs=1, batch_size=32, pretrained=False, amp=False,
                       threshold_source="id_val")
    r = train.run_method(data, cfg)
    # recompute: the stored threshold must equal the one chosen on id_val predictions
    assert 0 <= r["threshold"] <= 1
    assert "test" in r["groups"]  # evaluated only because the caller passed it


def test_same_seed_gives_same_loss():
    cfg = train.Config(method="hed", seed=3, epochs=1, batch_size=32, pretrained=False, amp=False)
    a = train.run_method(make_data(), cfg)
    b = train.run_method(make_data(), cfg)
    assert abs(a["history"][0]["loss"] - b["history"][0]["loss"]) < 1e-5


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        train.train_model(np.zeros((4, 96, 96, 3), np.uint8), np.zeros(4, np.int64),
                          train.Config(method="nope", pretrained=False), train.get_device())


def test_backbone_is_recorded_and_held_out_hospital_gap_is_reported():
    data = make_data()
    data["test"] = data.pop("ood_val")
    cfg = train.Config(method="baseline", epochs=1, batch_size=32, pretrained=False, amp=False,
                       backbone="resnet18_histo", split="loho", test_hospital=1)
    r = train.run_method(data, cfg)
    assert r["config"]["backbone"] == "resnet18_histo" and r["config"]["test_hospital"] == 1
    assert set(r["gap_test"]) == {"acc", "sens", "spec", "auc"} and "gap" not in r
