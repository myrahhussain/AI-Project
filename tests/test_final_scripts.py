"""Tests for the final-experiment scripts (reference choice, guards, scoring, summaries)."""
import json

import numpy as np
import pytest
import torch

from src import models, normalize, train
from scripts import run_experiments as rx
from scripts import predict_hospital as ph
from scripts import summarize as sm
from test_splits import synthetic_meta


def he_like(n, seed, tint=0):
    rng = np.random.default_rng(seed)
    x = np.full((n, 96, 96, 3), 235, np.uint8)
    x[:, 20:76, 20:76] = np.array([150 + tint, 80, 160], np.uint8)
    noise = rng.integers(-10, 10, x.shape)
    return np.clip(x.astype(int) + noise, 0, 255).astype(np.uint8)


def test_reference_is_always_a_training_patch_from_the_requested_hospital():
    x = np.concatenate([he_like(20, 0), he_like(20, 1, tint=40)])
    h = np.array([0] * 20 + [3] * 20)
    assert rx.reference_index(x, h, "pooled") == normalize.choose_reference(x)
    assert h[rx.reference_index(x, h, "h3")] == 3
    assert h[rx.reference_index(x, h, "h0")] == 0
    with pytest.raises(ValueError):
        rx.reference_index(x, h, "h4")          # not a training hospital here


def test_pooled_reference_keeps_the_interim_cache_names():
    assert rx.norm_cache_name("", "train", "pooled") == "train_norm.npy"
    assert rx.norm_cache_name("loho3_", "test", "h0") == "loho3_test_norm_h0.npy"


def test_defaults_match_the_interim_protocol():
    a = rx.parse_args([])
    assert (a.sigma, a.ref, a.backbone, a.split, a.epochs, a.batch_size) == (0.2, "pooled", "resnet18", "official", 5, 128)


def test_loho_refuses_to_run_before_the_test_hospital_is_unsealed():
    meta = synthetic_meta()
    a = rx.parse_args(["--split", "loho", "--test_hospital", "3", "--n_train", "1000", "--n_eval", "200"])
    with pytest.raises(SystemExit):
        rx.build_split(meta, a)
    a.unseal_test = True
    split, prefix = rx.build_split(meta, a)
    assert prefix == "loho3_" and set(split) == {"train", "id_val", "test"}
    assert set(meta.hospital[split["test"]]) == {3}
    assert 3 not in set(meta.hospital[split["train"]]) and 2 in set(meta.hospital[split["train"]])


def test_official_split_never_contains_hospital_2():
    meta = synthetic_meta()
    split, prefix = rx.build_split(meta, rx.parse_args(["--n_train", "1000", "--n_eval", "200"]))
    assert prefix == "" and "test" not in split
    for idx in split.values():
        assert 2 not in set(meta.hospital[idx])


def _fake_run(tmp_path, tag, method, seed, backbone="resnet18", ref="pooled"):
    m = models.make_model(pretrained=False, backbone=backbone)
    torch.save(m.state_dict(), tmp_path / "ckpt" / f"{tag}_{method}_seed{seed}.pt")
    cfg = train.Config(method=method, seed=seed, backbone=backbone, ref=ref, pretrained=False)
    from dataclasses import asdict
    r = {"method": method, "seed": seed, "config": asdict(cfg), "threshold": 0.5,
         "groups": {"id_val": {"all": {"acc": 0.9, "sens": 0.9, "spec": 0.9, "auc": 0.95}},
                    "test": {"all": {"acc": 0.8, "sens": 0.7, "spec": 0.9, "auc": 0.9}}},
         "gap": {"acc": 0.1, "sens": 0.2, "spec": 0.0, "auc": 0.05},
         "gap_test": {"acc": 0.1, "sens": 0.2, "spec": 0.0, "auc": 0.05}}
    r["groups"]["ood_val"] = r["groups"]["test"]
    (tmp_path / "res" / f"{tag}_{method}_seed{seed}.json").write_text(json.dumps(r))


def test_scoring_uses_each_runs_own_input_and_threshold(tmp_path):
    (tmp_path / "ckpt").mkdir(); (tmp_path / "res").mkdir(); (tmp_path / "probs").mkdir()
    _fake_run(tmp_path, "v3", "baseline", 0)
    _fake_run(tmp_path, "v3", "vahadane", 0)
    _fake_run(tmp_path, "bb_r18histo", "vahadane", 1, backbone="resnet18_histo", ref="h3")
    (tmp_path / "res" / "v3_hed_seed0.json").write_text(
        (tmp_path / "res" / "v3_baseline_seed0.json").read_text().replace('"baseline"', '"hed"'))   # no checkpoint
    rows = ph.run_rows(str(tmp_path / "res"), str(tmp_path / "ckpt"), ["v3", "bb_r18histo"])
    assert [n for n, _, _ in rows] == ["v3_baseline_seed0", "v3_vahadane_seed0", "bb_r18histo_vahadane_seed1"]
    assert [ph.input_key(r) for _, r, _ in rows] == ["raw", "norm_pooled", "norm_h3"]
    x = he_like(6, 0)
    y = np.array([0, 1, 0, 1, 0, 1])
    table = ph.score_rows(rows, {"raw": x, "norm_pooled": x, "norm_h3": x}, y, str(tmp_path / "probs"),
                          device=torch.device("cpu"), log=lambda *_: None)
    assert len(table) == 3 and all(0 <= t["acc"] <= 1 for t in table)
    assert table[2]["backbone"] == "resnet18_histo" and table[2]["tag"] == "bb_r18histo"
    assert (tmp_path / "probs" / "v3_baseline_seed0.npy").exists()
    df, summary = ph.summarise(table)
    assert set(summary["tag"]) == {"v3", "bb_r18histo"}


def test_summaries_find_tags_and_build_the_loho_table(tmp_path):
    (tmp_path / "ckpt").mkdir(); (tmp_path / "res").mkdir()
    for tag in ("v3", "abl_sigma0.05", "loho_h3"):
        for seed in (0, 1):
            _fake_run(tmp_path, tag, "baseline", seed)
    assert sm.tags_in(str(tmp_path / "res")) == ["abl_sigma0.05", "loho_h3", "v3"]
    runs = sm.run_table(str(tmp_path / "res"))
    assert set(runs["tag"]) == {"v3", "abl_sigma0.05", "loho_h3"} and set(runs["seeds"]) == {2}
    loho = sm.loho_table(str(tmp_path / "res"))
    assert loho.iloc[0]["held_out"] == 3 and loho.iloc[0]["test_acc"].startswith("0.800")
