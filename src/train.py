"""One training / evaluation routine shared by every method (matched budgets).

Methods
  baseline : flips + 90-degree rotations only
  hed      : baseline + HED stain augmentation (strength sigma)
  vahadane : baseline, on stain-normalised patches
  both     : stain-normalised patches + HED augmentation

For 'vahadane' and 'both' the CALLER passes data that is already normalised
(train AND evaluation arrays), so every method sees the same number of
patches, epochs, batch size and optimiser steps.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from . import augment, evaluate
from .models import make_model

METHODS = ("baseline", "hed", "vahadane", "both")


@dataclass
class Config:
    method: str = "baseline"
    seed: int = 0
    epochs: int = 5
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    sigma: float = 0.1            # HED augmentation strength (used by 'hed' and 'both')
    pretrained: bool = True
    amp: bool = True
    threshold_source: str = "id_val"   # 'id_val', 'ood_val' or 'fixed' (0.5): never the test hospital


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(x_uint8: np.ndarray, device) -> torch.Tensor:
    """(N, H, W, 3) uint8 -> (N, 3, H, W) float in [0, 1] on `device`."""
    return torch.from_numpy(x_uint8).to(device).permute(0, 3, 1, 2).float().div_(255.0)


@torch.no_grad()
def predict(model, x: np.ndarray, device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        out.append(torch.sigmoid(model(to_tensor(x[i : i + batch_size], device))).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def train_model(x: np.ndarray, y: np.ndarray, cfg: Config, device, log=print):
    """Train one model. Returns (model, history)."""
    if cfg.method not in METHODS:
        raise ValueError(f"unknown method {cfg.method}")
    use_hed = cfg.method in ("hed", "both")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    model = make_model(cfg.pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = len(x) // cfg.batch_size
    total_steps = steps_per_epoch * cfg.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1))
    amp = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_fn = nn.BCEWithLogitsLoss()
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        perm = rng.permutation(len(x))
        losses = []
        t0 = time.time()
        for s in range(steps_per_epoch):
            idx = np.sort(perm[s * cfg.batch_size : (s + 1) * cfg.batch_size])
            xb = to_tensor(x[idx], device)
            yb = torch.from_numpy(y[idx]).float().to(device)
            xb = augment.flip_rot90(xb, gen)
            if use_hed:
                xb = augment.hed_augment(xb, cfg.sigma, gen)   # float32, outside autocast
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = loss_fn(model(xb).float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "seconds": time.time() - t0})
        log(f"[{cfg.method} seed {cfg.seed}] epoch {epoch + 1}/{cfg.epochs} "
            f"loss {history[-1]['loss']:.4f} ({history[-1]['seconds']:.0f}s)")
    return model, history


def run_method(data: dict, cfg: Config, out_dir: str | None = None, tag: str = "run",
               ckpt_dir: str | None = None, log=print) -> dict:
    """Train on data['train'] and evaluate on every other split present.

    data: {'train': (x, y, hospital), 'id_val': (...), 'ood_val': (...), optional 'test': (...)}
    The threshold is chosen on validation data only. 'test' is evaluated only if present,
    which the caller should do exactly once, at the very end.
    """
    device = get_device()
    x_tr, y_tr, _ = data["train"]
    model, history = train_model(x_tr, y_tr, cfg, device, log)
    probs = {k: predict(model, v[0], device) for k, v in data.items() if k != "train"}

    if cfg.threshold_source == "fixed" or cfg.threshold_source not in probs:
        thr = 0.5
    else:
        thr = evaluate.best_threshold(data[cfg.threshold_source][1], probs[cfg.threshold_source])

    groups, groups_default = {}, {}
    for k, p in probs.items():
        _, yk, hk = data[k]
        groups[k] = evaluate.evaluate_groups(yk, p, hk, thr)
        groups_default[k] = evaluate.evaluate_groups(yk, p, hk, 0.5)
    result = {
        "method": cfg.method,
        "seed": cfg.seed,
        "config": asdict(cfg),
        "threshold": thr,
        "budget": {"n_train": int(len(x_tr)), "epochs": cfg.epochs, "batch_size": cfg.batch_size,
                   "steps": int((len(x_tr) // cfg.batch_size) * cfg.epochs)},
        "groups": groups,
        "groups_at_0.5": groups_default,
        "history": history,
    }
    if "id_val" in groups and "ood_val" in groups:
        result["gap"] = evaluate.gap(groups["id_val"]["all"], groups["ood_val"]["all"])
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{tag}_{cfg.method}_seed{cfg.seed}.json"), "w") as f:
            json.dump(result, f, indent=2)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{tag}_{cfg.method}_seed{cfg.seed}.pt"))
    return result
