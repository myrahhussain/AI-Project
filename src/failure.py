"""Failure analysis helpers: which patches are wrong, where do errors concentrate, what do CAMs show."""
from __future__ import annotations

import numpy as np


def error_sets(y, prob, thr: float) -> dict:
    """Indices of false positives (healthy called tumour) and false negatives (missed tumour)."""
    y, prob = np.asarray(y).astype(int), np.asarray(prob)
    pred = (prob >= thr).astype(int)
    return {"fp": np.where((pred == 1) & (y == 0))[0], "fn": np.where((pred == 0) & (y == 1))[0]}


def slide_error_table(y, prob, slide, thr: float) -> list[dict]:
    """Per-slide error counts, worst first. Shows whether errors concentrate in a few slides."""
    y, prob, slide = np.asarray(y).astype(int), np.asarray(prob), np.asarray(slide)
    pred = (prob >= thr).astype(int)
    rows = []
    for s in np.unique(slide):
        m = slide == s
        rows.append({"slide": int(s), "n": int(m.sum()),
                     "fp": int(((pred == 1) & (y == 0) & m).sum()),
                     "fn": int(((pred == 0) & (y == 1) & m).sum()),
                     "error_rate": float((pred[m] != y[m]).mean())})
    return sorted(rows, key=lambda r: -r["error_rate"])


def fixed_fraction(err_a: np.ndarray, err_b: np.ndarray) -> float:
    """Fraction of method A's errors that method B does not make."""
    a, b = set(map(int, err_a)), set(map(int, err_b))
    return float(len(a - b) / len(a)) if a else float("nan")


def cam_figure(x_uint8, idx, cams: dict, probs: dict, out: str, title: str = ""):
    """Rows = patches, columns = original + one CAM overlay per method (with its predicted probability)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .cam import overlay

    names = list(cams)
    fig, ax = plt.subplots(len(idx), 1 + len(names), figsize=(2 * (1 + len(names)), 2 * len(idx)), squeeze=False)
    for r, i in enumerate(idx):
        ax[r, 0].imshow(x_uint8[i]); ax[r, 0].axis("off")
        for c, n in enumerate(names, start=1):
            ax[r, c].imshow(overlay(x_uint8[i], cams[n][r])); ax[r, c].axis("off")
            ax[r, c].set_title(f"{n} p={probs[n][i]:.2f}", fontsize=8)
    ax[0, 0].set_title("original", fontsize=8)
    if title:
        import textwrap
        # wrap to the figure width (about 11 characters per inch at this font size) so it is never clipped
        title = textwrap.fill(title, max(20, int(fig.get_figwidth() * 11)))
        # reserve room for the wrapped lines (about 0.22 inch each) above the column headers
        top = 1 - (0.3 + 0.22 * title.count("\n")) / fig.get_figheight()
        fig.tight_layout(rect=(0, 0, 1, top))
        fig.suptitle(title, fontsize=10, y=0.999)
    else:
        fig.tight_layout()
    fig.savefig(out, dpi=150); plt.close(fig)


def analyze(x_uint8, y, slide, models: dict, thresholds: dict, out_dir: str, n_show: int = 12, device="cpu", inputs: dict | None = None):
    """Compare methods on one evaluation set (e.g. the unseen hospital).

    models / thresholds: {method: model} / {method: decision threshold chosen on validation data}.
    Writes cam figures for false positives and false negatives of the FIRST method (the baseline)
    and returns a summary dict. `inputs` optionally maps a method to the patches it was trained on
    (e.g. stain-normalized ones); methods not listed get x_uint8. CAMs are LayerCAM from the finer layers: a way to investigate
    failures, not proof of what the model 'reasons'.
    """
    import os
    import torch
    from . import cam as camlib
    from .train import predict

    os.makedirs(out_dir, exist_ok=True)
    names = list(models)
    inputs = inputs or {}
    xin = {n: inputs.get(n, x_uint8) for n in names}
    probs = {n: predict(models[n], xin[n], device) for n in names}
    errs = {n: error_sets(y, probs[n], thresholds[n]) for n in names}
    base = names[0]
    summary = {"counts": {n: {"fp": int(len(e["fp"])), "fn": int(len(e["fn"]))} for n, e in errs.items()},
               "worst_slides": slide_error_table(y, probs[base], slide, thresholds[base])[:5]}
    for n in names[1:]:
        summary[f"{n}_fixes_baseline_fp"] = fixed_fraction(errs[base]["fp"], errs[n]["fp"])
        summary[f"{n}_fixes_baseline_fn"] = fixed_fraction(errs[base]["fn"], errs[n]["fn"])
    for kind in ("fp", "fn"):
        idx = errs[base][kind]
        if len(idx) == 0:
            continue
        # most confident mistakes first
        order = np.argsort(-probs[base][idx] if kind == "fp" else probs[base][idx])
        idx = idx[order][:n_show]
        ts = {n: torch.from_numpy(xin[n][idx]).permute(0, 3, 1, 2).float().div(255).to(device) for n in names}
        cams = {n: camlib.fused_layercam(models[n].to(device), ts[n]).cpu().numpy() for n in names}
        label = "false positives (healthy called tumour)" if kind == "fp" else "false negatives (missed tumour)"
        cam_figure(x_uint8, idx, cams, probs, os.path.join(out_dir, f"{base}_{kind}_cams.png"),
                   f"{label}: baseline errors, same patches under each method")
    return summary
