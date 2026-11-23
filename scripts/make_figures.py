"""Make the interim-demo figures from the cached patches and saved results.

  python scripts/make_figures.py --cache <cache dir> --results <results dir> --out figures
Needs (from the experiment run): cached train, ood_val and their *_norm arrays
and <results>/interim_*_seed*.json
"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import augment, data

COLORS = {"baseline": "#6b7280", "hed": "#0f766e", "vahadane": "#b45309", "both": "#7c3aed"}


def pick(n_total, n, seed=0):
    return np.sort(np.random.default_rng(seed).choice(n_total, n, replace=False))


def aug_grid(x, out, sigmas=(0.0, 0.05, 0.1, 0.2)):
    t = torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255
    fig, ax = plt.subplots(len(x), len(sigmas), figsize=(2 * len(sigmas), 2 * len(x)))
    for j, s in enumerate(sigmas):
        y = augment.hed_augment(t, s, torch.Generator().manual_seed(0))
        for i in range(len(x)):
            ax[i, j].imshow(y[i].permute(1, 2, 0).clamp(0, 1).numpy()); ax[i, j].axis("off")
        ax[0, j].set_title("original" if s == 0 else f"HED sigma={s}", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def norm_grid(raw, norm, titles, out):
    fig, ax = plt.subplots(2, len(raw), figsize=(2 * len(raw), 4.4))
    for i in range(len(raw)):
        ax[0, i].imshow(raw[i]); ax[1, i].imshow(norm[i])
        ax[0, i].axis("off"); ax[1, i].axis("off")
        ax[0, i].set_title(titles[i], fontsize=8)
    ax[0, 0].text(-0.08, 0.5, "original", transform=ax[0, 0].transAxes, rotation=90, va="center", ha="right")
    ax[1, 0].text(-0.08, 0.5, "normalized", transform=ax[1, 0].transAxes, rotation=90, va="center", ha="right")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def results_chart(results_dir, out, tag="interim", zoom=False):
    runs = {}
    for f in sorted(glob.glob(os.path.join(results_dir, f"{tag}_*.json"))):
        r = json.load(open(f)); runs.setdefault(r["method"], []).append(r)
    if not runs:
        print("no result files found; skipping results chart"); return
    order = [m for m in ("baseline", "hed", "vahadane", "both") if m in runs]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, key, title in zip(axes, ("acc", "auc"), ("Accuracy on unseen hospital", "AUC on unseen hospital")):
        vals = [[r["groups"]["ood_val"]["all"][key] for r in runs[m]] for m in order]
        mean = [np.mean(v) for v in vals]; sd = [np.std(v) for v in vals]
        ax.bar(order, mean, yerr=sd, color=[COLORS[m] for m in order], capsize=4)
        for i, m in enumerate(mean): ax.text(i, m + sd[i] + 0.012, f"{m:.3f}", ha="center", fontsize=9)
        # honest default: the axis starts at 0 so small gaps look small; --zoom magnifies them
        ax.set_ylim((min(mean) - 0.06, min(1.0, max(mean) + 0.04)) if zoom else (0, 1.08)); ax.set_title(title, fontsize=11)
        if not zoom: ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Mean over seeds, error bars = std (" + ("zoomed y-axis)" if zoom else "y-axis starts at 0)"), fontsize=9, y=0.05)
    fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(out, dpi=150); plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True); p.add_argument("--results", required=True)
    p.add_argument("--out", default="figures"); p.add_argument("--tag", default="interim"); p.add_argument("--n", type=int, default=6)
    p.add_argument("--zoom", action="store_true", help="zoom the y-axis of the results chart")
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
    load = lambda name: data.load_cached_patches(a.cache, name)
    tr = load("train"); idx = pick(len(tr), a.n)
    aug_grid(np.stack([tr[i] for i in idx]), os.path.join(a.out, "hed_augmentation.png"))
    for split, title in (("train", "training hospitals"), ("ood_val", "unseen hospital 1")):
        try:
            raw, nrm = load(split), load(f"{split}_norm")
        except FileNotFoundError as e:
            print("skipping normalization figure:", e); continue
        if True:
            ii = pick(len(raw), a.n, seed=1)
            norm_grid([raw[i] for i in ii], [nrm[i] for i in ii], [title] * a.n,
                      os.path.join(a.out, f"normalization_{split}.png"))
    results_chart(a.results, os.path.join(a.out, "results_unseen_hospital.png"), a.tag, zoom=a.zoom)
    print("saved:", sorted(os.listdir(a.out)))
