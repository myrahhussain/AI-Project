"""Figure: original patches vs HED-augmented at several strengths. Works on real or synthetic patches."""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import augment


def make_grid(x_uint8, sigmas=(0.0, 0.05, 0.1, 0.2), out="results/aug_grid.png"):
    x = torch.from_numpy(x_uint8).permute(0, 3, 1, 2).float() / 255
    fig, ax = plt.subplots(len(x), len(sigmas), figsize=(2 * len(sigmas), 2 * len(x)))
    ax = np.atleast_2d(ax)
    for j, s in enumerate(sigmas):
        g = torch.Generator().manual_seed(0)
        y = augment.hed_augment(x, s, g)
        for i in range(len(x)):
            ax[i, j].imshow(y[i].permute(1, 2, 0).clamp(0, 1).numpy()); ax[i, j].axis("off")
        ax[0, j].set_title("original" if s == 0 else f"sigma={s}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--npy", help="uint8 (N,96,96,3) array"); p.add_argument("--out", default="results/aug_grid.png")
    a = p.parse_args()
    if a.npy: x = np.load(a.npy)[:4]
    else:
        rng = np.random.default_rng(0); x = (rng.random((4, 96, 96, 3)) * 60 + 150).astype(np.uint8)
        print("No --npy given: using random SYNTHETIC patches (layout check only).")
    print(make_grid(x, out=a.out))
