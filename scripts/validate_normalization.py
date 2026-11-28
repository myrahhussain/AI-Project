"""Does our stain estimate agree with an open-source implementation (torchstain's Macenko)?

  pip install torchstain
  python scripts/validate_normalization.py --root /content/data --cache <cache> --split ood_val

For each slide we pool up to N patches, estimate the two stain colours with our normalizer and with
torchstain's Macenko, and report the cosine similarity between matching stains (1.0 = same colour).
The two methods are different algorithms (Vahadane-style vs Macenko), so we expect similar, not
identical, colours. Also printed: similarity of each estimate to the fixed textbook H&E colours, as
context for what a 'high' number means. Uses only train / unseen-validation hospitals, never test.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src import data, normalize


def macenko_matrix(pixels_rgb: np.ndarray) -> np.ndarray:
    """torchstain Macenko stain colours (rows = 2 stains, unit length) from a stack of RGB pixels."""
    import torchstain
    n = torchstain.normalizers.MacenkoNormalizer(backend="numpy")
    he, _, _ = n._NumpyMacenkoNormalizer__compute_matrices(pixels_rgb.reshape(-1, 3).astype(np.float32), 240, 0.15, 0.15)
    he = np.asarray(he, dtype=np.float64).T          # rows = stains
    return he / np.linalg.norm(he, axis=1, keepdims=True)


def match_cosines(a: np.ndarray, b: np.ndarray) -> list:
    """Cosine similarity of the two stains of a and b, pairing them in the order that matches best."""
    s = a @ b.T
    straight, swapped = s[0, 0] + s[1, 1], s[0, 1] + s[1, 0]
    return [float(s[0, 0]), float(s[1, 1])] if straight >= swapped else [float(s[0, 1]), float(s[1, 0])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data"); p.add_argument("--cache", required=True)
    p.add_argument("--split", default="ood_val", choices=["train", "id_val", "ood_val"])
    p.add_argument("--n_train", type=int, default=40000); p.add_argument("--n_eval", type=int, default=8000)
    p.add_argument("--per_slide", type=int, default=100); p.add_argument("--out", default=None)
    a = p.parse_args()
    try:
        import torchstain  # noqa: F401
    except ImportError:
        print("torchstain missing: run  pip install torchstain  first"); return 1
    from wilds import get_dataset
    meta = data.load_meta(get_dataset(dataset="camelyon17", download=False, root_dir=a.root))
    split = data.make_official_split(meta, n_train=a.n_train, n_id_val=a.n_eval, n_ood_val=a.n_eval, seed=0, include_test=False)
    idx = split[a.split]; x = data.load_cached_patches(a.cache, a.split); slides = meta.slide_key[idx]
    assert len(x) == len(idx)
    ours_n = normalize.StainNormalizer(); rng = np.random.default_rng(0); rows = []
    for s in np.unique(slides):
        ii = np.where(slides == s)[0]; ii = rng.choice(ii, min(len(ii), a.per_slide), replace=False)
        patches = x[ii]
        w_ours = ours_n.estimate(patches)[0]
        tissue = patches[normalize.tissue_mask(patches)]
        if len(tissue) < 500:
            continue
        w_mac = macenko_matrix(tissue)
        rows.append({"slide": int(s), "ours_vs_macenko": match_cosines(w_ours, w_mac),
                     "ours_vs_default": match_cosines(w_ours, normalize.DEFAULT_W),
                     "macenko_vs_default": match_cosines(w_mac, normalize.DEFAULT_W)})
    for k in ("ours_vs_macenko", "ours_vs_default", "macenko_vs_default"):
        v = np.array([r[k] for r in rows])
        print(f"{k:20s} cosine  H: {v[:, 0].mean():.3f} (min {v[:, 0].min():.3f})   E: {v[:, 1].mean():.3f} (min {v[:, 1].min():.3f})")
    print(f"slides compared: {len(rows)}")
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
