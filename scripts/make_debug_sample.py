"""Save a small sample of real patches (and their normalised versions) for debugging slide-level normalization.

  python scripts/make_debug_sample.py --root /content/data --cache <cache dir> --out debug_sample.npz
Takes up to --per_slide patches from every slide in the training hospitals and the unseen hospital. Uses only
train / id_val / ood_val patches; the test hospital is never loaded.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src import data


def sample_per_slide(slide_keys, per_slide: int, seed: int = 0) -> np.ndarray:
    """Positions (sorted) of up to `per_slide` random patches from each slide."""
    rng = np.random.default_rng(seed)
    slide_keys = np.asarray(slide_keys)
    picked = []
    for s in np.unique(slide_keys):
        pos = np.where(slide_keys == s)[0]
        picked.append(rng.choice(pos, min(per_slide, len(pos)), replace=False))
    return np.sort(np.concatenate(picked))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data"); p.add_argument("--cache", required=True)
    p.add_argument("--out", default="debug_sample.npz"); p.add_argument("--per_slide", type=int, default=20)
    p.add_argument("--n_train", type=int, default=40000); p.add_argument("--n_eval", type=int, default=8000)
    a = p.parse_args()
    from wilds import get_dataset
    meta = data.load_meta(get_dataset(dataset="camelyon17", download=False, root_dir=a.root))
    split = data.make_official_split(meta, n_train=a.n_train, n_id_val=a.n_eval, n_ood_val=a.n_eval, seed=0, include_test=False)
    out = {}
    for name in ("train", "id_val", "ood_val"):
        idx = split[name]
        raw = data.load_cached_patches(a.cache, name)
        assert len(raw) == len(idx), f"{name}: cache has {len(raw)} patches, split has {len(idx)}"
        pos = sample_per_slide(meta.slide_key[idx], a.per_slide)
        out[f"{name}_x"] = raw[pos]
        out[f"{name}_slide"] = meta.slide_key[idx][pos]
        out[f"{name}_hospital"] = meta.hospital[idx][pos]
        out[f"{name}_y"] = meta.y[idx][pos]
        try:
            out[f"{name}_norm"] = data.load_cached_patches(a.cache, f"{name}_norm")[pos]
        except FileNotFoundError:
            print("no normalised cache for", name)
    np.savez_compressed(a.out, **out)
    print("wrote", a.out, f"({os.path.getsize(a.out) / 1e6:.1f} MB)", {k: v.shape for k, v in out.items() if k.endswith('_x')})


if __name__ == "__main__":
    main()
