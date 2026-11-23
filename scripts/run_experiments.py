"""Run all four methods x seeds on the Camelyon17 subset (run in Colab with a GPU).

  python scripts/run_experiments.py --root /content/data --cache /content/drive/MyDrive/ai_project/cache \
      --out results --n_train 40000 --epochs 5 --seeds 0 1 2
Test hospital (2) is NEVER loaded here.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src import data, normalize, train


def load_split(dataset, meta, split, cache):
    out = {}
    for name, idx in split.items():
        x = data.load_patches(dataset, idx, cache_path=os.path.join(cache, f"{name}.npy") if cache else None)
        out[name] = (x, meta.y[idx].astype(np.int64), meta.hospital[idx], meta.slide_key[idx])
    return out


def normalise_all(raw, cache):
    """Stain-normalise every split to one reference patch from the TRAINING hospitals."""
    norm = normalize.StainNormalizer()
    ref = normalize.choose_reference(raw["train"][0])
    norm.fit(raw["train"][0][ref])
    out = {}
    for name, (x, y, h, s) in raw.items():
        path = os.path.join(cache, f"{name}_norm.npy") if cache else None
        if path and os.path.exists(path):
            xn = np.load(path)
        else:
            t = time.time(); xn = norm.transform_batch(x, slides=s, mode="slide", progress=True)
            print(f"normalised {name} in {time.time() - t:.0f}s")
            if path: np.save(path, xn)
        out[name] = (xn, y, h, s)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data"); p.add_argument("--cache", default=None); p.add_argument("--out", default="results")
    p.add_argument("--n_train", type=int, default=40000); p.add_argument("--n_eval", type=int, default=8000)
    p.add_argument("--epochs", type=int, default=5); p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--sigma", type=float, default=0.2); p.add_argument("--hed_mode", default="relative", choices=["relative", "absolute"]); p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--methods", nargs="+", default=list(train.METHODS)); p.add_argument("--tag", default="interim")
    p.add_argument("--ckpt_dir", default=None, help="save model weights here (needed for failure analysis)")
    a = p.parse_args()
    from wilds import get_dataset
    ds = get_dataset(dataset="camelyon17", download=False, root_dir=a.root)
    meta = data.load_meta(ds)
    split = data.make_official_split(meta, n_train=a.n_train, n_id_val=a.n_eval, n_ood_val=a.n_eval, seed=0, include_test=False)
    print(data.check_split(meta, split, allow_test=False))
    if a.cache: os.makedirs(a.cache, exist_ok=True)
    raw = load_split(ds, meta, split, a.cache)
    normed = None
    for method in a.methods:
        if method in ("vahadane", "both") and normed is None:
            normed = normalise_all(raw, a.cache)
        src = normed if method in ("vahadane", "both") else raw
        d = {k: v[:3] for k, v in src.items()}
        for seed in a.seeds:
            f = os.path.join(a.out, f"{a.tag}_{method}_seed{seed}.json")
            if os.path.exists(f): print("skip (done)", f); continue
            cfg = train.Config(method=method, seed=seed, epochs=a.epochs, batch_size=a.batch_size, sigma=a.sigma, hed_mode=a.hed_mode)
            train.run_method(d, cfg, out_dir=a.out, tag=a.tag, ckpt_dir=a.ckpt_dir)
    from src import evaluate
    print(evaluate.summary_table(a.out, a.tag).to_string())


if __name__ == "__main__":
    main()
