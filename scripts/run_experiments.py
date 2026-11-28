"""Train and evaluate methods x seeds on Camelyon17-WILDS (run in Colab with a GPU).

Official split (default)
  train on hospitals 0, 3, 4; id_val = held-out slides of those hospitals; ood_val = hospital 1.
  Hospital 2 is never loaded here. The one-time final test is scripts/predict_hospital.py.

    python scripts/run_experiments.py --root /content/data --cache <cache> --out <results> \
        --seeds 0 1 2 --tag v3 --ckpt_dir <ckpt>

Ablations (official split, hospital 1)
    --sigma 0.05 --methods hed both --tag abl_sigma0.05        HED strength
    --ref h0 --methods vahadane --tag abl_ref_h0               normalisation reference from one lab

Backbones (official split, hospital 1)
    --backbone resnet18_histo --weights_path tenpercent_resnet18.ckpt --tag bb_r18histo --ckpt_dir <ckpt>
    --backbone resnet50_histo --weights_path mocov2_rn50_ep200.torch --tag bb_r50histo --ckpt_dir <ckpt>

Leave-one-hospital-out (uses hospital 2, so it needs --unseal_test)
    --split loho --test_hospital 3 --unseal_test --tag loho_h3 --cache /content/cache_loho
  Train on the other four hospitals; the threshold comes from their held-out slides; the
  held-out hospital is reported as 'test'.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src import data, normalize, train

REFS = ("pooled", "h0", "h1", "h2", "h3", "h4")


def load_split(dataset, meta, split, cache, prefix=""):
    """{name: (patches, labels, hospital, slide_key)}; raw patches cached as <cache>/<prefix><name>.npy.npz."""
    out = {}
    for name, idx in split.items():
        path = os.path.join(cache, f"{prefix}{name}.npy") if cache else None
        x = data.load_patches(dataset, idx, cache_path=path)
        out[name] = (x, meta.y[idx].astype(np.int64), meta.hospital[idx], meta.slide_key[idx])
    return out


def reference_index(x_train: np.ndarray, h_train: np.ndarray, ref: str = "pooled") -> int:
    """Index (into x_train) of the normalisation reference patch. Always from TRAINING data.

    'pooled': the most typical patch over all training hospitals (the interim setting).
    'h<k>'  : the most typical training patch from hospital k only ("normalise to lab k's colours").
    """
    if ref == "pooled":
        return int(normalize.choose_reference(x_train))
    if ref not in REFS:
        raise ValueError(f"unknown reference {ref}; choose from {REFS}")
    sub = np.where(np.asarray(h_train) == int(ref[1:]))[0]
    if len(sub) == 0:
        raise ValueError(f"reference {ref}: that hospital is not in the training data")
    return int(sub[normalize.choose_reference(x_train[sub])])


def fit_normalizer(x_train, h_train, ref="pooled") -> normalize.StainNormalizer:
    norm = normalize.StainNormalizer()
    norm.fit(x_train[reference_index(x_train, h_train, ref)])
    return norm


def norm_cache_name(prefix: str, name: str, ref: str) -> str:
    # 'pooled' keeps the interim file names so existing caches are reused
    return f"{prefix}{name}_norm.npy" if ref == "pooled" else f"{prefix}{name}_norm_{ref}.npy"


def normalise_all(raw, cache, prefix="", ref="pooled"):
    """Stain-normalise every split to one reference patch chosen from the TRAINING hospitals."""
    norm = None
    out = {}
    for name, (x, y, h, s) in raw.items():
        path = os.path.join(cache, norm_cache_name(prefix, name, ref)) if cache else None
        if path and os.path.exists(path):
            xn = np.load(path)
        else:
            if norm is None:
                norm = fit_normalizer(raw["train"][0], raw["train"][2], ref)
            t = time.time(); xn = norm.transform_batch(x, slides=s, mode="slide", progress=True)
            print(f"normalised {prefix}{name} (ref {ref}) in {time.time() - t:.0f}s")
            if path: np.save(path, xn)
        out[name] = (xn, y, h, s)
    return out


def build_split(meta, a):
    """Returns (split, cache prefix)."""
    if a.split == "official":
        split = data.make_official_split(meta, n_train=a.n_train, n_id_val=a.n_eval, n_ood_val=a.n_eval,
                                         seed=0, include_test=False)
        print(data.check_split(meta, split, allow_test=False))
        return split, ""
    if not a.unseal_test:
        sys.exit("Leave-one-hospital-out uses hospital 2 (as training data or as the held-out hospital).\n"
                 "Run the one-time final test first, then pass --unseal_test.")
    if a.test_hospital is None:
        sys.exit("--split loho needs --test_hospital")
    split = data.make_loho_split(meta, test_hospital=a.test_hospital, n_train=a.n_train, n_id_val=a.n_eval,
                                 seed=0, include_test=True, n_test=a.n_eval)
    print(data.check_split(meta, split, test_hospitals=(a.test_hospital,), allow_test=True))
    return split, f"loho{a.test_hospital}_"


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data"); p.add_argument("--cache", default=None); p.add_argument("--out", default="results")
    p.add_argument("--n_train", type=int, default=40000); p.add_argument("--n_eval", type=int, default=8000)
    p.add_argument("--epochs", type=int, default=5); p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--sigma", type=float, default=0.2); p.add_argument("--hed_mode", default="relative", choices=["relative", "absolute"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--methods", nargs="+", default=list(train.METHODS)); p.add_argument("--tag", default="interim")
    p.add_argument("--ckpt_dir", default=None, help="save model weights here (needed for the final test and failure analysis)")
    p.add_argument("--backbone", default="resnet18", choices=list(train.BACKBONES))
    p.add_argument("--weights_path", default=None, help="weight file for resnet18_histo / resnet50_histo")
    p.add_argument("--ref", default="pooled", choices=list(REFS), help="normalisation reference (ablation)")
    p.add_argument("--split", default="official", choices=["official", "loho"])
    p.add_argument("--test_hospital", type=int, default=None, help="held-out hospital for --split loho")
    p.add_argument("--unseal_test", action="store_true", help="allow hospital 2 data (only after the final test)")
    p.add_argument("--no_pretrained", action="store_true", help="random start weights (offline smoke tests only)")
    return p.parse_args(argv)


def main():
    a = parse_args()
    from wilds import get_dataset
    ds = get_dataset(dataset="camelyon17", download=False, root_dir=a.root)
    meta = data.load_meta(ds)
    split, prefix = build_split(meta, a)
    if a.cache: os.makedirs(a.cache, exist_ok=True)
    raw = load_split(ds, meta, split, a.cache, prefix)
    normed = None
    for method in a.methods:
        if method in ("vahadane", "both") and normed is None:
            normed = normalise_all(raw, a.cache, prefix, a.ref)
        src = normed if method in ("vahadane", "both") else raw
        d = {k: v[:3] for k, v in src.items()}
        for seed in a.seeds:
            f = os.path.join(a.out, f"{a.tag}_{method}_seed{seed}.json")
            if os.path.exists(f): print("skip (done)", f); continue
            cfg = train.Config(method=method, seed=seed, epochs=a.epochs, batch_size=a.batch_size, sigma=a.sigma,
                               hed_mode=a.hed_mode, backbone=a.backbone, weights_path=a.weights_path,
                               ref=a.ref, split=a.split, test_hospital=a.test_hospital,
                               pretrained=not a.no_pretrained)
            t = time.time()
            train.run_method(d, cfg, out_dir=a.out, tag=a.tag, ckpt_dir=a.ckpt_dir)
            print(f"{a.tag} {method} seed {seed}: {(time.time() - t) / 60:.1f} min")
    from src import evaluate
    print(evaluate.summary_table(a.out, a.tag).to_string())


if __name__ == "__main__":
    main()
