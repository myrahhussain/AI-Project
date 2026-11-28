"""Where does the model fail on the unseen hospital? (run in Colab after training with --ckpt_dir)

  python scripts/failure_analysis.py --root /content/data --cache <cache> --ckpt_dir <ckpt> \
      --results <results_ckpt> --out <out> --methods baseline hed
Uses ONLY the unseen validation hospital (1). The test hospital is never loaded.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from src import data, failure
from src.models import make_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data"); p.add_argument("--cache", required=True)
    p.add_argument("--ckpt_dir", required=True); p.add_argument("--results", required=True)
    p.add_argument("--out", default="failure_analysis"); p.add_argument("--tag", default="ckpt")
    p.add_argument("--methods", nargs="+", default=["baseline", "hed"]); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_train", type=int, default=40000); p.add_argument("--n_eval", type=int, default=8000)
    p.add_argument("--n_show", type=int, default=12)
    p.add_argument("--norm_methods", nargs="+", default=["vahadane", "both"],
                   help="methods trained on stain-normalized patches; they are evaluated on ood_val_norm")
    a = p.parse_args()
    from wilds import get_dataset
    meta = data.load_meta(get_dataset(dataset="camelyon17", download=False, root_dir=a.root))
    split = data.make_official_split(meta, n_train=a.n_train, n_id_val=a.n_eval, n_ood_val=a.n_eval, seed=0, include_test=False)
    idx = split["ood_val"]
    x = data.load_cached_patches(a.cache, "ood_val")
    assert len(x) == len(idx), f"cache has {len(x)} patches but the split has {len(idx)}: use the same --n_train/--n_eval as the run"
    inputs = {m: data.load_cached_patches(a.cache, "ood_val_norm") for m in a.methods if m in a.norm_methods}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, thr = {}, {}
    for m in a.methods:
        net = make_model(pretrained=False)
        net.load_state_dict(torch.load(os.path.join(a.ckpt_dir, f"{a.tag}_{m}_seed{a.seed}.pt"), map_location=dev))
        models[m] = net.to(dev)
        thr[m] = json.load(open(os.path.join(a.results, f"{a.tag}_{m}_seed{a.seed}.json")))["threshold"]
    summary = failure.analyze(x, meta.y[idx], meta.slide_key[idx], models, thr, a.out, a.n_show, dev, inputs)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
