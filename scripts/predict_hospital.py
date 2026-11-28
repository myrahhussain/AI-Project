"""Score EVERY patch of one unseen hospital with saved official-split models.

Used twice:
  * hospital 1 (validation hospital): all 34,904 patches, for the final table and metastasis size
  * hospital 2 (the sealed test hospital): the ONE-TIME final test. Run it once, after all
    settings are frozen, with every model you will report, and do not change anything afterwards.

    python scripts/predict_hospital.py --root /content/data --cache <official cache> --results <results> \
        --ckpt_dir <ckpt> --tags v3 bb_r18histo bb_r50histo --hospital 1 --out <final> --cache_all /content/cache_all
    ... --hospital 2 --unseal_test

For each saved run (tag, method, seed) with a checkpoint, the patches are given to the model the way
it was trained (raw, or stain-normalised to the same training reference), and the run's own
threshold (chosen on id_val) is applied.

Outputs in <out>:
  probs_h<H>/index.npy                    WILDS indices of the hospital's patches (row order)
  probs_h<H>/<tag>_<method>_seed<s>.npy   tumour probabilities (float16), same order
  final_h<H>.csv                          one row per run: accuracy, sensitivity, specificity, AUC
  final_h<H>_summary.csv                  mean ± std over seeds per tag and method
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src import data, evaluate, train
from scripts.run_experiments import fit_normalizer, REFS


def run_rows(results_dir, ckpt_dir, tags):
    """(name, result dict, checkpoint path) for every saved run of the given tags that has a checkpoint."""
    rows, missing = [], []
    for tag in tags:
        for method, rs in sorted(evaluate.load_results(results_dir, tag).items()):
            for r in sorted(rs, key=lambda r: r["seed"]):
                name = f"{tag}_{method}_seed{r['seed']}"
                ck = os.path.join(ckpt_dir, name + ".pt")
                (rows if os.path.exists(ck) else missing).append((name, r, ck))
    for name, _, _ in missing:
        print("no checkpoint, skipped:", name)
    return rows


def score_rows(rows, x_by_input, y, out_probs, device=None, log=print):
    """Predict with each run on its input array and apply the run's stored threshold.

    x_by_input: {'raw': patches, 'norm_<ref>': normalised patches}. Returns one metrics dict per run.
    """
    table = []
    for name, r, ck in rows:
        key = input_key(r)
        path = os.path.join(out_probs, name + ".npy")
        if os.path.exists(path):
            p = np.load(path).astype(np.float32)
        else:
            t = time.time()
            model = train.model_from_run(r["config"], ck, device)
            p = train.predict(model, x_by_input[key], next(model.parameters()).device)
            np.save(path, p.astype(np.float16))
            log(f"{name}: scored {len(p)} patches in {time.time() - t:.0f}s")
        m = evaluate.metrics(y, p, r["threshold"])
        table.append({"run": name, "tag": name.rsplit("_", 2)[0], "method": r["method"], "seed": r["seed"],
                      "backbone": r["config"].get("backbone", "resnet18"), "threshold": r["threshold"],
                      **{k: m[k] for k in ("n", "acc", "sens", "spec", "auc")},
                      "id_val_acc": r["groups"]["id_val"]["all"]["acc"]})
    return table


def input_key(r) -> str:
    if r["method"] in ("vahadane", "both"):
        return "norm_" + r["config"].get("ref", "pooled")
    return "raw"


def summarise(table):
    import pandas as pd
    df = pd.DataFrame(table)
    out = []
    for (tag, method), g in df.groupby(["tag", "method"], sort=False):
        row = {"tag": tag, "method": method, "backbone": g["backbone"].iloc[0], "seeds": len(g)}
        for k in ("acc", "sens", "spec", "auc"):
            row[k] = f"{g[k].mean():.3f} ± {g[k].std(ddof=0):.3f}"
        row["gap_acc"] = f"{(g['id_val_acc'] - g['acc']).mean():.3f}"
        out.append(row)
    return df, pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data"); p.add_argument("--cache", required=True, help="official-split cache (train patches)")
    p.add_argument("--results", required=True); p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--tags", nargs="+", required=True); p.add_argument("--hospital", type=int, required=True, choices=[1, 2])
    p.add_argument("--out", required=True); p.add_argument("--cache_all", default="/content/cache_all")
    p.add_argument("--n_train", type=int, default=40000); p.add_argument("--n_eval", type=int, default=8000)
    p.add_argument("--unseal_test", action="store_true")
    a = p.parse_args()
    if a.hospital == data.TEST_HOSPITAL and not a.unseal_test:
        sys.exit("Hospital 2 is the sealed test hospital. This is the one-time final test: freeze every\n"
                 "setting first, include every model you will report, then re-run with --unseal_test.")

    from wilds import get_dataset
    ds = get_dataset(dataset="camelyon17", download=False, root_dir=a.root)
    meta = data.load_meta(ds)
    idx = np.where(meta.hospital == a.hospital)[0]
    y, slides = meta.y[idx], meta.slide_key[idx]
    rows = run_rows(a.results, a.ckpt_dir, a.tags)
    if not rows:
        sys.exit("no runs with checkpoints found for these tags")
    out_probs = os.path.join(a.out, f"probs_h{a.hospital}")
    os.makedirs(out_probs, exist_ok=True); os.makedirs(a.cache_all, exist_ok=True)
    np.save(os.path.join(out_probs, "index.npy"), idx)

    t = time.time()
    x = data.load_patches(ds, idx, cache_path=os.path.join(a.cache_all, f"all_h{a.hospital}.npy"))
    print(f"hospital {a.hospital}: {len(idx)} patches loaded in {time.time() - t:.0f}s")
    x_by_input = {"raw": x}
    refs = sorted({input_key(r)[5:] for _, r, _ in rows if input_key(r) != "raw"})
    if refs:
        # the official split's training patches, to rebuild exactly the reference each run used
        split = data.make_official_split(meta, n_train=a.n_train, n_id_val=a.n_eval, n_ood_val=a.n_eval, seed=0)
        z = np.load(os.path.join(a.cache, "train.npy.npz"))
        if not np.array_equal(z["idx"], split["train"]):
            sys.exit("the cached training patches do not match the official split: check --cache, --n_train, --n_eval")
        x_tr, h_tr = z["x"], meta.hospital[split["train"]]
    for ref in refs:
        assert ref in REFS
        path = os.path.join(a.cache_all, f"all_h{a.hospital}_norm_{ref}.npy")
        if os.path.exists(path):
            x_by_input["norm_" + ref] = np.load(path)
            continue
        t = time.time()
        xn = fit_normalizer(x_tr, h_tr, ref).transform_batch(x, slides=slides, mode="slide", progress=True)
        np.save(path, xn); x_by_input["norm_" + ref] = xn
        print(f"normalised hospital {a.hospital} to reference {ref} in {time.time() - t:.0f}s")

    table = score_rows(rows, x_by_input, y, out_probs)
    df, summary = summarise(table)
    df.to_csv(os.path.join(a.out, f"final_h{a.hospital}.csv"), index=False)
    summary.to_csv(os.path.join(a.out, f"final_h{a.hospital}_summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
