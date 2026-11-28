"""Metastasis size per slide (ITC / micro / macro) from patch labels and from model predictions.

    python scripts/metastasis_size.py --root /content/data --stage_labels stage_labels.csv \
        --final <predict_hospital out> --hospitals 1 2 --out <size out>

Step 1 (no model): measure each of the 50 slides from the pathologists' tumour patches and compare
  with CAMELYON17's official slide labels. This checks the measuring method itself.
Step 2 (models): measure the slides of each unseen hospital again, this time from each model's
  tumour calls (saved by predict_hospital.py), and compare with the official label and with the
  step-1 size. Over-calling tumour inflates sizes (over-staging); under-calling shrinks them.

Outputs in <out>: size_gt_check.csv, size_slides.csv, size_summary.csv
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from src import size


def wilds_metadata(dataset) -> pd.DataFrame:
    """WILDS Camelyon17 metadata rows (row i = patch i) with patient, node, x_coord, y_coord, tumor, center."""
    df = dataset._metadata_df.reset_index(drop=True)
    need = {"patient", "node", "x_coord", "y_coord", "tumor", "center"}
    if not need <= set(df.columns):
        raise KeyError(f"metadata columns are {list(df.columns)}; expected {sorted(need)}")
    return df


def gt_check(meta_df: pd.DataFrame, official: dict) -> pd.DataFrame:
    t = size.per_slide(meta_df, meta_df["tumor"].to_numpy() == 1, official=official)
    hosp = meta_df.groupby(["patient", "node"])["center"].first().reset_index()
    hosp["slide"] = [size.slide_name(p, n) for p, n in zip(hosp["patient"], hosp["node"])]
    return t.merge(hosp[["slide", "center"]].rename(columns={"center": "hospital"}), on="slide")


def model_sizes(meta_df, gt, final_dir, hospital):
    """Per-run, per-slide sizes from saved probabilities of one hospital."""
    pdir = os.path.join(final_dir, f"probs_h{hospital}")
    idx = np.load(os.path.join(pdir, "index.npy"))
    runs = pd.read_csv(os.path.join(final_dir, f"final_h{hospital}.csv"))
    sub = meta_df.iloc[idx]
    slides, summary = [], []
    for _, r in runs.iterrows():
        p = np.load(os.path.join(pdir, r["run"] + ".npy")).astype(np.float32)
        t = size.per_slide(sub, p >= r["threshold"])
        t.insert(0, "run", r["run"]); t["tag"], t["method"], t["seed"], t["hospital"] = r["tag"], r["method"], r["seed"], hospital
        slides.append(t)
        summary.append({"run": r["run"], "tag": r["tag"], "method": r["method"], "seed": r["seed"], "hospital": hospital,
                        **size.compare(t, gt)})
    return pd.concat(slides), pd.DataFrame(summary)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data"); p.add_argument("--stage_labels", required=True)
    p.add_argument("--final", default=None, help="output folder of predict_hospital.py")
    p.add_argument("--hospitals", type=int, nargs="*", default=[1]); p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from wilds import get_dataset
    ds = get_dataset(dataset="camelyon17", download=False, root_dir=a.root)
    meta_df = wilds_metadata(ds)
    official = size.load_stage_labels(a.stage_labels)

    gt = gt_check(meta_df, official)
    gt.to_csv(os.path.join(a.out, "size_gt_check.csv"), index=False)
    known = gt[gt["official"].isin(size.CATEGORIES)]
    print(f"Step 1: {len(gt)} slides, {len(known)} with an official label")
    print(pd.crosstab(known["official"], known["category"], rownames=["official"], colnames=["from patches"]))
    print(f"agreement: {(known['official'] == known['category']).mean():.2f}")

    if a.final:
        all_slides, all_summary = [], []
        for h in a.hospitals:
            if not os.path.exists(os.path.join(a.final, f"final_h{h}.csv")):
                print(f"hospital {h}: no predictions yet, skipped"); continue
            s, m = model_sizes(meta_df, gt, a.final, h)
            all_slides.append(s); all_summary.append(m)
        if all_summary:
            pd.concat(all_slides).to_csv(os.path.join(a.out, "size_slides.csv"), index=False)
            summ = pd.concat(all_summary)
            summ.to_csv(os.path.join(a.out, "size_summary.csv"), index=False)
            cols = ["category_agreement", "over_staged", "under_staged", "mae_mm", "mean_error_mm"]
            print(summ.groupby(["hospital", "tag", "method"])[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
