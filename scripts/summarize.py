"""Collect every finished run into small tables to send back (mean ± std over seeds).

    python scripts/summarize.py --results <results> --out <summary dir>

Writes:
  summary_runs.csv    one row per tag and method (id_val / ood_val / test metrics and gaps)
  loho.csv            leave-one-hospital-out: held-out hospital x method
  summary.md          the same tables as text
"""
import argparse, glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from src import evaluate, train

_RUN = re.compile(r"^(?P<tag>.+)_(?P<method>" + "|".join(train.METHODS) + r")_seed(?P<seed>\d+)\.json$")


def tags_in(results_dir: str) -> list:
    tags = []
    for f in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        m = _RUN.match(os.path.basename(f))
        if m and m["tag"] not in tags:
            tags.append(m["tag"])
    return tags


def run_table(results_dir: str) -> pd.DataFrame:
    frames = []
    for tag in tags_in(results_dir):
        t = evaluate.summary_table(results_dir, tag)
        if len(t):
            any_run = next(iter(evaluate.load_results(results_dir, tag).values()))[0]
            cfg = any_run["config"]
            t.insert(0, "tag", tag)
            t.insert(2, "backbone", cfg.get("backbone", "resnet18"))
            t.insert(3, "sigma", cfg.get("sigma")); t.insert(4, "ref", cfg.get("ref", "pooled"))
            frames.append(t)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def loho_table(results_dir: str) -> pd.DataFrame:
    rows = []
    for tag in tags_in(results_dir):
        m = re.match(r"^loho_h(\d)$", tag)
        if not m:
            continue
        for method, rs in evaluate.load_results(results_dir, tag).items():
            row = {"held_out": int(m[1]), "method": method, "seeds": len(rs)}
            for key, path in (("test_acc", ("groups", "test", "all", "acc")), ("test_auc", ("groups", "test", "all", "auc")),
                              ("test_sens", ("groups", "test", "all", "sens")), ("test_spec", ("groups", "test", "all", "spec")),
                              ("id_val_acc", ("groups", "id_val", "all", "acc")), ("gap_acc", ("gap_test", "acc"))):
                v = []
                for r in rs:
                    x = r
                    for k in path:
                        x = x[k]
                    v.append(x)
                row[key] = f"{np.mean(v):.3f} ± {np.std(v):.3f}"
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["held_out", "method"]) if rows else pd.DataFrame()


def _md(df: pd.DataFrame) -> str:
    if not len(df):
        return "none"
    try:
        return df.to_markdown(index=False)
    except ImportError:          # 'tabulate' not installed
        return df.to_string(index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True); p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs, loho = run_table(a.results), loho_table(a.results)
    runs.to_csv(os.path.join(a.out, "summary_runs.csv"), index=False)
    loho.to_csv(os.path.join(a.out, "loho.csv"), index=False)
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write("# Runs (mean ± std over seeds)\n\n" + _md(runs) + "\n\n")
        f.write("# Leave-one-hospital-out\n\n" + _md(loho) + "\n")
    print(open(os.path.join(a.out, "summary.md")).read())


if __name__ == "__main__":
    main()
