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
