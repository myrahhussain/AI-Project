"""Measure training speed on the current GPU and estimate minutes per run (run first in Colab).

    python scripts/speed_test.py                       # resnet18 only
    python scripts/speed_test.py --backbones resnet18 resnet18_histo resnet50_histo
Random weights are used (no downloads); speed does not depend on the starting weights.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from src import train

p = argparse.ArgumentParser()
p.add_argument("--backbones", nargs="+", default=["resnet18"])
p.add_argument("--n_train", type=int, default=40000); p.add_argument("--epochs", type=int, default=5)
a = p.parse_args()
dev = train.get_device(); print("device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "")
x = (np.random.rand(2048, 96, 96, 3) * 255).astype(np.uint8); y = np.random.randint(0, 2, 2048)
for bb in a.backbones:
    for m in ("baseline", "hed"):
        cfg = train.Config(method=m, epochs=1, batch_size=128, pretrained=False, backbone=bb)
        train.train_model(x[:256], y[:256], cfg, dev, log=lambda *_: None)          # warm-up
        t = time.time(); train.train_model(x, y, cfg, dev, log=lambda *_: None)
        if dev.type == "cuda": torch.cuda.synchronize()
        ips = len(x) / (time.time() - t)
        print(f"{bb:15s} {m:8s} {ips:6.0f} images/s  ->  about {a.n_train * a.epochs / ips / 60:.1f} min per run")
