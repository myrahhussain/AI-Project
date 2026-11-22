"""Measure training images/second on the current GPU (run first in Colab)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from src import train
dev = train.get_device(); print("device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "")
x = (np.random.rand(2048, 96, 96, 3) * 255).astype(np.uint8); y = np.random.randint(0, 2, 2048)
for m in ("baseline", "hed"):
    cfg = train.Config(method=m, epochs=1, batch_size=128, pretrained=os.environ.get("NO_PRETRAINED") != "1")
    t = time.time(); train.train_model(x, y, cfg, dev, log=lambda *_: None)
    if dev.type == "cuda": torch.cuda.synchronize()
    print(f"{m}: {len(x) / (time.time() - t):.0f} images/second")
