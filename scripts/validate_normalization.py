"""Compare our stain-matrix estimate with torchstain (Macenko) on the same patches, if installable."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src import normalize


def main(npy=None):
    if npy: imgs = np.load(npy)[:200]
    else:
        print("No real patches given (pass a .npy path): COMPARISON NOT RUN on real data."); return 1
    ours = normalize.StainNormalizer().estimate(imgs)[0]
    try:
        import torchstain  # noqa: F401
    except Exception as e:
        print(f"torchstain not installed ({e}): COMPARISON NOT RUN"); print("our W:\n", ours); return 1
    import torch
    n = torchstain.normalizers.MacenkoNormalizer(backend="numpy")
    n.fit(imgs[0])
    ref = np.array(n.HERef)
    ref = ref / np.linalg.norm(ref, axis=0, keepdims=True)
    cos = [float(np.dot(ours[i], ref[:, i])) for i in range(2)]
    print("cosine similarity (ours vs torchstain Macenko H, E):", cos)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
