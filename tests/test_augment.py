import numpy as np
import pytest
import torch
from skimage.color import hed2rgb, hed_from_rgb, rgb2hed, rgb_from_hed

from src import augment


def test_stain_matrices_match_skimage():
    ours = torch.tensor(augment._RGB_FROM_HED, dtype=torch.float64)
    assert np.allclose(ours.numpy(), rgb_from_hed, atol=1e-8)
    assert np.allclose(torch.linalg.inv(ours).numpy(), hed_from_rgb, atol=1e-8)


def test_rgb_to_hed_matches_skimage():
    rng = np.random.default_rng(0)
    img = rng.random((2, 24, 24, 3)).astype(np.float32)
    ref = np.stack([rgb2hed(i) for i in img])
    ours = augment.rgb_to_hed(torch.from_numpy(img).permute(0, 3, 1, 2)).numpy()
    assert np.allclose(ours, ref, atol=1e-4)
