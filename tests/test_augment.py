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


def test_hed_to_rgb_matches_skimage():
    rng = np.random.default_rng(1)
    hed = (rng.random((2, 24, 24, 3)) * 0.6).astype(np.float32)
    ref = np.stack([hed2rgb(h) for h in hed])
    ours = augment.hed_to_rgb(torch.from_numpy(hed)).permute(0, 2, 3, 1).numpy()
    assert np.allclose(ours, ref, atol=1e-4)


def test_roundtrip_recovers_stain_concentrations():
    rng = np.random.default_rng(2)
    hed = torch.from_numpy((rng.random((2, 16, 16, 3)) * 0.4).astype(np.float32))
    rgb = augment.hed_to_rgb(hed)
    back = augment.rgb_to_hed(rgb)
    assert torch.allclose(back, hed, atol=2e-3)


def _tissue_like(n=4, size=32, seed=3):
    rng = np.random.default_rng(seed)
    hed = torch.from_numpy((rng.random((n, size, size, 3)) * np.array([0.5, 0.4, 0.05])).astype(np.float32))
    return augment.hed_to_rgb(hed)


def test_sigma_zero_is_identity():
    x = _tissue_like()
    assert torch.equal(augment.hed_augment(x, 0.0), x)


def test_output_shape_and_range():
    x = _tissue_like()
    y = augment.hed_augment(x, 0.2, torch.Generator().manual_seed(0))
    assert y.shape == x.shape
    assert float(y.min()) >= 0 and float(y.max()) <= 1
    assert torch.isfinite(y).all()


def test_seeded_augmentation_is_deterministic_and_seeds_differ():
    x = _tissue_like()
    a = augment.hed_augment(x, 0.2, torch.Generator().manual_seed(7))
    b = augment.hed_augment(x, 0.2, torch.Generator().manual_seed(7))
    c = augment.hed_augment(x, 0.2, torch.Generator().manual_seed(8))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_larger_sigma_changes_colours_more():
    x = _tissue_like(n=16)
    diffs = []
    for s in (0.05, 0.1, 0.2):
        y = augment.hed_augment(x, s, torch.Generator().manual_seed(0))
        diffs.append(float((y - x).abs().mean()))
    assert diffs[0] < diffs[1] < diffs[2]
