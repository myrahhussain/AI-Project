import torch

from src import cam
from src.models import make_model


def _model():
    torch.manual_seed(0)
    return make_model(pretrained=False)


def test_default_cam_shape_and_range():
    m, x = _model(), torch.rand(3, 3, 96, 96)
    c = cam.compute_cam(m, x)
    assert c.shape == (3, 96, 96)
    assert float(c.min()) >= 0 and float(c.max()) <= 1 + 1e-6


def test_gradcam_and_layercam_run_on_every_layer():
    m, x = _model(), torch.rand(2, 3, 96, 96)
    for layer in cam.LAYERS:
        for method in ("gradcam", "layercam"):
            c = cam.compute_cam(m, x, layer, method)
            assert c.shape == (2, 96, 96) and torch.isfinite(c).all()
            assert float(c.min()) >= 0 and float(c.max()) <= 1 + 1e-6


def test_unknown_method_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        cam.compute_cam(_model(), torch.rand(1, 3, 96, 96), method="nope")


def test_hooks_are_removed_after_use():
    m, x = _model(), torch.rand(1, 3, 96, 96)
    cam.compute_cam(m, x)
    assert all(len(mod._forward_hooks) == 0 for mod in m.modules())


def test_fused_cam_uses_fine_layers_and_overlay_shape():
    import numpy as np
    m, x = _model(), torch.rand(2, 3, 96, 96)
    f = cam.fused_layercam(m, x)
    assert f.shape == (2, 96, 96) and float(f.max()) <= 1 + 1e-6
    o = cam.overlay((x[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8), f[0].numpy())
    assert o.shape == (96, 96, 3) and o.dtype == np.uint8


def test_earlier_layers_localise_better_than_layer4():
    """Trained on synthetic 'tumours' (dark purple square): the finer layer should put more of
    its map on the square than the 3x3 layer4 map does."""
    import numpy as np
    from src import train

    def synth(n, seed):
        rng = np.random.default_rng(seed)
        x = (rng.random((n, 96, 96, 3)) * 30 + 200).astype(np.float32)
        y = (rng.random(n) < 0.5).astype(np.int64)
        for i in np.where(y == 1)[0]:
            r, c = rng.integers(8, 48, 2)
            x[i, r:r + 40, c:c + 40] = [110, 60, 150]
        return x.clip(0, 255).astype(np.uint8), y

    x, y = synth(256, 0)
    cfg = train.Config(method="baseline", epochs=6, batch_size=32, pretrained=False, amp=False, lr=1e-3)
    model, _ = train.train_model(x, y, cfg, torch.device("cpu"), log=lambda *a: None)
    xt, yt = synth(64, 1)
    sel = np.where(yt == 1)[0][:16]
    t = torch.from_numpy(xt[sel]).permute(0, 3, 1, 2).float() / 255
    inside = torch.from_numpy(xt[sel][..., 1] < 100).float()
    frac = lambda c: ((c * inside).flatten(1).sum(1) / c.flatten(1).sum(1).clamp_min(1e-8)).mean().item()
    coarse = frac(cam.compute_cam(model, t, "backbone.layer4", "gradcam"))
    fine = frac(cam.compute_cam(model, t, "backbone.layer2", "layercam"))
    assert fine > coarse + 0.1
