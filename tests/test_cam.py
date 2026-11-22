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
