"""Class activation maps for the ResNet-18 patch classifier.

Stage 2: Grad-CAM and LayerCAM, selectable layer, and a multi-layer fusion.
At 96x96 input the last ResNet stage (layer4) is only 3x3, so its maps are very
coarse; layer2 (12x12) and layer3 (6x6) localise better.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

LAYERS = ("backbone.layer2", "backbone.layer3", "backbone.layer4")


def _layer(model, name: str):
    return dict(model.named_modules())[name]


def _normalise(cam: torch.Tensor) -> torch.Tensor:
    top = cam.flatten(1).max(dim=1).values.clamp_min(1e-8).view(-1, 1, 1)
    return cam / top


def compute_cam(model, x: torch.Tensor, layer_name: str = "backbone.layer4", method: str = "gradcam") -> torch.Tensor:
    """CAM for the tumour logit. x: (N, 3, H, W) in [0, 1]. Returns (N, H, W) in [0, 1].

    method "gradcam":  channel weights = spatial mean of the gradient.
    method "layercam": every location is weighted by its own positive gradient,
                       which keeps spatial detail from earlier layers.
    """
    if method not in ("gradcam", "layercam"):
        raise ValueError(f"unknown method {method}")
    store = {}

    def fwd(_m, _i, out):
        store["a"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))

    handle = _layer(model, layer_name).register_forward_hook(fwd)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        model(x).sum().backward()          # samples are independent, so one backward covers all
    finally:
        handle.remove()
    a, g = store["a"].detach(), store["g"].detach()
    if method == "gradcam":
        cam = F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(dim=1, keepdim=True))
    else:
        cam = F.relu((F.relu(g) * a).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    return _normalise(cam)


def fused_layercam(model, x: torch.Tensor, layers=LAYERS) -> torch.Tensor:
    """Average of per-layer LayerCAMs (each normalised first)."""
    return _normalise(torch.stack([compute_cam(model, x, l, "layercam") for l in layers]).mean(dim=0))
