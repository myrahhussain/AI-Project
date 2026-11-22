"""Class activation maps for the ResNet-18 patch classifier (stage 1: Grad-CAM)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _layer(model, name: str):
    return dict(model.named_modules())[name]


def grad_cam(model, x: torch.Tensor, layer_name: str = "backbone.layer4") -> torch.Tensor:
    """Grad-CAM for the tumour logit. x: (N, 3, H, W) in [0, 1]. Returns (N, H, W) in [0, 1]."""
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
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    top = cam.flatten(1).max(dim=1).values.clamp_min(1e-8).view(-1, 1, 1)
    return cam / top
