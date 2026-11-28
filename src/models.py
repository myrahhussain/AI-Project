"""Model definitions. Every backbone ends in a single tumour logit.

All backbones are fine-tuned end to end with the same recipe (optimiser, epochs, data),
so only the network and its starting weights change.

  resnet18        ResNet-18 (He et al. 2016), ImageNet weights. Our main model.
  resnet18_histo  Same ResNet-18, weights from self-supervised training on 57 histology
                  datasets (Ciga, Xu & Martel 2022). Isolates "ImageNet vs histology start".
  resnet50_histo  ResNet-50, MoCo v2 self-supervised on ~33M TCGA + internal histology patches
                  (Kang et al. 2023, Lunit). Its pretraining data does not include CAMELYON.
  convnext_tiny   ConvNeXt-T (Liu et al. 2022), ImageNet weights. A modern CNN (optional).

Weights for the two histology backbones are downloaded by the notebook and passed in with
`weights_path` (see WEIGHT_URLS). Checkpoints from the interim runs (resnet18) still load,
because the resnet18 module layout is unchanged.
"""
from __future__ import annotations

import pickle
import types

import torch
import torch.nn as nn
import torchvision

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BACKBONES = ("resnet18", "resnet18_histo", "resnet50_histo", "convnext_tiny")

WEIGHT_URLS = {
    "resnet18_histo": "https://github.com/ozanciga/self-supervised-histopathology/releases/download/"
                      "tenpercent/tenpercent_resnet18.ckpt",
    "resnet50_histo": "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/"
                      "pretrained-weights/mocov2_rn50_ep200.torch",
}


# ---------------------------------------------------------------- safe checkpoint loading
class _Stub:
    """Stands in for non-tensor objects saved inside training checkpoints (e.g. Lightning callbacks)."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


_ALLOWED_GLOBALS = {
    ("collections", "OrderedDict"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_parameter"),
    ("torch", "Size"),
}


class _RestrictedUnpickler(pickle.Unpickler):
    """Only rebuilds tensors and dicts; every other class becomes a harmless stub, so no
    code stored in the file can run."""

    def find_class(self, module, name):
        if (module, name) in _ALLOWED_GLOBALS or (module == "torch" and name.endswith("Storage")):
            return super().find_class(module, name)
        return _Stub


_restricted_pickle = types.SimpleNamespace(Unpickler=_RestrictedUnpickler, load=pickle.load, __name__="restricted_pickle")


def load_state_dict_file(path: str) -> dict:
    """Read a checkpoint's tensors without executing anything stored in it."""
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        obj = torch.load(path, map_location="cpu", pickle_module=_restricted_pickle, weights_only=False)
    if isinstance(obj, dict) and isinstance(obj.get("state_dict"), dict):
        obj = obj["state_dict"]
    return {k: v for k, v in obj.items() if torch.is_tensor(v)}


def load_backbone_weights(net: nn.Module, state: dict, strip=("model.", "resnet."), min_frac: float = 0.95) -> int:
    """Copy matching tensors into `net`, ignoring its classification head.

    Raises if fewer than `min_frac` of the non-head tensors match, so a wrong file fails loudly
    instead of silently training from random weights. Returns the number of tensors loaded.
    """
    renamed = {}
    for k, v in state.items():
        for s in strip:
            k = k.replace(s, "")
        renamed[k] = v
    own = net.state_dict()
    body = [k for k in own if not k.startswith("fc.")]
    matched = {k: renamed[k] for k in body if k in renamed and renamed[k].shape == own[k].shape}
    if len(matched) < min_frac * len(body):
        missing = [k for k in body if k not in matched][:5]
        raise RuntimeError(f"only {len(matched)}/{len(body)} backbone tensors matched; e.g. missing {missing}")
    own.update(matched)
    net.load_state_dict(own)
    return len(matched)


# ---------------------------------------------------------------- the classifier
class PatchClassifier(nn.Module):
    """Backbone with a 1-logit head. Input: float images in [0, 1], shape (N, 3, H, W).

    Per-channel mean/std normalisation happens inside the model so that stain augmentation
    can work in plain [0, 1] RGB. All backbones use ImageNet statistics (the histology
    weight releases do not publish their own).
    """

    def __init__(self, pretrained: bool = True, backbone: str = "resnet18", weights_path: str | None = None):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"unknown backbone {backbone}; choose from {BACKBONES}")
        self.backbone_name = backbone
        if backbone == "resnet18":
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = torchvision.models.resnet18(weights=weights)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
        elif backbone in ("resnet18_histo", "resnet50_histo"):
            arch = torchvision.models.resnet18 if backbone == "resnet18_histo" else torchvision.models.resnet50
            self.backbone = arch(weights=None)
            if pretrained:
                if not weights_path:
                    raise ValueError(f"{backbone} needs weights_path (download from {WEIGHT_URLS[backbone]})")
                load_backbone_weights(self.backbone, load_state_dict_file(weights_path))
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
        else:  # convnext_tiny
            weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = torchvision.models.convnext_tiny(weights=weights)
            self.backbone.classifier[2] = nn.Linear(self.backbone.classifier[2].in_features, 1)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone((x - self.mean) / self.std).reshape(-1)  # logits, shape (N,)


def make_model(pretrained: bool = True, backbone: str = "resnet18", weights_path: str | None = None) -> PatchClassifier:
    return PatchClassifier(pretrained, backbone, weights_path)


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
