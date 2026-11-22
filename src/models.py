"""Model definition: one architecture (ResNet-18) is used for every method."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class PatchClassifier(nn.Module):
    """ResNet-18 with a 1-logit head. Input: float images in [0, 1], shape (N, 3, H, W).

    ImageNet mean/std normalisation happens inside the model so that all stain
    augmentation can work in plain [0, 1] RGB.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = torchvision.models.resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
        self.register_buffer("mean", torch.tensor(_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone((x - self.mean) / self.std).squeeze(1)  # logits, shape (N,)


def make_model(pretrained: bool = True) -> PatchClassifier:
    return PatchClassifier(pretrained)
