"""
backbone.py
-----------
Shared convolutional backbone used by the multi-branch ReID network.

The backbone extracts a spatial feature map that is then consumed by
several task-specific heads (identity, clothes, body-attributes, ...),
implemented in identity_branch.py / attribute_branch.py.

A ResNet-50 (IBN-a style, stride-reduced in the last stage as is common
in ReID literature) is used by default because it offers a good
accuracy/speed trade-off, but any backbone that exposes a
`forward_features(x) -> (B, C, H, W)` map can be substituted.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torchvision


class IBN(nn.Module):
    """Instance-Batch Normalization layer (half IN, half BN).

    IBN layers are known to improve cross-domain generalization in ReID,
    which matters here because the model is evaluated across camera
    domains (see the automatic-annotation / cross-domain evaluation
    section of the pipeline).
    """

    def __init__(self, num_channels: int):
        super().__init__()
        half = num_channels // 2
        self.half = half
        self.instance_norm = nn.InstanceNorm2d(half, affine=True)
        self.batch_norm = nn.BatchNorm2d(num_channels - half)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        split = torch.split(x, [self.half, x.size(1) - self.half], dim=1)
        out1 = self.instance_norm(split[0].contiguous())
        out2 = self.batch_norm(split[1].contiguous())
        return torch.cat([out1, out2], dim=1)


class ReIDBackbone(nn.Module):
    """
    ResNet-50 backbone adapted for ReID:
      * last stage stride reduced 2 -> 1 to keep higher spatial resolution
      * first two residual stages get IBN normalization for domain
        robustness.
    """

    def __init__(self, pretrained: bool = True, last_stride: int = 1):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = torchvision.models.resnet50(weights=weights)

        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self._apply_ibn(self.layer1)
        self._apply_ibn(self.layer2)

        if last_stride == 1:
            self._set_stride_one(self.layer4)

        self.out_channels = 2048

    @staticmethod
    def _apply_ibn(layer: nn.Sequential) -> None:
        for block in layer:
            out_channels = block.bn1.num_features
            block.bn1 = IBN(out_channels)

    @staticmethod
    def _set_stride_one(layer: nn.Sequential) -> None:
        first_block = layer[0]
        first_block.conv2.stride = (1, 1)
        if first_block.downsample is not None:
            first_block.downsample[0].stride = (1, 1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # (B, 2048, H/16, W/16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)
