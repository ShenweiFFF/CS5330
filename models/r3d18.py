"""
3D ResNet-18 for spatiotemporal feature extraction.

Architecture reference:
  Tran et al. "A Closer Look at Spatiotemporal Convolutions for Action
  Recognition," CVPR 2018.  (R3D variant — full 3D convolutions throughout)

Input shape:  (B, 3, T, H, W)   — e.g. (B, 3, 16, 224, 224)
Output shape: (B, 512)           when feature_only=True
              (B, num_classes)   otherwise
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BasicBlock3d(nn.Module):
    """Two 3×3×3 convolutions with a residual connection.

    Spatial stride is applied to the first conv and the downsample shortcut;
    temporal stride is fixed at 1 to preserve motion information across frames.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_ch, out_ch, kernel_size=3,
            stride=(1, stride, stride), padding=1, bias=False,
        )
        self.bn1  = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_ch, out_ch, kernel_size=3,
            stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_ch)

        self.downsample: nn.Module | None = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1,
                          stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def _make_layer(in_ch: int, out_ch: int, num_blocks: int,
                stride: int) -> nn.Sequential:
    layers: list[nn.Module] = [BasicBlock3d(in_ch, out_ch, stride)]
    for _ in range(1, num_blocks):
        layers.append(BasicBlock3d(out_ch, out_ch))
    return nn.Sequential(*layers)


class R3D18(nn.Module):
    """3D ResNet-18 backbone.

    Args:
        num_classes  : number of action categories (ignored when feature_only=True)
        feature_only : if True, return the 512-d feature vector before the
                       classification head — used by FusionModel
    """

    feature_dim: int = 512

    def __init__(self, num_classes: int = 101,
                 feature_only: bool = False) -> None:
        super().__init__()
        self.feature_only = feature_only

        # Stem: temporal kernel-3 to capture short-range motion
        self.stem = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7),
                      stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3),
                         stride=(1, 2, 2), padding=(0, 1, 1)),
        )

        self.layer1 = _make_layer(64,  64,  num_blocks=2, stride=1)
        self.layer2 = _make_layer(64,  128, num_blocks=2, stride=2)
        self.layer3 = _make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = _make_layer(256, 512, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        if not feature_only:
            self.fc = nn.Linear(512, num_classes)
            nn.init.normal_(self.fc.weight, 0, 0.01)
            nn.init.zeros_(self.fc.bias)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, T, H, W)
        x = self.stem(x)       # (B,  64, T,    H/4,  W/4)
        x = self.layer1(x)     # (B,  64, T,    H/4,  W/4)
        x = self.layer2(x)     # (B, 128, T,    H/8,  W/8)
        x = self.layer3(x)     # (B, 256, T,    H/16, W/16)
        x = self.layer4(x)     # (B, 512, T,    H/32, W/32)
        x = self.avgpool(x)    # (B, 512, 1,    1,    1)
        x = x.flatten(1)       # (B, 512)
        if self.feature_only:
            return x
        return self.fc(x)      # (B, num_classes)
