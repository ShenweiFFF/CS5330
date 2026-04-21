"""
FusionModel: two-branch action recognition network.

Architecture (paper §III.A):
  RGB branch   → R3D18         → 512-d spatiotemporal feature
  Flow branch  → 2D ResNet-18  → 512-d motion feature  (Person B)
  Fusion head  → Linear(1024 → 512) → ReLU → Dropout(0.5) → Linear(512 → 101)

Modes
-----
"rgb_only"
    Only the RGB branch (R3D18) is active. Returns class logits directly.
    Use this mode when training or evaluating the RGB branch in isolation.

"fusion"
    Both branches are active. Call:
        model(x_rgb, x_flow=feat_flow)
    where feat_flow is (B, 512) — the 512-d feature from Person B's
    flow ResNet-18 (avgpool output before the classification FC).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .r3d18 import R3D18
import config


class FusionModel(nn.Module):
    def __init__(self, num_classes: int = config.NUM_CLASSES,
                 mode: str = config.FUSION_MODE) -> None:
        """
        Args:
            num_classes : total number of action categories
            mode        : "rgb_only" | "fusion"
        """
        super().__init__()
        self.mode = mode

        self.rgb_branch = R3D18(
            num_classes=num_classes,
            feature_only=(mode == "fusion"),
        )

        if mode == "fusion":
            # Flow branch output dimension — matches Person B's ResNet-18 avgpool
            flow_feat_dim = 512
            fused_dim     = self.rgb_branch.feature_dim + flow_feat_dim   # 1024

            self.fusion_head = nn.Sequential(
                nn.Linear(fused_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(config.DROPOUT),
                nn.Linear(512, num_classes),
            )

    def forward(self, x_rgb: torch.Tensor,
                x_flow: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x_rgb  : (B, 3, T, H, W)  RGB video clip
            x_flow : (B, 512)          512-d flow feature from Person B's branch
                                       (required only in fusion mode)
        Returns:
            logits : (B, num_classes)
        """
        if self.mode == "rgb_only":
            return self.rgb_branch(x_rgb)

        feat_rgb = self.rgb_branch(x_rgb)                   # (B, 512)
        assert x_flow is not None, (
            "fusion mode requires x_flow — pass Person B's 512-d feature vector"
        )
        fused = torch.cat([feat_rgb, x_flow], dim=1)        # (B, 1024)
        return self.fusion_head(fused)                       # (B, num_classes)
