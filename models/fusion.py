"""
FusionModel: wraps C3D and optionally fuses with a second branch.

Modes
-----
"3d_only"
    Only the C3D branch is active. Use this until Person B's optical flow
    branch is ready.

"fusion"
    C3D features (4096-d) are concatenated with Person B's feature vector
    and passed through a 2-layer FC head for final classification.
    Call: model(x_3d, x_other=feat_b)
    where feat_b is (B, D) — adjust other_feat_dim below once Person B
    finalises their output dimension.
"""

import torch
import torch.nn as nn

from .c3d import C3D
import config


class FusionModel(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES, mode=config.FUSION_MODE):
        """
        Args:
            num_classes : total number of action categories
            mode        : "3d_only" or "fusion"
        """
        super().__init__()
        self.mode = mode

        self.c3d = C3D(
            num_classes=num_classes,
            dropout=config.DROPOUT,
            feature_only=(mode == "fusion"),
        )

        if mode == "fusion":
            # TODO (Person B): update other_feat_dim to match your branch output size
            other_feat_dim = 4096
            fused_dim = self.c3d.feature_dim + other_feat_dim   # 8192

            self.fusion_head = nn.Sequential(
                nn.Linear(fused_dim, 2048),
                nn.ReLU(inplace=True),
                nn.Dropout(config.DROPOUT),
                nn.Linear(2048, num_classes),
            )

    def forward(self, x_3d, x_other=None):
        """
        Args:
            x_3d   : (B, 3, T, H, W)  RGB video clip
            x_other: (B, D)            feature vector from Person B's branch
                                       (required only in fusion mode)
        Returns:
            logits : (B, num_classes)
        """
        if self.mode == "3d_only":
            return self.c3d(x_3d)

        feat_3d = self.c3d(x_3d)                          # (B, 4096)
        assert x_other is not None, (
            "fusion mode requires x_other — pass Person B's feature vector"
        )
        fused = torch.cat([feat_3d, x_other], dim=1)      # (B, 8192)
        return self.fusion_head(fused)                     # (B, num_classes)
