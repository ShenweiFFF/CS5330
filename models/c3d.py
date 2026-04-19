"""
C3D: Learning Spatiotemporal Features with 3D Convolutional Networks
  Tran et al., ICCV 2015

Input shape:  (B, 3, 16, 112, 112)
Output shape: (B, num_classes)  or  (B, 4096) when feature_only=True
"""

import torch
import torch.nn as nn


class C3D(nn.Module):
    def __init__(self, num_classes=101, dropout=0.5, feature_only=False):
        """
        Args:
            num_classes  : number of output classes (ignored when feature_only=True)
            dropout      : dropout probability applied after fc6 and fc7
            feature_only : if True, return 4096-d feature vector instead of logits;
                           used when fusing with a second branch
        """
        super().__init__()
        self.feature_only = feature_only

        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv5 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2)

        # After pool5: (B, 512, 1, 3, 3) -> flatten -> 4608
        self.fc6 = nn.Sequential(
            nn.Linear(512 * 1 * 3 * 3, 4096), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.fc7 = nn.Sequential(
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(dropout))

        if not feature_only:
            self.classifier = nn.Linear(4096, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 3, 16, 112, 112)
        x = self.pool1(self.conv1(x))   # (B,  64, 16, 56, 56)
        x = self.pool2(self.conv2(x))   # (B, 128,  8, 28, 28)
        x = self.pool3(self.conv3(x))   # (B, 256,  4, 14, 14)
        x = self.pool4(self.conv4(x))   # (B, 512,  2,  7,  7)
        x = self.pool5(self.conv5(x))   # (B, 512,  1,  3,  3)

        x = x.flatten(1)                # (B, 4608)
        x = self.fc6(x)                 # (B, 4096)
        x = self.fc7(x)                 # (B, 4096)

        if self.feature_only:
            return x                    # (B, 4096)
        return self.classifier(x)       # (B, num_classes)

    @property
    def feature_dim(self):
        return 4096
