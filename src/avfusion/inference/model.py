"""VehicleSoundNet: small MobileNetV2-style CNN over log-mel patches.

~0.6 M params — sized for the job: 5-class vehicle-sound detection at 1 Hz on
an edge box, not AudioSet's 521 classes. For higher recall, transfer-learn
from PANNs CNN10 (torch, AudioSet-pretrained) and export the same way; the
ONNX/TRT path downstream is identical (tools/export_onnx.py --arch panns).

Input:  [B, 1, 64, 96]  standardized log-mel
Output: [B, num_classes] logits
"""
import torch
import torch.nn as nn


class _InvertedResidual(nn.Module):
    def __init__(self, c_in, c_out, stride, expand=4):
        super().__init__()
        hidden = c_in * expand
        self.use_res = stride == 1 and c_in == c_out
        self.block = nn.Sequential(
            nn.Conv2d(c_in, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
        )

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_res else out


class VehicleSoundNet(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU6(inplace=True))
        self.blocks = nn.Sequential(
            _InvertedResidual(16, 24, stride=2),
            _InvertedResidual(24, 24, stride=1),
            _InvertedResidual(24, 48, stride=2),
            _InvertedResidual(48, 48, stride=1),
            _InvertedResidual(48, 96, stride=2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.2), nn.Linear(96, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))
