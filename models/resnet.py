"""CIFAR-adapted ResNet18 and ResNet34.

Standard ImageNet ResNets use a 7x7 stride-2 stem + maxpool which reduces
32x32 inputs too aggressively. We replace the stem with a 3x3 stride-1 conv
and drop the maxpool — a standard adaptation for CIFAR-sized images.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class ResNet(nn.Module):
    """Generic CIFAR-adapted ResNet with custom head."""

    def __init__(
        self,
        block: type[BasicBlock],
        layers: list[int],
        num_classes: int = 100,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = 64

        # CIFAR stem: 3x3 stride-1 preserves spatial resolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1, dropout=dropout)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dropout=dropout)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dropout=dropout)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dropout=dropout)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(
        self,
        block: type[BasicBlock],
        out_channels: int,
        num_blocks: int,
        stride: int,
        dropout: float,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, stride=s, dropout=dropout))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.fc(torch.flatten(x, 1))


def ResNet18(num_classes: int = 100, dropout: float = 0.1) -> ResNet:
    """ResNet-18 with CIFAR-adapted stem and custom classification head."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, dropout=dropout)


def ResNet34(num_classes: int = 100, dropout: float = 0.1) -> ResNet:
    """ResNet-34 with CIFAR-adapted stem and custom classification head."""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, dropout=dropout)
