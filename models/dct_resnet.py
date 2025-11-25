# coding=utf-8

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from torch import nn
from torchvision.models.resnet import resnet18, resnet34, resnet50

from layers.dct_layers import NaiveDCT2D

__all__ = [
    "DCTResNet18",
    "DCTResNet34",
    "DCTResNet50",
]


class DCTResNet(nn.Module):
    def __init__(self, num_class, block_points, pretrained_model):
        super().__init__()
        resnet = pretrained_model
        self.freq = NaiveDCT2D(block_points)
        self.layer0 = nn.Sequential(
            nn.Conv2d(
                3 * block_points * block_points,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            resnet.bn1,
            resnet.relu,
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(512, num_class)

    def forward(self, x):
        x = self.freq(x)
        n, c, f, h, w = x.size()
        x = x.view(n, -1, h, w)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class DCTResNet18(DCTResNet):
    def __init__(self, num_class, block_points, pretrained_resnet=True):
        super().__init__(num_class, block_points, resnet18(pretrained=pretrained_resnet))


class DCTResNet34(DCTResNet):
    def __init__(self, num_class, block_points, pretrained_resnet=True):
        super().__init__(num_class, block_points, resnet34(pretrained=pretrained_resnet))


class DCTResNet50(DCTResNet):
    def __init__(self, num_class, block_points, pretrained_resnet=True):
        super().__init__(num_class, block_points, resnet50(pretrained=pretrained_resnet))
        self.fc = nn.Linear(2048, num_class)
