# coding=utf-8

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import *

__all__ = [
    "BlockResNet18",
    "BlockResNet34",
    "BlockResNet50",
]


class BlockResNet(nn.Module):
    def __init__(self, num_class, block_points, pretrained_model):
        super().__init__()
        resnet = pretrained_model
        self.block_points = block_points
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
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
        _, _, h, w = x.size()
        x = F.interpolate(x, (int(h / self.block_points), int(w / self.block_points)))
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class BlockResNet18(BlockResNet):
    def __init__(self, num_class, block_points, pretrained_resnet=True):
        super().__init__(num_class, block_points, resnet18(pretrained=pretrained_resnet))


class BlockResNet34(BlockResNet):
    def __init__(self, num_class, block_points, pretrained_resnet=True):
        super().__init__(num_class, block_points, resnet34(pretrained=pretrained_resnet))


class BlockResNet50(BlockResNet):
    def __init__(self, num_class, block_points, pretrained_resnet=True):
        super().__init__(num_class, block_points, resnet50(pretrained=pretrained_resnet))
        self.fc = nn.Linear(2048, num_class)
