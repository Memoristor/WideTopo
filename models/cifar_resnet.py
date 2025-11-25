# coding=utf-8

import torch
from torch import nn
from torchvision.models.resnet import *

from tools.weight_init import general_init

__all__ = ["CIFARResNet18", "CIFARResNet34", "CIFARResNet50", "CIFARResNet101"]


class CIFARResNet(nn.Module):
    def __init__(self, num_class, pretrained_model):
        super().__init__()
        resnet = pretrained_model
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
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class CIFARResNet18(CIFARResNet):
    def __init__(self, num_class, pretrained=True):
        super().__init__(num_class, resnet18(pretrained=pretrained))

        if not pretrained:
            self.apply(general_init)


class CIFARResNet34(CIFARResNet):
    def __init__(self, num_class, pretrained=True):
        super().__init__(num_class, resnet34(pretrained=pretrained))

        if not pretrained:
            self.apply(general_init)


class CIFARResNet50(CIFARResNet):
    def __init__(self, num_class, pretrained=True):
        super().__init__(num_class, resnet50(pretrained=pretrained))
        self.fc = nn.Linear(2048, num_class)

        if not pretrained:
            self.apply(general_init)


class CIFARResNet101(CIFARResNet):
    def __init__(self, num_class, pretrained=True):
        super().__init__(num_class, resnet101(pretrained=pretrained))
        self.fc = nn.Linear(2048, num_class)

        if not pretrained:
            self.apply(general_init)
