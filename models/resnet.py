# coding=utf-8

import torch
from torch import nn
from torchvision.models.resnet import *

from tools.weight_init import general_init

__all__ = ["ResNet18", "ResNet34", "ResNet50", "ResNet101"]


class ResNetImp(nn.Module):
    def __init__(self, num_class, pretrained_model):
        super().__init__()
        resnet = pretrained_model
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(512, num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet18(ResNetImp):
    def __init__(self, num_class, pretrained=True):
        super().__init__(num_class, resnet18(pretrained=pretrained))

        if not pretrained:
            self.apply(general_init)


class ResNet34(ResNetImp):
    def __init__(self, num_class, pretrained=True):
        super().__init__(num_class, resnet34(pretrained=pretrained))

        if not pretrained:
            self.apply(general_init)


class ResNet50(ResNetImp):
    def __init__(self, num_class, pretrained=True):
        super().__init__(num_class, resnet50(pretrained=pretrained))
        self.fc = nn.Linear(2048, num_class)

        if not pretrained:
            self.apply(general_init)


class ResNet101(ResNetImp):
    def __init__(self, num_class, pretrained=True):
        super().__init__(num_class, resnet101(pretrained=pretrained))
        self.fc = nn.Linear(2048, num_class)

        if not pretrained:
            self.apply(general_init)
