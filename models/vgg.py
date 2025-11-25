# coding=utf-8

from torch import nn
from torchvision.models.vgg import *

from tools.weight_init import general_init

__all__ = ["VGG11", "VGG13", "VGG16", "VGG19"]


class VGG11(nn.Module):
    def __init__(self, num_class, pretrained_model=None):
        super().__init__()
        self.vgg = vgg11(num_classes=num_class)

        if not pretrained_model:
            self.apply(general_init)

    def forward(self, x):
        x = self.vgg(x)
        return x


class VGG11BN(nn.Module):
    def __init__(self, num_class, pretrained_model=None):
        super().__init__()
        self.vgg = vgg11_bn(num_classes=num_class)

        if not pretrained_model:
            self.apply(general_init)

    def forward(self, x):
        x = self.vgg(x)
        return x


class VGG13(nn.Module):
    def __init__(self, num_class, pretrained_model=None):
        super().__init__()
        self.vgg = vgg13(num_classes=num_class)

        if not pretrained_model:
            self.apply(general_init)

    def forward(self, x):
        x = self.vgg(x)
        return x


class VGG13BN(nn.Module):
    def __init__(self, num_class, pretrained_model=None):
        super().__init__()
        self.vgg = vgg13_bn(num_classes=num_class)

        if not pretrained_model:
            self.apply(general_init)

    def forward(self, x):
        x = self.vgg(x)
        return x


class VGG16(nn.Module):
    def __init__(self, num_class, pretrained_model=None):
        super().__init__()
        self.vgg = vgg16(num_classes=num_class)

        if not pretrained_model:
            self.apply(general_init)

    def forward(self, x):
        x = self.vgg(x)
        return x


class VGG16BN(nn.Module):
    def __init__(self, num_class, pretrained_model=None):
        super().__init__()
        self.vgg = vgg16_bn(num_classes=num_class)

        if not pretrained_model:
            self.apply(general_init)

    def forward(self, x):
        x = self.vgg(x)
        return x


class VGG19(nn.Module):
    def __init__(self, num_class, pretrained_model=None):
        super().__init__()
        self.vgg = vgg19(num_classes=num_class)

        if not pretrained_model:
            self.apply(general_init)

    def forward(self, x):
        x = self.vgg(x)
        return x


class VGG19BN(nn.Module):
    def __init__(self, num_class, pretrained_model=None):
        super().__init__()
        self.vgg = vgg19_bn(num_classes=num_class)

        if not pretrained_model:
            self.apply(general_init)

    def forward(self, x):
        x = self.vgg(x)
        return x
