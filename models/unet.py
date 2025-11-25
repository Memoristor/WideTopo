# coding=utf-8

from segmentation_models_pytorch import Unet, UnetPlusPlus
from torch import nn

__all__ = [
    "UNetResNet18",
    "UNetResNet34",
    "UNetResNet50",
    "UNetResNet101",
    "UNetPlusPlusResNet18",
    "UNetPlusPlusResNet34",
    "UNetPlusPlusResNet50",
    "UNetPlusPlusResNet101",
]


class UNetResNet18(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.net = Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            classes=num_class,
        )

    def forward(self, x):
        output = self.net(x)
        return output


class UNetResNet34(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.net = Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            classes=num_class,
        )

    def forward(self, x):
        output = self.net(x)
        return output


class UNetResNet50(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.net = Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            classes=num_class,
        )

    def forward(self, x):
        output = self.net(x)
        return output


class UNetResNet101(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.net = Unet(
            encoder_name="resnet101",
            encoder_weights="imagenet",
            classes=num_class,
        )

    def forward(self, x):
        output = self.net(x)
        return output


class UNetPlusPlusResNet18(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.net = UnetPlusPlus(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            classes=num_class,
        )

    def forward(self, x):
        output = self.net(x)
        return output


class UNetPlusPlusResNet34(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.net = UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            classes=num_class,
        )

    def forward(self, x):
        output = self.net(x)
        return output


class UNetPlusPlusResNet50(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.net = UnetPlusPlus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            classes=num_class,
        )

    def forward(self, x):
        output = self.net(x)
        return output


class UNetPlusPlusResNet101(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.net = UnetPlusPlus(
            encoder_name="resnet101",
            encoder_weights="imagenet",
            classes=num_class,
        )

    def forward(self, x):
        output = self.net(x)
        return output
