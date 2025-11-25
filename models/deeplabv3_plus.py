# coding=utf-8

from segmentation_models_pytorch import DeepLabV3Plus
from torch import nn

__all__ = [
    "DeepLabV3PlusResNet18",
    "DeepLabV3PlusResNet34",
    "DeepLabV3PlusResNet50",
    "DeepLabV3PlusResNet101",
]


class DeepLabV3PlusResNet18(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.net = DeepLabV3Plus(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            classes=num_class,
        )

    def forward(self, x):
        output = self.net(x)
        return output


class DeepLabV3PlusResNet34(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.net = DeepLabV3Plus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            classes=num_class,
        )

    def forward(self, x):
        output = self.net(x)
        return output


class DeepLabV3PlusResNet50(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.net = DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            classes=num_class,
        )

    def forward(self, x):
        output = self.net(x)
        return output


class DeepLabV3PlusResNet101(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.net = DeepLabV3Plus(
            encoder_name="resnet101",
            encoder_weights="imagenet",
            classes=num_class,
        )

    def forward(self, x):
        output = self.net(x)
        return output
