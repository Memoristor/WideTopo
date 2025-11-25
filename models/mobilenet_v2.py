# coding=utf-8

from torch import nn
from torchvision.models import mobilenet_v2
from torchvision.models.mobilenet import MobileNet_V2_Weights

from tools import weight_init

__all__ = ["MobileNetV2"]


class MobileNetV2(nn.Module):
    def __init__(self, pretrained=True, **kwargs):
        super(MobileNetV2, self).__init__()

        if pretrained:
            self.net = mobilenet_v2(
                weights=MobileNet_V2_Weights.IMAGENET1K_V2,
                **kwargs,
            )
        else:
            self.net = mobilenet_v2(**kwargs)
            self.net.apply(weight_init.kaiming_normal_init)

    def forward(self, x):
        return self.net(x)
