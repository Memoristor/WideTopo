# coding=utf-8

from pytorch_pretrained_vit import *
from torch import nn
from torch.nn import init

__all__ = [
    "ViTB16IN21K",
    "ViTB16IN1K",
    "ViTB32IN21K",
    "ViTB32IN1K",
]


class PretrainedVIT(nn.Module):
    def __init__(
        self,
        num_classes,
        name,
        image_size=224,
        pretrained=True,
    ):
        super().__init__()
        self.net = ViT(
            name=name,
            pretrained=pretrained,
            num_classes=num_classes,
            image_size=image_size,
        )

        fc = self.net.fc
        init.trunc_normal_(fc.weight, std=0.02)
        if fc.bias is not None:
            init.constant_(fc.bias, 0)

    def forward(self, x):
        output = self.net(x)
        return output

    def __getstate__(self):
        for block in self.net.transformer.blocks:
            block.attn.scores = None  # item `block.attn.scores` can not be deep copied
        return self.__dict__.copy()


class ViTB16IN21K(PretrainedVIT):
    def __init__(
        self,
        num_classes,
        name="B_16",
        image_size=224,
        pretrained=True,
    ):
        super(ViTB16IN21K, self).__init__(
            name=name,
            pretrained=pretrained,
            num_classes=num_classes,
            image_size=image_size,
        )


class ViTB16IN1K(PretrainedVIT):
    def __init__(
        self,
        num_classes,
        name="B_16_imagenet1k",
        image_size=224,
        pretrained=True,
    ):
        super(ViTB16IN1K, self).__init__(
            name=name,
            pretrained=pretrained,
            num_classes=num_classes,
            image_size=image_size,
        )


class ViTB32IN21K(PretrainedVIT):
    def __init__(
        self,
        num_classes,
        name="B_32",
        image_size=224,
        pretrained=True,
    ):
        super(ViTB32IN21K, self).__init__(
            name=name,
            pretrained=pretrained,
            num_classes=num_classes,
            image_size=image_size,
        )


class ViTB32IN1K(PretrainedVIT):
    def __init__(
        self,
        num_classes,
        name="B_32_imagenet1k",
        image_size=224,
        pretrained=True,
    ):
        super(ViTB32IN1K, self).__init__(
            name=name,
            pretrained=pretrained,
            num_classes=num_classes,
            image_size=image_size,
        )
