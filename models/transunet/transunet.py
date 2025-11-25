# coding=utf-8

import os

import numpy as np
from torch import nn

from .vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from .vit_seg_modeling import VisionTransformer

__all__ = [
    "TransUNetR50ViTB16",
    "TransUNetR50ViTL16",
    "TransUNetViTB16",
    "TransUNetViTB32",
]


class TransUNet(nn.Module):
    def __init__(self, vit_name, num_class, img_size=(224, 224)):
        super(TransUNet, self).__init__()
        config_vit = CONFIGS_ViT_seg[vit_name]
        config_vit.n_classes = num_class

        if "R50" in vit_name:
            config_vit.patches.grid = (
                int(img_size[0] / config_vit.patches.size[0]),
                int(img_size[1] / config_vit.patches.size[1]),
            )

        self.net = VisionTransformer(
            config_vit, img_size=img_size, num_classes=config_vit.n_classes
        )

        if os.path.exists(config_vit.pretrained_path):
            print(f"found weight at: {config_vit.pretrained_path}")
            self.net.load_from(weights=np.load(config_vit.pretrained_path))
        else:
            print(f"can not load weight from: {config_vit.pretrained_path}")

    def forward(self, x):
        return self.net(x)


class TransUNetR50ViTB16(TransUNet):
    def __init__(self, num_class, img_size=(224, 224)):
        super(TransUNetR50ViTB16, self).__init__("R50-ViT-B_16", num_class, img_size)


class TransUNetR50ViTL16(TransUNet):
    def __init__(self, num_class, img_size=(224, 224)):
        super(TransUNetR50ViTL16, self).__init__("R50-ViT-L_16", num_class, img_size)


class TransUNetViTB16(TransUNet):
    def __init__(self, num_class, img_size=(224, 224)):
        super(TransUNetViTB16, self).__init__("ViT-B_16", num_class, img_size)


class TransUNetViTB32(TransUNet):
    def __init__(self, num_class, img_size=(224, 224)):
        super(TransUNetViTB32, self).__init__("ViT-B_32", num_class, img_size)
