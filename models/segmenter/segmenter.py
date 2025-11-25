# coding=utf-8


import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.weight_init import load_init

from .factory import create_decoder, create_vit
from .utils import padding, unpadding

__all__ = [
    "Segmenter",
    "SegmT16Mask",
    "SegmS16Mask",
]

configs = {
    "model": {
        "deit_tiny_distilled_patch16_224": {
            "image_size": 224,
            "patch_size": 16,
            "d_model": 192,
            "n_heads": 3,
            "n_layers": 12,
            "normalization": "deit",
            "distilled": True,
        },
        "deit_small_distilled_patch16_224": {
            "image_size": 224,
            "patch_size": 16,
            "d_model": 384,
            "n_heads": 6,
            "n_layers": 12,
            "normalization": "deit",
            "distilled": True,
        },
        "deit_base_distilled_patch16_224": {
            "image_size": 224,
            "patch_size": 16,
            "d_model": 768,
            "n_heads": 12,
            "n_layers": 12,
            "normalization": "deit",
            "distilled": True,
        },
        "deit_base_distilled_patch16_384": {
            "image_size": 384,
            "patch_size": 16,
            "d_model": 768,
            "n_heads": 12,
            "n_layers": 12,
            "normalization": "deit",
            "distilled": True,
        },
        "vit_base_patch8_384": {
            "image_size": 384,
            "patch_size": 8,
            "d_model": 768,
            "n_heads": 12,
            "n_layers": 12,
            "normalization": "vit",
            "distilled": False,
        },
        "vit_tiny_patch16_384": {
            "image_size": 384,
            "patch_size": 16,
            "d_model": 192,
            "n_heads": 3,
            "n_layers": 12,
            "normalization": "vit",
            "distilled": False,
            "pretrained_url_key": "augreg_in21k_ft_in1k",
        },
        "vit_small_patch16_384": {
            "image_size": 384,
            "patch_size": 16,
            "d_model": 384,
            "n_heads": 6,
            "n_layers": 12,
            "normalization": "vit",
            "distilled": False,
            "pretrained_url_key": "augreg_in21k_ft_in1k",
        },
        "vit_base_patch16_384": {
            "image_size": 384,
            "patch_size": 16,
            "d_model": 768,
            "n_heads": 12,
            "n_layers": 12,
            "normalization": "vit",
            "distilled": False,
        },
        "vit_large_patch16_384": {
            "image_size": 384,
            "patch_size": 16,
            "d_model": 1024,
            "n_heads": 16,
            "n_layers": 24,
            "normalization": "vit",
        },
        "vit_small_patch32_384": {
            "image_size": 384,
            "patch_size": 32,
            "d_model": 384,
            "n_heads": 6,
            "n_layers": 12,
            "normalization": "vit",
            "distilled": False,
        },
        "vit_base_patch32_384": {
            "image_size": 384,
            "patch_size": 32,
            "d_model": 768,
            "n_heads": 12,
            "n_layers": 12,
            "normalization": "vit",
        },
        "vit_large_patch32_384": {
            "image_size": 384,
            "patch_size": 32,
            "d_model": 1024,
            "n_heads": 16,
            "n_layers": 24,
            "normalization": "vit",
        },
    },
    "decoder": {
        "linear": {},
        "deeplab_dec": {"encoder_layer": -1},
        "mask_transformer": {"drop_path_rate": 0.0, "dropout": 0.1, "n_layers": 2},
    },
    "dataset": {
        "ade20k": {
            "epochs": 64,
            "eval_freq": 2,
            "batch_size": 8,
            "learning_rate": 0.001,
            "im_size": 512,
            "crop_size": 512,
            "window_size": 512,
            "window_stride": 512,
        },
        "pascal_context": {
            "epochs": 256,
            "eval_freq": 8,
            "batch_size": 16,
            "learning_rate": 0.001,
            "im_size": 520,
            "crop_size": 480,
            "window_size": 480,
            "window_stride": 320,
        },
        "cityscapes": {
            "epochs": 216,
            "eval_freq": 4,
            "batch_size": 8,
            "learning_rate": 0.01,
            "im_size": 1024,
            "crop_size": 768,
            "window_size": 768,
            "window_stride": 512,
        },
    },
}


class Segmenter(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
        pretrained_cpt_path,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.decoder = decoder

        if os.path.exists(pretrained_cpt_path):
            load_init(self, torch.load(pretrained_cpt_path)["model"])

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def forward(self, im):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        masks = self.decoder(x, (H, W))

        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        masks = unpadding(masks, (H_ori, W_ori))

        return masks

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)


class SegmImpl(Segmenter):
    """Segmenter base on `vit_tiny_patch16_384`"""

    def __init__(
        self,
        backbone="vit_tiny_patch16_384",
        decoder="mask_transformer",
        img_size=(512, 512),
        n_cls=19,
        dropout=0.0,
        drop_path=0.1,
        normalization=None,
        pretrained_cpt_path="models/pretrained/segm_t16_mask.pth",
    ):
        model_cfg = configs["model"][backbone]
        model_cfg["image_size"] = img_size
        model_cfg["backbone"] = backbone
        model_cfg["dropout"] = dropout
        model_cfg["drop_path_rate"] = drop_path
        model_cfg["normalization"] = normalization

        if "mask_transformer" in decoder:
            decoder_cfg = configs["decoder"]["mask_transformer"]
        else:
            decoder_cfg = configs["decoder"][decoder]

        decoder_cfg["name"] = decoder
        decoder_cfg["n_cls"] = n_cls

        encoder = create_vit(model_cfg)
        decoder = create_decoder(encoder, decoder_cfg)
        super().__init__(encoder, decoder, n_cls, pretrained_cpt_path)


class SegmT16Mask(SegmImpl):
    """Segmenter base on `vit_tiny_patch16_384`"""

    def __init__(
        self,
        backbone="vit_tiny_patch16_384",
        decoder="mask_transformer",
        img_size=(512, 512),
        n_cls=19,
        dropout=0.0,
        drop_path=0.1,
        normalization=None,
        pretrained_cpt_path="models/pretrained/segm_t16_mask.pth",
    ):
        super().__init__(
            backbone,
            decoder,
            img_size,
            n_cls,
            dropout,
            drop_path,
            normalization,
            pretrained_cpt_path,
        )


class SegmS16Mask(SegmImpl):
    """Segmenter base on `vit_small_patch16_384`"""

    def __init__(
        self,
        backbone="vit_small_patch16_384",
        decoder="mask_transformer",
        img_size=(512, 512),
        n_cls=19,
        dropout=0.0,
        drop_path=0.1,
        normalization=None,
        pretrained_cpt_path="models/pretrained/segm_s16_mask.pth",
    ):
        super().__init__(
            backbone,
            decoder,
            img_size,
            n_cls,
            dropout,
            drop_path,
            normalization,
            pretrained_cpt_path,
        )
