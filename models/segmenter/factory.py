# coding=utf-8


from timm.models.helpers import load_custom_pretrained, load_pretrained
from timm.models.registry import register_model
from timm.models.vision_transformer import _create_vision_transformer, default_cfgs

from .decoder import DecoderLinear, MaskTransformer
from .utils import checkpoint_filter_fn
from .vit import VisionTransformer


@register_model
def vit_base_patch8_384(pretrained=False, **kwargs):
    """ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch8_384",
        pretrained=pretrained,
        default_cfg=dict(
            url="",
            input_size=(3, 384, 384),
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            num_classes=1000,
        ),
        **model_kwargs,
    )
    return model


def create_vit(model_cfg):
    """create vit for segmentation encoder"""
    model_cfg = model_cfg.copy()

    backbone = model_cfg.pop("backbone")
    normalization = model_cfg.pop("normalization")

    model_cfg["n_cls"] = 1000

    mlp_expansion_ratio = 4
    model_cfg["d_ff"] = mlp_expansion_ratio * model_cfg["d_model"]

    pretrained_cfg = dict(
        pretrained=False,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.0,
        drop_block_rate=None,
    )

    if "pretrained_url_key" in model_cfg.keys() and backbone in default_cfgs:
        url_key = model_cfg.pop("pretrained_url_key")
        pretrained_cfg["pretrained"] = True
        pretrained_cfg["url"] = default_cfgs[backbone].cfgs[url_key].url

    # print(pretrained_cfg)

    model = VisionTransformer(**model_cfg)
    if "deit" in backbone:
        load_pretrained(model, pretrained_cfg, filter_fn=checkpoint_filter_fn)
    else:
        load_custom_pretrained(model, pretrained_cfg)

    return model


def create_decoder(encoder, decoder_cfg):
    """create segmentation decoder"""
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop("name")
    decoder_cfg["d_encoder"] = encoder.d_model
    decoder_cfg["patch_size"] = encoder.patch_size

    if "linear" in name:
        decoder = DecoderLinear(**decoder_cfg)
    elif name == "mask_transformer":
        dim = encoder.d_model
        n_heads = dim // 64
        decoder_cfg["n_heads"] = n_heads
        decoder_cfg["d_model"] = dim
        decoder_cfg["d_ff"] = 4 * dim
        decoder = MaskTransformer(**decoder_cfg)
    else:
        raise ValueError(f"Unknown decoder: {name}")
    return decoder
