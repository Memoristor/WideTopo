# coding=utf-8

from typing import Optional

from torch import nn
from torch.nn import functional as F
from transformers import SegformerForSemanticSegmentation

__all__ = ["SegformerB0", "SegformerB1", "SegformerB5"]


class Segformer(nn.Module):
    def __init__(
        self,
        num_classes,
        pretrained_model,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        super().__init__()

        net = SegformerForSemanticSegmentation.from_pretrained(pretrained_model)
        net.decode_head.classifier = nn.Conv2d(
            net.config.decoder_hidden_size, num_classes, kernel_size=1
        )
        self.net = net
        self.output_hidden_states = output_hidden_states
        self.return_dict = return_dict

    def forward(self, x):
        out = self.net(
            pixel_values=x,
            output_hidden_states=self.output_hidden_states,
            return_dict=self.return_dict,
        )

        logits = F.interpolate(out.logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        hidden_states = out.hidden_states

        if hidden_states is not None:
            return logits, *hidden_states
        else:
            return logits


class SegformerB0(Segformer):
    def __init__(
        self,
        num_classes,
        pretrained_model="nvidia/segformer-b0-finetuned-ade-512-512",
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        super().__init__(
            num_classes=num_classes,
            pretrained_model=pretrained_model,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class SegformerB1(Segformer):
    def __init__(
        self,
        num_classes,
        pretrained_model="nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        super().__init__(
            num_classes=num_classes,
            pretrained_model=pretrained_model,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class SegformerB5(Segformer):
    def __init__(
        self,
        num_classes,
        pretrained_model="nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        super().__init__(
            num_classes=num_classes,
            pretrained_model=pretrained_model,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
