from numpy import arange
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...core import register

from .efficientvit.models.nn import ConvLayer

from .efficientvit.models.efficientvit.backbone import (
    efficientvit_backbone_l2,
    efficientvit_backbone_l0,
    efficientvit_backbone_l1,
    efficientnat_backbone_l0,
    efficientnat_backbone_l1,
    efficientnat_backbone_l2,
)

build_evit_function = {
    "l2": efficientvit_backbone_l2,
    "l1": efficientvit_backbone_l1,
    "l0": efficientvit_backbone_l0,
}

build_enat_fuction = {
    "l2": efficientnat_backbone_l2,
    "l1": efficientnat_backbone_l1,
    "l0": efficientnat_backbone_l0,
}

pretrained_urls = {
    "l2": "/opt/data/private/hjn/fna-detection/RT-DETR/yolov10/l2-r384.pt",
    "l1": "/opt/data/private/hjn/fna-detection/exp/l1-r224.pt",
}


@register()
class EfficientViT(nn.Module):

    def __init__(self, name, freeze_at, return_idx, freeze_norm, pretrained, add=False):
        super().__init__()
        self.model = build_evit_function[name](
            return_idx=return_idx, freeze_norm=freeze_norm, freeze_at=freeze_at, add=add
        )

        self.add = add

        if self.add:
            self.channel_convs = []
            for i in range(len(return_idx)):
                conv_layer = ConvLayer(
                    in_channels=self.model.width_list[return_idx[i]],
                    out_channels=self.model.width_list[return_idx[0]],
                    kernel_size=1,
                    act_func=None,
                )
                setattr(self, f"channel_convs{i}", conv_layer)
            self.average_pool = nn.AvgPool2d(kernel_size=4, stride=4)

        if pretrained:
            state = torch.load(pretrained_urls[name])["state_dict"]
            new_state_dict = {}
            for key, value in state.items():
                if key.startswith("backbone."):
                    new_key = key[len("backbone.") :]
                    new_state_dict[new_key] = value
            self.model.load_state_dict(new_state_dict)
            print("Load EfficientViT Weights")

    def forward(self, x):
        if self.add == False:
            return self.model(x)
        else:
            output = self.model(x)
            final_stage_fea_shape = output[0].shape[2:]
            final_stage_fea = None
            for id, x in enumerate(output):
                conv_layer = getattr(self, f"channel_convs{id}")
                x = conv_layer(x)
                x = (
                    x
                    if id == 0
                    else F.interpolate(
                        x,
                        size=final_stage_fea_shape,
                        mode="bilinear",
                        align_corners=False,
                    )
                )
                final_stage_fea = x if final_stage_fea == None else final_stage_fea + x

            final_stage_fea = self.average_pool(final_stage_fea)
            output[-1] = final_stage_fea
            return output


@register()
class EfficientNAT(nn.Module):

    def __init__(
        self, name, freeze_at, return_idx, freeze_norm, pretrained, dilations=None
    ):
        super().__init__()

        dilation_inputs = []
        if dilations is not None:
            for dilation in dilations:
                if dilation == "None":
                    dilation_inputs.append(None)
                else:
                    dilation_inputs.append(dilation)

            dilations = dilation_inputs

        self.model = build_enat_fuction[name](
            return_idx=return_idx,
            freeze_norm=freeze_norm,
            freeze_at=freeze_at,
            dilations=dilations,
        )

        if pretrained:
            state = torch.load(pretrained_urls[name])["state_dict"]
            new_state_dict = {}
            for key, value in state.items():
                if key.startswith("backbone.") and (
                    not key.startswith("backbone.stages.4")
                ):
                    new_key = key[len("backbone.") :]
                    new_state_dict[new_key] = value
            self.model.load_state_dict(new_state_dict, strict=False)
            print("Load EfficientViT Weights to EfficientNAT")

    def forward(self, x):
        return self.model(x)
