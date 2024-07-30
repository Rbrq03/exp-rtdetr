from gc import freeze
import torch
import torch.nn as nn
from ...core import register

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
    "l2": "/opt/data/private/hjn/fna-detection/RT-DETR/yolov10/l2-r384.pt"
}


@register()
class EfficientViT(nn.Module):

    def __init__(self, name, freeze_at, return_idx, freeze_norm, pretrained, add=False):
        super().__init__()
        self.model = build_evit_function[name](
            return_idx=return_idx, freeze_norm=freeze_norm, freeze_at=freeze_at, add=add
        )

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
        return self.model(x)


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
