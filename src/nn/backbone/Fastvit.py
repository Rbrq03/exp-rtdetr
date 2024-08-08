import torch
import torch.nn as nn
from timm.models import create_model
from .common import FrozenBatchNorm2d
from .fastvit.modules.mobileone import reparameterize_model
from ...core import register

build_fastvit_name = {"sa24": "fastvit_sa24", "ma36": "fastvit_ma36"}

pretrained_path = {
    "sa24": "/opt/data/private/hjn/fna-detection/exp/fastvit_sa24.pth.tar",
    "ma36": "/opt/data/private/hjn/fna-detection/exp/fastvit_ma36.pth.tar",
}


@register()
class FastViT(nn.Module):

    def __init__(
        self,
        name,
        freeze_at=-1,
        return_idx=[1, 2, 3],
        freeze_norm=True,
        pretrained=True,
    ):
        super().__init__()

        timm_model = create_model(build_fastvit_name[name])

        self.return_idx = return_idx

        if freeze_norm:
            self._freeze_norm(timm_model)

        if pretrained:
            checkpoint = torch.load(pretrained_path[name])
            timm_model.load_state_dict(checkpoint["state_dict"])
            print("load state dict")

        self.load_from_timm(timm_model)

        if freeze_at >= 0:
            self._freeze_norm(self.patch_embed)

    def load_from_timm(self, timm_model):

        self.patch_embed = timm_model.patch_embed

        self.stages = nn.ModuleList(
            [
                timm_model.network[0],
                nn.Sequential(*timm_model.network[1:3]),
                nn.Sequential(*timm_model.network[3:5]),
                nn.Sequential(*timm_model.network[5:8]),
            ]
        )

    def forward(self, x):
        output = []
        x = self.patch_embed(x)
        for i in range(4):
            x = self.stages[i](x)

            if i in self.return_idx:
                output.append(x)

        return output

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False
