# kd_engines/transkd_ops.py

from __future__ import annotations
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def hcl(fstudent: List[torch.Tensor], fteacher: List[torch.Tensor]) -> torch.Tensor:
    """
    Hierarchical Context Loss (HCL)
    - 각 stage에서 MSE + multi-scale pooling (4,2,1) 조합.
    """
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n, c, h, w = fs.shape
        loss = F.mse_loss(fs, ft, reduction="mean")
        cnt = 1.0
        tot = 1.0
        for l in [4, 2, 1]:
            if l >= h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l, l))
            tmpft = F.adaptive_avg_pool2d(ft, (l, l))
            cnt /= 2.0
            loss = loss + F.mse_loss(tmpfs, tmpft, reduction="mean") * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all


class ChannelNorm(nn.Module):
    """
    Feature map (N,C,H,W) → (N,C,H*W) 후 spatial softmax.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, featmap: torch.Tensor) -> torch.Tensor:
        n, c, h, w = featmap.shape
        featmap = featmap.reshape(n, c, -1)
        featmap = featmap.softmax(dim=-1)
        return featmap


class CriterionCWD(nn.Module):
    """
    Channel-wise distillation (CWD).
    """

    def __init__(
        self,
        norm_type: str = "channel",
        divergence: str = "mse",
        temperature: float = 1.0,
    ) -> None:
        super().__init__()

        if norm_type == "channel":
            self.normalize = ChannelNorm()
        elif norm_type == "spatial":
            self.normalize = nn.Softmax(dim=1)
        elif norm_type == "channel_mean":
            self.normalize = lambda x: x.view(x.size(0), x.size(1), -1).mean(-1)
        else:
            self.normalize = None
        self.norm_type = norm_type

        self.temperature = float(temperature)

        if divergence == "mse":
            self.criterion: nn.Module = nn.MSELoss(reduction="sum")
        elif divergence == "kl":
            self.criterion = nn.KLDivLoss(reduction="sum")
        else:
            raise ValueError(f"Unsupported divergence: {divergence}")
        self.divergence = divergence

    def forward(self, preds_S: torch.Tensor, preds_T: torch.Tensor) -> torch.Tensor:
        """
        preds_S, preds_T: (N,C,H,W)
        """
        n, c, h, w = preds_S.shape
        if self.normalize is not None:
            norm_s = self.normalize(preds_S / self.temperature)
            norm_t = self.normalize(preds_T.detach() / self.temperature)
        else:
            norm_s = preds_S
            norm_t = preds_T.detach()

        if self.divergence == "kl":
            norm_s = norm_s.log()

        loss = self.criterion(norm_s, norm_t)

        if self.norm_type in ("channel", "channel_mean"):
            loss = loss / (n * c)
        else:
            loss = loss / (n * h * w)

        return loss * (self.temperature ** 2)


class EmbedChannelNorm(nn.Module):
    """
    Embedding (N,L,D) → 마지막 축에 대해 softmax.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        return embed.softmax(dim=-1)


class CriterionEmbedCWD(nn.Module):
    """
    Patch embedding에 대한 CWD.
    입력: (N,L,D)
    """

    def __init__(
        self,
        norm_type: str = "channel",
        divergence: str = "mse",
        temperature: float = 1.0,
    ) -> None:
        super().__init__()

        if norm_type == "channel":
            self.normalize = EmbedChannelNorm()
        elif norm_type == "spatial":
            self.normalize = nn.Softmax(dim=1)
        elif norm_type == "channel_mean":
            self.normalize = lambda x: x.view(x.size(0), x.size(1), -1).mean(-1)
        else:
            self.normalize = None
        self.norm_type = norm_type

        self.temperature = float(temperature)

        if divergence == "mse":
            self.criterion: nn.Module = nn.MSELoss(reduction="sum")
        elif divergence == "kl":
            self.criterion = nn.KLDivLoss(reduction="sum")
        else:
            raise ValueError(f"Unsupported divergence: {divergence}")
        self.divergence = divergence

    def forward(self, preds_S: torch.Tensor, preds_T: torch.Tensor) -> torch.Tensor:
        n, l, d = preds_S.shape
        if self.normalize is not None:
            norm_s = self.normalize(preds_S / self.temperature)
            norm_t = self.normalize(preds_T.detach() / self.temperature)
        else:
            norm_s = preds_S
            norm_t = preds_T.detach()

        if self.divergence == "kl":
            norm_s = norm_s.log()

        loss = self.criterion(norm_s, norm_t)

        if self.norm_type in ("channel", "channel_mean"):
            loss = loss / (n * l)
        else:
            loss = loss / (n * d)

        return loss * (self.temperature ** 2)
