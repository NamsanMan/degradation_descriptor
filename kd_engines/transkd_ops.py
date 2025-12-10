# kd_engines/transkd_ops.py

from __future__ import annotations
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script_if_tracing
def hcl(fstudent: List[torch.Tensor], fteacher: List[torch.Tensor]) -> torch.Tensor:
    """
    Hierarchical Context Loss (HCL)
    - TransKD에서 사용하는 multi-scale feature distillation loss.
    - 각 stage에서 MSE + pyramid downsample(M=4,2,1) 평균으로 계산.

    fstudent, fteacher: 길이 L의 list, 각 원소 shape = (B, C, H, W)
    """
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n, c, h, w = fs.shape
        loss = F.mse_loss(fs, ft, reduction="mean")
        cnt = 1.0
        tot = 1.0
        # 계층적 context: 4×4, 2×2, 1×1로 downsample해서 추가 MSE
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
    Feature map을 (N,C,H,W) → (N,C,H*W) 후 spatial softmax.
    TransKD의 CWD에서 'channel' norm에 사용.
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
    Channel-wise distillation loss (CWD) – TransKD에서 사용.

    - self.normalize로 teacher/student를 정규화 후
    - divergence(mse or kl) 로스 계산
    - norm_type: 'channel', 'spatial', 'channel_mean', 'none'
    """

    def __init__(
        self,
        norm_type: str = "channel",
        divergence: str = "mse",
        temperature: float = 1.0,
    ) -> None:
        super().__init__()

        # normalize 함수 정의
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

        # loss 함수 정의
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
            if self.norm_type == "channel_mean":
                norm_s = self.normalize(preds_S / self.temperature)
                norm_t = self.normalize(preds_T.detach() / self.temperature)
            else:
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
    Embedding 토큰(level)용 softmax 정규화.
    입력: (N, L, D) 또는 (N, C, L) 형태를 가정 (TransKD 스타일).
    여기서는 (N, L, D) 기준으로 구현.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        # (N, L, D) 가정. 마지막 축에 대해 softmax.
        embed = embed.softmax(dim=-1)
        return embed


class CriterionEmbedCWD(nn.Module):
    """
    Patch embedding에 대한 CWD.
    - norm_type='channel' -> EmbedChannelNorm
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
        """
        preds_S, preds_T: (N, L, D) 형태 가정.
        """
        n, l, d = preds_S.shape
        if self.normalize is not None:
            if self.norm_type == "channel_mean":
                norm_s = self.normalize(preds_S / self.temperature)
                norm_t = self.normalize(preds_T.detach() / self.temperature)
            else:
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


def hcl_with_weight(
    fstudent: List[torch.Tensor],
    fteacher: List[torch.Tensor],
    fea_w: float = 1.0,
) -> torch.Tensor:
    """
    필요하면 feature weight를 곱한 HCL (TransKD 코드의 변형).
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
        loss_all = loss_all + fea_w * loss
    return loss_all
