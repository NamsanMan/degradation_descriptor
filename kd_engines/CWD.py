from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from .base_engine import BaseKDEngine


def _split_student_teacher_imgs(imgs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dataloader can return:
      - imgs: Tensor (student only)
      - imgs: (student_img, teacher_img)
    """
    if isinstance(imgs, (tuple, list)) and len(imgs) == 2:
        if torch.is_tensor(imgs[0]) and torch.is_tensor(imgs[1]):
            return imgs[0], imgs[1]
    if torch.is_tensor(imgs):
        return imgs, imgs
    raise TypeError(f"Unsupported imgs type: {type(imgs)}")


class ChannelNorm(nn.Module):
    """
    Same as original snippet:
      (N,C,H,W) -> reshape (N,C,H*W) -> softmax over last dim (spatial)
    """

    def forward(self, featmap: torch.Tensor) -> torch.Tensor:
        n, c, h, w = featmap.shape
        featmap = featmap.reshape(n, c, -1)
        featmap = featmap.softmax(dim=-1)
        return featmap


class CriterionCWD(nn.Module):
    """
    Channel-wise Distillation (CWD)
    - norm_type: {'channel','spatial','channel_mean','none'}
    - divergence: {'mse','kl'}
    """

    def __init__(self, norm_type: str = "channel", divergence: str = "kl", temperature: float = 1.0):
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

        self.temperature = 1.0
        if divergence == "mse":
            self.criterion = nn.MSELoss(reduction="sum")
        elif divergence == "kl":
            self.criterion = nn.KLDivLoss(reduction="sum")
            self.temperature = float(temperature)
        else:
            raise ValueError(f"Unsupported divergence: {divergence}")
        self.divergence = divergence

    def forward(self, preds_S: torch.Tensor, preds_T: torch.Tensor) -> torch.Tensor:
        n, c, h, w = preds_S.shape

        if self.normalize is not None:
            norm_s = self.normalize(preds_S / self.temperature)
            norm_t = self.normalize(preds_T.detach() / self.temperature)
        else:
            # keep original odd behavior (for strict similarity)
            norm_s = preds_S[0]
            norm_t = preds_T[0].detach()

        if self.divergence == "kl":
            norm_s = norm_s.log()

        loss = self.criterion(norm_s, norm_t)

        if self.norm_type in ("channel", "channel_mean"):
            loss /= (n * c)
        else:
            loss /= (n * h * w)

        return loss * (self.temperature ** 2)


def _make_bn(num_channels: int) -> nn.Module:
    """
    원본은 InPlaceABNSync였으나, 해당 확장/Autograd가 없으면 재현 불가.
    - 분산 학습(SyncBN)이 활성인 경우: SyncBatchNorm
    - 그 외: BatchNorm2d
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return nn.SyncBatchNorm(num_channels)
    return nn.BatchNorm2d(num_channels)


class ConvFeaturesProj(nn.Module):
    """
    Original idea:
      student last-stage feature (C_in=256) -> 1x1 conv -> BN -> ReLU -> (C_out=512)
    """

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False),
            _make_bn(out_c),
            nn.ReLU(inplace=False),
        )

        # Kaiming init (원본 InPlaceABN 환경과 최대한 유사하게)
        nn.init.kaiming_uniform_(self.proj[0].weight, a=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


@dataclass
class CWDParams:
    # loss weights
    w_ce_student: float = 1.0
    lambda_cwd: float = 1.0

    # CWD options
    norm_type: str = "channel"  # {'channel','spatial','channel_mean','none'}
    divergence: str = "kl"  # {'kl','mse'}
    temperature: float = 1.0

    # feature stage: always last stage (원본 코드가 last만 씀)
    ignore_index: int = config.DATA.IGNORE_INDEX

    # channels: SegFormer MiT-B0 last=256, MiT-B2/B3 last=512 (원본 가정)
    in_channels_last: int = 256
    out_channels_last: int = 512


class CWDEngine(BaseKDEngine):
    """
    Channel-Wise Distillation engine (feature-only; no logit KD).

    Loss:
      L = w_ce_student * CE(student_logits, GT) + lambda_cwd * CWD(proj(student_last_feat), teacher_last_feat)
    """

    def __init__(self, teacher: nn.Module, student: nn.Module, **kwargs):
        super().__init__(teacher, student)

        p = {}
        try:
            p = config.KD.ENGINE_PARAMS or {}
        except Exception:
            p = {}

        self.params = CWDParams(
            w_ce_student=float(p.get("w_ce_student", 1.0)),
            lambda_cwd=float(p.get("lambda_cwd", 1.0)),
            norm_type=str(p.get("norm_type", "channel")),
            divergence=str(p.get("divergence", "kl")),
            temperature=float(p.get("temperature", 1.0)),
            ignore_index=int(getattr(config.DATA, "IGNORE_INDEX", 255)),
            in_channels_last=int(p.get("in_channels_last", 256)),
            out_channels_last=int(p.get("out_channels_last", 512)),
        )

        self.ce = nn.CrossEntropyLoss(ignore_index=self.params.ignore_index)
        self.cwd = CriterionCWD(
            norm_type=self.params.norm_type,
            divergence=self.params.divergence,
            temperature=self.params.temperature,
        )

        # projection head (learnable extra params)
        self.proj = ConvFeaturesProj(self.params.in_channels_last, self.params.out_channels_last)

    def get_extra_parameters(self) -> Iterable[nn.Parameter]:
        return self.proj.parameters()

    @torch.no_grad()
    def _forward_teacher(self, x_t: torch.Tensor):
        self.teacher.eval()
        # expected: (feats, logits, embeds)
        return self.teacher(x_t, is_feat=True)

    def _forward_student(self, x_s: torch.Tensor):
        # expected: (feats, logits, embeds)
        return self.student(x_s, is_feat=True)

    def compute_losses(self, imgs: Any, masks: torch.Tensor, device) -> Dict[str, Any]:
        x_s, x_t = _split_student_teacher_imgs(imgs)

        # student
        feats_s, s_logits, _ = self._forward_student(x_s)
        feat_s_last = feats_s[-1]  # (B, C_s, Hs, Ws)
        proj_s = self.proj(feat_s_last)  # (B, C_t, Hs, Ws)  where C_t=512

        # teacher
        with torch.no_grad():
            feats_t, t_logits, _ = self._forward_teacher(x_t)
            feat_t_last = feats_t[-1]  # (B, 512, Ht, Wt) in original TransKD

        # align spatial if needed
        if feat_t_last.shape[-2:] != proj_s.shape[-2:]:
            feat_t_last = F.interpolate(feat_t_last, size=proj_s.shape[-2:], mode="bilinear", align_corners=False)

        # losses
        ce_student = self.ce(s_logits, masks) * self.params.w_ce_student
        kd_feat = self.cwd(proj_s, feat_t_last) * self.params.lambda_cwd

        total = ce_student + kd_feat

        return {
            "total": total,
            "ce_student": ce_student.detach(),
            "kd_feat": kd_feat.detach(),
            # for your logging/visualization
            "s_logits": s_logits.detach(),
            "t_logits": t_logits.detach() if torch.is_tensor(t_logits) else None,
            "student_input": x_s.detach(),
            "teacher_input": x_t.detach(),
        }

    def set_epoch(self, epoch: int):
        return None