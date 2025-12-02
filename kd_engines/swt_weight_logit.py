from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_engine import BaseKDEngine


class HaarSWT(nn.Module):
    """Lightweight 2×2 Haar SWT applied per-channel."""

    def __init__(self, in_channels: int):
        super().__init__()
        k_ll = torch.tensor([[0.25, 0.25], [0.25, 0.25]])
        k_lh = torch.tensor([[0.25, -0.25], [0.25, -0.25]])
        k_hl = torch.tensor([[0.25, 0.25], [-0.25, -0.25]])
        k_hh = torch.tensor([[0.25, -0.25], [-0.25, 0.25]])

        filters = torch.stack([k_ll, k_lh, k_hl, k_hh], dim=0).unsqueeze(1)
        self.register_buffer("filters", filters.repeat(in_channels, 1, 1, 1))
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 1, 0, 1), mode="reflect")
        out = F.conv2d(x, self.filters, stride=1, groups=self.in_channels)
        return out.view(x.shape[0], self.in_channels, 4, x.shape[2], x.shape[3])


class SWTLogitKD(BaseKDEngine):
    """
    Minimal SWT-based logit KD engine.

    - CE: standard CE (no attention)
    - KD: pixel-wise KL, weighted by teacher stage-1 SWT energy
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        w_ce: float = 1.0,
        w_kd: float = 0.2,
        temperature: float = 2.0,
        ignore_index: int = 255,
        teacher_stage: int = 1,
        energy_temperature: float = 1.5,
        freeze_teacher: bool = True,
    ):
        super().__init__(teacher, student)

        self.w_ce = float(w_ce)
        self.w_kd = float(w_kd)
        self.temperature = float(temperature)
        self.ignore_index = int(ignore_index)
        self.teacher_stage = int(teacher_stage)
        self.energy_temperature = float(energy_temperature)

        self.ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.kl = nn.KLDivLoss(reduction="none")

        if freeze_teacher:
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()

    def _forward_with_feats(
        self, model: nn.Module, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        logits, feats = model(x, return_feats=True)
        feats = (feats,) if isinstance(feats, torch.Tensor) else tuple(feats)
        return logits, feats

    def _energy_attention(self, feat: torch.Tensor) -> torch.Tensor:
        swt = HaarSWT(feat.shape[1]).to(feat.device, feat.dtype)
        lh, hl, hh = swt(feat)[:, :, 1:4].unbind(dim=2)

        energy = (lh.abs() + hl.abs() + hh.abs()).mean(dim=1, keepdim=True)
        energy = energy / (energy.mean(dim=[1, 2, 3], keepdim=True) + 1e-6)

        attn = torch.softmax(
            energy.flatten(2) * self.energy_temperature, dim=-1
        ).view_as(energy)

        return attn.detach()

    def compute_losses(self, imgs, masks, device):
        if isinstance(imgs, (tuple, list)):
            s_img, t_img = imgs
        else:
            s_img = t_img = imgs

        s_logits, _ = self._forward_with_feats(self.student, s_img)

        with torch.no_grad():
            t_logits, t_feats = self._forward_with_feats(self.teacher, t_img)
            t_feat = t_feats[self.teacher_stage]
            attn = self._energy_attention(t_feat)

        # ---- CE (standard) ----
        loss_ce = self.ce(s_logits, masks)

        # ---- logit KD (energy weighted) ----
        log_ps = F.log_softmax(s_logits / self.temperature, dim=1)
        p_t = F.softmax(t_logits / self.temperature, dim=1)

        kd_map = self.kl(log_ps, p_t).sum(dim=1)  # (B,H,W)

        if attn.shape[-2:] != kd_map.shape[-2:]:
            attn = F.interpolate(attn, kd_map.shape[-2:], mode="bilinear", align_corners=False)

        valid = (masks != self.ignore_index).float()
        weight = attn.squeeze(1) * valid

        loss_kd = (kd_map * weight).sum() / (weight.sum().clamp(min=1.0))
        loss_kd = loss_kd * (self.temperature ** 2)

        total = self.w_ce * loss_ce + self.w_kd * loss_kd

        return {
            "total": total,
            "ce": loss_ce.detach(),
            "kd_logit": loss_kd.detach(),
            "swt_attention": attn,
        }
