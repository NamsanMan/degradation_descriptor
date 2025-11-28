"""
SWT-aware knowledge distillation engine.

The teacher provides high-frequency context via a Haar stationary wavelet
transform (SWT). Energy maps from the teacher guide where the student should
pay attention (boundary/edge-like zones) and how strongly to enforce
feature/logit distillation.
"""

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
        # groups=in_channels for per-channel filtering
        self.register_buffer("filters", filters.repeat(in_channels, 1, 1, 1))
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels but got {x.shape[1]}")
        x_pad = F.pad(x, (0, 1, 0, 1), mode="reflect")
        out = F.conv2d(x_pad, self.filters, stride=1, padding=0, groups=self.in_channels)
        # (B, 4*C, H, W) -> (B, C, 4, H, W)
        return out.view(x.shape[0], self.in_channels, 4, x.shape[2], x.shape[3])


class SWTTunedKDEngine(BaseKDEngine):
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        w_ce_student: float,
        w_kd_logit: float,
        w_kd_feat: float,
        temperature: float,
        ignore_index: int,
        teacher_stage: int = -2,
        student_stage: int = -2,
        energy_temperature: float = 1.5,
        freeze_teacher: bool = True,
    ) -> None:
        super().__init__(teacher, student)
        self.w_ce_student = float(w_ce_student)
        self.w_kd_logit = float(w_kd_logit)
        self.w_kd_feat = float(w_kd_feat)
        self.temperature = float(temperature)
        self.ignore_index = int(ignore_index)
        self.teacher_stage = int(teacher_stage)
        self.student_stage = int(student_stage)
        self.energy_temperature = float(energy_temperature)
        self._freeze_teacher = bool(freeze_teacher)

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.kl_loss = nn.KLDivLoss(reduction="none")

        # Optional 1x1 projections to align student feature channels to teacher.
        self.feat_adapters = nn.ModuleDict()

        if self._freeze_teacher:
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        if self._freeze_teacher:
            self.teacher.eval()
        return self

    def _forward_with_feats(self, model: nn.Module, imgs: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        out = model(imgs, return_feats=True)
        if isinstance(out, tuple):
            if len(out) == 2:
                logits, feats = out
            else:
                logits, feats = out[0], out[1:]
        elif isinstance(out, dict):
            logits, feats = out.get("logits"), out.get("feats")
        else:
            logits, feats = out

        feats_tuple = (feats,) if isinstance(feats, torch.Tensor) else tuple(feats)
        return logits, feats_tuple

    @staticmethod
    def _select_feat(feats: Tuple[torch.Tensor, ...], index: int) -> torch.Tensor:
        if index < 0:
            index = len(feats) + index
        return feats[index]

    @staticmethod
    def _parse_inputs(imgs) -> Tuple[torch.Tensor, torch.Tensor]:
        # Accept (student, teacher) tuple/dict, otherwise share the same tensor.
        if isinstance(imgs, (tuple, list)) and len(imgs) == 2:
            return imgs[0], imgs[1]
        if isinstance(imgs, dict):
            student_img = imgs.get("student", imgs.get("lr"))
            teacher_img = imgs.get("teacher", imgs.get("hr", student_img))
            if student_img is None:
                student_img = teacher_img
            if teacher_img is None:
                teacher_img = student_img
            return student_img, teacher_img
        return imgs, imgs

    def _energy_map(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # feat: (B, C, H, W)
        swt = HaarSWT(feat.shape[1]).to(feat.device, feat.dtype)
        swt_out = swt(feat)
        # components: 0=LL,1=LH,2=HL,3=HH
        lh = swt_out[:, :, 1]
        hl = swt_out[:, :, 2]
        hh = swt_out[:, :, 3]
        energy = (lh.abs() + hl.abs() + hh.abs()).mean(dim=1, keepdim=True)  # (B,1,H,W)
        norm_energy = energy / (energy.mean(dim=[1, 2, 3], keepdim=True) + 1e-6)
        attn = torch.softmax(norm_energy.flatten(2) * self.energy_temperature, dim=-1)
        attn = attn.view_as(norm_energy)
        return energy.detach(), attn

    def _maybe_project_student_feat(self, s_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        """Adapt student feature channels to match the teacher if needed."""
        if s_feat.shape[1] == t_feat.shape[1]:
            return s_feat

        key = f"s{self.student_stage}_t{self.teacher_stage}"
        if key not in self.feat_adapters:
            self.feat_adapters[key] = nn.Conv2d(
                in_channels=s_feat.shape[1],
                out_channels=t_feat.shape[1],
                kernel_size=1,
                bias=False,
            )
        adapter = self.feat_adapters[key].to(s_feat.device, dtype=s_feat.dtype)
        return adapter(s_feat)

    def _weighted_feat_loss(self, s_feat: torch.Tensor, t_feat: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if s_feat.shape[-2:] != t_feat.shape[-2:]:
            t_feat = F.interpolate(t_feat, size=s_feat.shape[-2:], mode="bilinear", align_corners=True)
        if weight.shape[-2:] != s_feat.shape[-2:]:
            weight = F.interpolate(weight, size=s_feat.shape[-2:], mode="bilinear", align_corners=True)
        diff = (s_feat - t_feat.detach()) ** 2
        weighted = diff * weight
        denom = weight.sum().clamp(min=1e-6)
        return weighted.sum() / denom

    def _weighted_logit_kd(self, s_logits: torch.Tensor, t_logits: torch.Tensor, masks: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        log_p_s = F.log_softmax(s_logits / self.temperature, dim=1)
        p_t = F.softmax(t_logits / self.temperature, dim=1)
        pixel_loss = self.kl_loss(log_p_s, p_t).sum(dim=1)  # (B,H,W)
        valid_mask = (masks != self.ignore_index).float().unsqueeze(1)
        if weight.shape[-2:] != s_logits.shape[-2:]:
            weight = F.interpolate(weight, size=s_logits.shape[-2:], mode="bilinear", align_corners=True)
        weight = weight * valid_mask
        denom = weight.sum().clamp(min=1e-6)
        loss = (pixel_loss.unsqueeze(1) * weight).sum() / denom
        return loss * (self.temperature ** 2)

    def compute_losses(self, imgs, masks, device):
        s_imgs, t_imgs = self._parse_inputs(imgs)
        s_logits, s_feats = self._forward_with_feats(self.student, s_imgs)

        if self._freeze_teacher:
            with torch.no_grad():
                t_logits, t_feats = self._forward_with_feats(self.teacher, t_imgs)
        else:
            t_logits, t_feats = self._forward_with_feats(self.teacher, t_imgs)

        s_feats = tuple(s_feats)
        t_feats = tuple(t_feats)

        ce_student = self.ce_loss(s_logits, masks)

        t_feat_sel = self._select_feat(t_feats, self.teacher_stage)
        s_feat_sel = self._select_feat(s_feats, self.student_stage)
        s_feat_sel = self._maybe_project_student_feat(s_feat_sel, t_feat_sel)
        raw_energy, energy_attn = self._energy_map(t_feat_sel)

        kd_logit = self._weighted_logit_kd(s_logits, t_logits, masks, energy_attn)
        kd_feat = self._weighted_feat_loss(s_feat_sel, t_feat_sel, energy_attn)

        total = self.w_ce_student * ce_student + self.w_kd_logit * kd_logit + self.w_kd_feat * kd_feat

        return {
            "total": total,
            "ce_student": ce_student.detach(),
            "kd_logit": kd_logit.detach(),
            "kd_feat": kd_feat.detach(),
            "s_logits": s_logits.detach(),
            "t_logits": t_logits.detach(),
            "student_input": s_imgs.detach(),
            "teacher_input": t_imgs.detach(),
            "swt_energy": raw_energy.detach(),
            "swt_attention": energy_attn.detach(),
        }

    def get_extra_parameters(self):
        return list(self.feat_adapters.parameters())