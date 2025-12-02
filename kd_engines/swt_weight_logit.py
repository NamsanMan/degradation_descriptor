"""
SWT-aware knowledge distillation engine.

The teacher provides high-frequency context via a Haar stationary wavelet
transform (SWT). Energy maps from the teacher guide where the student should
pay attention (boundary/edge-like zones) and how strongly to enforce
logit distillation.
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
        k_hh = torch.tensor([[0.25, -0.25], [-0.25,  0.25]])

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


class SWTLogitKD(BaseKDEngine):
    """
    SWT 기반 logit KD + attention-weighted CE 전용 엔진 (Feature KD 제거 버전).

    사용 가정:
      - w_ce_student = 1.0
      - w_kd_logit   = 0.2
      - w_kd_feat    = 0      → Feature KD 없음
      - teacher_stage = student_stage = 1 (동일 stage)
      - high_ce_scale = 1.0   → CE = ce_low + ce_high
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        w_ce_student: float,
        w_kd_logit: float,
        w_kd_feat: float,            # 설정에는 존재하지만, 실제로는 사용하지 않음 (0으로 가정)
        temperature: float,
        ignore_index: int,
        teacher_stage: int = 1,
        student_stage: int = 1,      # teacher_stage와 동일하다고 가정, 내부에서는 사용하지 않음
        energy_temperature: float = 1.5,
        freeze_teacher: bool = True,
        high_ce_scale: float = 1.0,  # 항상 1.0이라고 가정
    ) -> None:
        super().__init__(teacher, student)
        self.w_ce_student = float(w_ce_student)
        self.w_kd_logit = float(w_kd_logit)
        self.temperature = float(temperature)
        self.ignore_index = int(ignore_index)
        self.teacher_stage = int(teacher_stage)
        self.energy_temperature = float(energy_temperature)
        self._freeze_teacher = bool(freeze_teacher)

        # 기본 CE / KL
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.kl_loss = nn.KLDivLoss(reduction="none")

        if self._freeze_teacher:
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        if self._freeze_teacher:
            # teacher는 항상 eval 유지 (Dropout/DropPath 비활성)
            self.teacher.eval()
        return self

    def _forward_with_feats(
        self,
        model: nn.Module,
        imgs: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        out = model(imgs, return_feats=True)
        if isinstance(out, tuple):
            if len(out) == 2:
                logits, feats = out
            else:
                logits, feats = out[0], out[1:]
        elif isinstance(out, dict):
            logits, feats = out.get("logits"), out.get("feats")
        else:
            logits, feats = out, ()
        feats_tuple = (feats,) if isinstance(feats, torch.Tensor) else tuple(feats)
        return logits, feats_tuple

    @staticmethod
    def _select_feat(feats: Tuple[torch.Tensor, ...], index: int) -> torch.Tensor:
        if index < 0:
            index = len(feats) + index
        return feats[index]

    @staticmethod
    def _parse_inputs(imgs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        입력 형식:
          - (student_img, teacher_img) tuple/list
          - dict: { "student"/"lr", "teacher"/"hr" }
          - 단일 텐서: student/teacher 동일 입력
        """
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
        """
        Teacher feature → SWT → HF(LH/HL/HH) energy → normalized energy → softmax attention.
        """
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
        return energy.detach(), attn  # energy는 로그용, attn은 CE/KD에 사용

    def _weighted_logit_kd(
        self,
        s_logits: torch.Tensor,
        t_logits: torch.Tensor,
        masks: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        SWT 기반 attention(weight)을 이용한 logit KL-Divergence.
        """
        log_p_s = F.log_softmax(s_logits / self.temperature, dim=1)
        p_t = F.softmax(t_logits / self.temperature, dim=1)

        pixel_loss = self.kl_loss(log_p_s, p_t).sum(dim=1)  # (B,H,W)

        valid_mask = (masks != self.ignore_index).float().unsqueeze(1)  # (B,1,H,W)

        if weight.shape[-2:] != s_logits.shape[-2:]:
            weight = F.interpolate(
                weight,
                size=s_logits.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )

        weight = weight * valid_mask
        denom = weight.sum().clamp(min=1e-6)
        loss = (pixel_loss.unsqueeze(1) * weight).sum() / denom
        return loss * (self.temperature ** 2)

    def _ce_with_attention(
        self,
        s_logits: torch.Tensor,
        masks: torch.Tensor,
        energy_attn: torch.Tensor,
    ):
        """
        CE를 high-attention / low-attention으로 분리해서 계산.
        이 버전에서는 high_ce_scale = 1.0 가정 → 최종 CE = ce_low + ce_high.
        """
        # per-pixel CE: (B,H,W)
        ce_map = F.cross_entropy(
            s_logits,
            masks,
            ignore_index=self.ignore_index,
            reduction="none",
        )  # (B,H,W)

        valid_mask = (masks != self.ignore_index).float().unsqueeze(1)  # (B,1,H,W)

        # attention 해상도를 logits에 맞춤
        attn = energy_attn
        if attn.shape[-2:] != s_logits.shape[-2:]:
            attn = F.interpolate(
                attn,
                size=s_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # high / low attention mask (상위 20% vs 나머지)
        attn_flat = attn.view(attn.size(0), -1)
        q_high = torch.quantile(attn_flat, 0.8, dim=1, keepdim=True)  # (B,1)
        high_mask = (attn_flat >= q_high).view_as(attn)               # (B,1,H,W)
        low_mask = ~high_mask                                         # bool

        high_mask = high_mask.float() * valid_mask  # ignore_index 제외
        low_mask = low_mask.float() * valid_mask

        ce_map = ce_map.unsqueeze(1)  # (B,1,H,W)

        denom_low = low_mask.sum().clamp(min=1.0)
        denom_high = high_mask.sum().clamp(min=1.0)

        ce_low = (ce_map * low_mask).sum() / denom_low
        ce_high = (ce_map * high_mask).sum() / denom_high

        # high_ce_scale = 1.0 가정 → 단순 합
        ce_total = ce_low + ce_high

        return ce_total, ce_low.detach(), ce_high.detach()

    def compute_losses(self, imgs, masks, device):
        # 입력 파싱
        s_imgs, t_imgs = self._parse_inputs(imgs)

        # student forward
        s_logits, s_feats = self._forward_with_feats(self.student, s_imgs)

        # teacher forward
        if self._freeze_teacher:
            with torch.no_grad():
                t_logits, t_feats = self._forward_with_feats(self.teacher, t_imgs)
        else:
            t_logits, t_feats = self._forward_with_feats(self.teacher, t_imgs)

        t_feats = tuple(t_feats)

        # 1) teacher feature 선택
        t_feat_sel = self._select_feat(t_feats, self.teacher_stage)

        # 2) teacher feature에서 SWT energy / attention 추출
        raw_energy_t, energy_attn_t = self._energy_map(t_feat_sel)

        # 3) CE: high / low attention 분리 (teacher 기준 attention 사용)
        ce_student, ce_low, ce_high = self._ce_with_attention(
            s_logits,
            masks,
            energy_attn_t,
        )

        # 4) KD: teacher 기준 attention 사용
        kd_logit = self._weighted_logit_kd(s_logits, t_logits, masks, energy_attn_t)

        # 5) total loss (Feature KD 없음)
        total = (
            self.w_ce_student * ce_student
            + self.w_kd_logit * kd_logit
        )

        return {
            "total": total,
            "ce_student": ce_student.detach(),
            "ce_student_low": ce_low,
            "ce_student_high": ce_high,
            "kd_logit": kd_logit.detach(),
            "s_logits": s_logits.detach(),
            "t_logits": t_logits.detach(),
            "student_input": s_imgs.detach(),
            "teacher_input": t_imgs.detach(),

            # teacher 기준 SWT (TensorBoard용)
            "swt_energy": raw_energy_t.detach(),
            "swt_attention": energy_attn_t.detach(),
        }

    def get_extra_parameters(self):
        # Feature KD / projection이 없으므로 추가 학습 파라미터 없음
        return []
