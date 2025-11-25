# kd_engines/glahfa_kd.py
"""KD engine that combines PSAM (GLA) and HSAM(HFA) losses."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_engine import BaseKDEngine
from .psam_align import PSAMAlign
from .hsam_hfa import HeterogeneousFeatureAlignLoss


class HMKD(BaseKDEngine):
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        w_ce_student: float,
        w_gla: float,
        w_hfa: float,
        ignore_index: int,
        gla_embed_dim: int = 64,
        gla_patch_size: int = 8,
        gla_stride: Optional[int] = None,
        gla_teacher_stage: int = 0,
        gla_student_stage: int = 0,
        hfa_aligned_channels: int = 160,
        hfa_offset_scale: float = 2.0,
        hfa_align_corners: bool = True,
        hfa_teacher_stage: int = -1,
        hfa_student_stage: int = -1,
        freeze_teacher: bool = True,
    ) -> None:
        super().__init__(teacher, student)

        self.w_ce_student = float(w_ce_student)
        self.w_gla = float(w_gla)
        self.w_hfa = float(w_hfa)
        self.ignore_index = int(ignore_index)
        self._freeze_teacher = bool(freeze_teacher)

        self.gla_embed_dim = int(gla_embed_dim)
        self.gla_patch_size = int(gla_patch_size)
        self.gla_stride = gla_stride
        self.gla_teacher_stage = int(gla_teacher_stage)
        self.gla_student_stage = int(gla_student_stage)

        self.hfa_aligned_channels = int(hfa_aligned_channels)
        self.hfa_offset_scale = float(hfa_offset_scale)
        self.hfa_align_corners = bool(hfa_align_corners)
        self.hfa_teacher_stage = int(hfa_teacher_stage)
        self.hfa_student_stage = int(hfa_student_stage)

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        self.gla_module: Optional[PSAMAlign] = None
        self.hfa_module: Optional[HeterogeneousFeatureAlignLoss] = None

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
        try:
            out = model(imgs, return_feats=True)
            if isinstance(out, tuple) and len(out) == 2:
                logits, feats = out
            elif isinstance(out, dict) and "logits" in out and "feats" in out:
                logits, feats = out["logits"], out["feats"]
            else:
                logits, feats = out  # type: ignore[misc]
            feats = (feats,) if isinstance(feats, torch.Tensor) else tuple(feats)
            return logits, feats
        except TypeError:
            pass

        if hasattr(model, "config"):
            if hasattr(model.config, "output_hidden_states") and not model.config.output_hidden_states:
                model.config.output_hidden_states = True
            outputs = model(imgs)
            logits = getattr(outputs, "logits", outputs[0] if isinstance(outputs, tuple) else None)
            feats = getattr(outputs, "encoder_hidden_states", None)
            if feats is None:
                feats = getattr(outputs, "hidden_states", None)
            if feats is None or len(feats) < 4:
                raise RuntimeError("Failed to obtain encoder features from teacher/student model.")
            feats = tuple(feats[-4:])
            if logits is None:
                raise RuntimeError("Model output does not contain logits.")
            return logits, feats

        raise RuntimeError("Model must support return_feats=True or expose hidden_states.")

    def _select_feat(self, feats: Sequence[torch.Tensor], index: int) -> torch.Tensor:
        if index < 0:
            index = len(feats) + index
        if not (0 <= index < len(feats)):
            raise IndexError(f"Stage index {index} is out of range for {len(feats)} features.")
        return feats[index]

    def _ensure_gla_module(self, device: torch.device) -> None:
        if self.gla_module is None:
            self.gla_module = PSAMAlign(
                embed_dim=self.gla_embed_dim,
                patch_size=self.gla_patch_size,
                stride=self.gla_stride,
            ).to(device)

    def _ensure_hfa_module(self, device: torch.device) -> None:
        if self.hfa_module is None:
            self.hfa_module = HeterogeneousFeatureAlignLoss(
                aligned_channels=self.hfa_aligned_channels,
                offset_scale=self.hfa_offset_scale,
                align_corners=self.hfa_align_corners,
            ).to(device)

    def compute_losses(self, imgs: torch.Tensor, masks: torch.Tensor, device: torch.device):
        s_logits, s_feats = self._forward_with_feats(self.student, imgs)

        if self._freeze_teacher:
            with torch.no_grad():
                t_logits, t_feats = self._forward_with_feats(self.teacher, imgs)
        else:
            t_logits, t_feats = self._forward_with_feats(self.teacher, imgs)

        s_feats = tuple(s_feats)
        t_feats = tuple(t_feats)

        # --- CE (void 무시: ignore_index) ---
        ce_student = self.ce_loss(s_logits, masks)

        # binary valid mask: True=유효, False=void
        valid_full = (masks != self.ignore_index).float()   # (B,H,W)

        # --- PSAM(GLA) ---
        gla_loss = s_logits.new_tensor(0.0)
        if self.w_gla > 0.0:
            s_stage = self._select_feat(s_feats, self.gla_student_stage)   # (B,Cs,Hs,Ws)
            t_stage = self._select_feat(t_feats, self.gla_teacher_stage).detach()
            self._ensure_gla_module(s_stage.device)
            assert self.gla_module is not None
            # 학생 스테이지 해상도에 맞춰 valid mask 리사이즈 (nearest)
            valid_s = F.interpolate(valid_full.unsqueeze(1), size=s_stage.shape[-2:], mode="nearest").squeeze(1)
            gla_loss = self.gla_module(t_stage, s_stage, valid_mask=valid_s)

        # --- HSAM(HFA) ---
        hfa_loss = s_logits.new_tensor(0.0)
        if self.w_hfa > 0.0:
            s_stage = self._select_feat(s_feats, self.hfa_student_stage)
            t_stage = self._select_feat(t_feats, self.hfa_teacher_stage).detach()
            self._ensure_hfa_module(s_stage.device)
            assert self.hfa_module is not None
            valid_s = F.interpolate(valid_full.unsqueeze(1), size=s_stage.shape[-2:], mode="nearest").squeeze(1)
            hfa_loss = self.hfa_module(t_stage, s_stage, valid_mask=valid_s)

        total = self.w_ce_student * ce_student + self.w_gla * gla_loss + self.w_hfa * hfa_loss

        return {
            "total": total,
            "ce_student": ce_student.detach(),
            "gla": gla_loss.detach(),
            "hfa": hfa_loss.detach(),
            "s_logits": s_logits,
        }

    def get_extra_parameters(self):
        params = []
        if self.gla_module is not None:
            params.extend(self.gla_module.student_parameters())
        if self.hfa_module is not None:
            params.extend(self.hfa_module.parameters())
        return params
