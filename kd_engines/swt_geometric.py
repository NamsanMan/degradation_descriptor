"""
Prototype-based KD with SWT weighting.

This module distills student features toward teacher-derived class prototypes
while emphasizing high-frequency regions detected by Haar SWT energy maps.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_engine import BaseKDEngine


class HaarSWT(nn.Module):
    """Lightweight 2×2 Haar SWT applied per-channel with energy attention."""

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
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels but got {x.shape[1]}")
        x_pad = F.pad(x, (0, 1, 0, 1), mode="reflect")
        out = F.conv2d(x_pad, self.filters, stride=1, padding=0, groups=self.in_channels)
        return out.view(x.shape[0], self.in_channels, 4, x.shape[2], x.shape[3])

    def energy_attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns SWT energy and a soft attention map over spatial positions.
        """
        swt_out = self.forward(x)
        lh = swt_out[:, :, 1]
        hl = swt_out[:, :, 2]
        hh = swt_out[:, :, 3]

        energy = (lh.abs() + hl.abs() + hh.abs()).mean(dim=1, keepdim=True)
        norm_energy = energy / (energy.mean(dim=[1, 2, 3], keepdim=True) + 1e-6)
        attn = torch.softmax(norm_energy.flatten(2), dim=-1).view_as(norm_energy)
        return norm_energy, attn


class PrototypeKDEngine(nn.Module):
    """Class-wise Prototype KD with SWT weighting."""

    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        momentum: float = 0.999,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.momentum = float(momentum)
        self.ignore_index = int(ignore_index)

        self.register_buffer("prototypes", torch.zeros(num_classes, feat_dim))
        self.register_buffer("proto_initialized", torch.tensor(False))

    @torch.no_grad()
    def _update_prototypes(self, t_feat: torch.Tensor, mask: torch.Tensor):
        assert t_feat.dim() == 4, "Teacher feature must be 4D (B,C,H,W)."
        assert mask.dim() == 3, "Mask must be 3D (B,H,W)."

        B, C, H, W = t_feat.shape

        feat_flat = t_feat.permute(0, 2, 3, 1).reshape(-1, C)
        mask_flat = mask.reshape(-1)

        valid = mask_flat != self.ignore_index
        if not valid.any():
            return

        feat_flat = feat_flat[valid]
        cls_flat = mask_flat[valid].long()

        for cls_id in range(self.num_classes):
            cls_mask = cls_flat == cls_id
            if not cls_mask.any():
                continue

            cls_feats = feat_flat[cls_mask]
            cls_mean = cls_feats.mean(dim=0)

            if bool(self.proto_initialized):
                self.prototypes[cls_id].mul_(self.momentum).add_(cls_mean * (1.0 - self.momentum))
            else:
                self.prototypes[cls_id].copy_(cls_mean)

        if not bool(self.proto_initialized):
            self.proto_initialized[...] = True

        proto_norm = self.prototypes.norm(dim=-1, keepdim=True).clamp_(min=1e-6)
        self.prototypes.div_(proto_norm)

    def compute_loss(
        self,
        s_feat: torch.Tensor,
        t_feat: torch.Tensor,
        mask: torch.Tensor,
        swt_weight: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, C, H, W = s_feat.shape

        with torch.no_grad():
            self._update_prototypes(t_feat.detach(), mask)

        if not bool(self.proto_initialized):
            zero = s_feat.new_tensor(0.0)
            return {
                "loss": zero,
                "sim_map": s_feat.new_zeros((B, H, W)),
                "weighted_loss_map": s_feat.new_zeros((B, H, W)),
                "mean_sim": zero,
                "proto_norm": self.prototypes.norm(dim=-1).detach(),
                "proto_valid_mask": (self.prototypes.norm(dim=-1) > 0).detach(),
            }

        s_feat_perm = s_feat.permute(0, 2, 3, 1)
        s_feat_norm = F.normalize(s_feat_perm, p=2, dim=-1)

        safe_mask = mask.clone()
        safe_mask[mask == self.ignore_index] = 0

        proto_norm = self.prototypes.norm(dim=-1)
        proto_valid = proto_norm > 0

        target_protos = F.embedding(safe_mask.long(), self.prototypes)
        proto_valid_per_pixel = proto_valid[safe_mask.long()]

        sim_map = (s_feat_norm * target_protos).sum(dim=-1)
        raw_loss_map = 1.0 - sim_map

        valid_mask = (mask != self.ignore_index) & proto_valid_per_pixel
        valid_mask_f = valid_mask.float()

        if swt_weight.dim() != 4 or swt_weight.size(1) != 1:
            raise ValueError("swt_weight must be (B,1,H,W)")
        swt_map = swt_weight.squeeze(1)

        weighted_loss_map = raw_loss_map * valid_mask_f * swt_map

        denom = (valid_mask_f * swt_map).sum().clamp(min=1.0)
        final_loss = weighted_loss_map.sum() / denom

        denom_sim = valid_mask_f.sum().clamp(min=1.0)
        mean_sim = (sim_map * valid_mask_f).sum() / denom_sim

        return {
            "loss": final_loss,
            "sim_map": sim_map.detach(),
            "weighted_loss_map": weighted_loss_map.detach(),
            "mean_sim": mean_sim.detach(),
            "proto_norm": proto_norm.detach(),
            "proto_valid_mask": proto_valid.detach(),
        }


class SWTProtoKDEngine(BaseKDEngine):
    """SWT attention + CE + Prototype KD with warm-up."""

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        num_classes: int,
        feat_dim: int,
        teacher_stage: int = 1,
        w_ce: float = 1.0,
        w_proto: float = 0.1,
        ignore_index: int = 255,
        momentum: float = 0.999,
        proto_start_epoch: int = 3,
        freeze_teacher: bool = True,
    ) -> None:
        super().__init__(teacher, student)
        self.teacher_stage = teacher_stage
        self.w_ce = float(w_ce)
        self.w_proto = float(w_proto)
        self.ignore_index = int(ignore_index)
        self.proto_start_epoch = int(proto_start_epoch)
        self.current_epoch = 0
        self._freeze_teacher = bool(freeze_teacher)

        self.swt = HaarSWT(in_channels=feat_dim)
        self.proto_kd = PrototypeKDEngine(
            num_classes=num_classes,
            feat_dim=feat_dim,
            momentum=momentum,
            ignore_index=ignore_index,
        )

        # Project student/teacher features to a shared channel size before KD/SWT.
        self.student_proj = nn.LazyConv2d(out_channels=feat_dim, kernel_size=1)
        self.teacher_proj = nn.LazyConv2d(out_channels=feat_dim, kernel_size=1)

        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="mean")

        if self._freeze_teacher:
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        if self._freeze_teacher:
            self.teacher.eval()
        return self

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)

    @staticmethod
    def _forward_with_feats(model: nn.Module, imgs: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        out = model(imgs, return_feats=True)
        if isinstance(out, tuple):
            logits, feats = (out[0], out[1]) if len(out) == 2 else (out[0], out[1:])
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

    def _maybe_resize_mask(self, mask: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        if mask.shape[-2:] == target_hw:
            return mask
        return F.interpolate(mask.unsqueeze(1).float(), size=target_hw, mode="nearest").squeeze(1).long()

    def compute_losses(self, imgs, masks, device):
        # --- Support separate LR (student) and HR (teacher) inputs ---
        if isinstance(imgs, (list, tuple)) and len(imgs) >= 2:
            student_in, teacher_in = imgs[0], imgs[1]
        elif isinstance(imgs, dict):
            student_in = imgs.get("student", imgs.get("lr", imgs.get("image")))
            teacher_in = imgs.get("teacher", imgs.get("hr", student_in))
        else:
            student_in = teacher_in = imgs

        if student_in is None or teacher_in is None:
            raise ValueError("Both student and teacher inputs must be provided (or shared).")

        s_logits, s_feats = self._forward_with_feats(self.student, student_in)
        with torch.no_grad():
            t_logits, t_feats = self._forward_with_feats(self.teacher, teacher_in)

        # 1. Raw Feature 추출 (수정됨)
        raw_s_feat = self._select_feat(s_feats, self.teacher_stage)
        raw_t_feat = self._select_feat(t_feats, self.teacher_stage)

        # 2. Projector 통과 (Prototype 계산용 - 학습됨)
        s_feat_proj = self.student_proj(raw_s_feat)
        t_feat_proj = self.teacher_proj(raw_t_feat)

        masks_feat = self._maybe_resize_mask(masks, s_feat_proj.shape[-2:])
        loss_ce = self.ce_loss_fn(s_logits, masks)

        # 3. [핵심 수정] SWT 입력은 Projector를 거치지 않은 'Raw HR Feature' 사용
        #    (Raw Feature가 Projector 출력과 채널이 다를 수 있으므로 SWT 초기화 시 주의 필요)
        #    만약 차원이 다르다면, SWT용으로 별도의 고정된 1x1 conv를 쓰거나
        #    raw_t_feat 자체를 입력으로 써야 함. (보통 Channel-wise SWT라 상관없음)

        # 여기서 raw_t_feat를 넣습니다.
        swt_energy, swt_attn = self.swt.energy_attention(raw_t_feat)

        # 4. Prototype KD 계산 (Projected Feature 사용)
        proto_out = self.proto_kd.compute_loss(
            s_feat=s_feat_proj,
            t_feat=t_feat_proj,
            mask=masks_feat,
            swt_weight=swt_attn,  # 가이드는 Raw에서 옴
        )

        epoch_1_based = self.current_epoch + 1
        w_proto_eff = 0.0 if epoch_1_based < self.proto_start_epoch else self.w_proto

        total_loss = self.w_ce * loss_ce + w_proto_eff * proto_out["loss"]

        return {
            "total": total_loss,
            "ce": loss_ce,
            "proto": proto_out["loss"],
            "proto_mean_sim": proto_out["mean_sim"],
            "proto_norm_mean": proto_out["proto_norm"].mean(),
            "proto_valid_ratio": proto_out["proto_valid_mask"].float().mean(),
            "proto_w": torch.tensor(w_proto_eff, device=total_loss.device),
            "proto_norm_vec": proto_out["proto_norm"].detach(),
            "s_logits": s_logits.detach(),
            "t_logits": t_logits.detach(),
            "swt_energy": swt_energy.detach(),
            "swt_attention": swt_attn.detach(),
            "proto_sim_map": proto_out["sim_map"].detach().unsqueeze(1),
            "proto_weighted_loss_map": proto_out["weighted_loss_map"].detach().unsqueeze(1),
            "student_input": student_in.detach(),
            "teacher_input": teacher_in.detach(),
        }

    def get_extra_parameters(self):
        # 1x1 projectors should be optimized alongside the student.
        return list(self.student_proj.parameters()) + list(self.teacher_proj.parameters())