"""
SWT FPN 2-source attention Logit KD engine.

2-source attention:
  (A) semantic boundary from teacher logits (prob-gradient of top1 prob)
  (B) localization prior from teacher encoder SWT HF energy (multi-stage FPN fusion)

Final attention:
  final_attention = normalize(logit_boundary) * normalize(swt_fpn_importance)

Used for:
  - attention-weighted student CE
  - attention-weighted logit KD (KL)

Warm-up (optional, engine-internal via set_epoch):
  - ramp KD weight from 0 -> w_kd_logit over warmup_kd_epochs
  - ramp attention strength (blend from base_map to final_attention) over warmup_attn_epochs

Debug outputs (TensorBoard-friendly):
  - logit_boundary
  - swt_energy_s{i}, swt_importance_s{i} (per stage)
  - swt_fpn_importance (fused)
  - final_attention, warmup_attention
  - attn_mean/std/min/max (valid pixels)
  - warmup_alpha_kd, warmup_alpha_attn, kd_weight_eff
"""

from __future__ import annotations

from typing import Tuple, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_engine import BaseKDEngine


class HaarSWT(nn.Module):
    """Lightweight 2×2 Haar SWT applied per-channel (groups=C)."""

    def __init__(self, in_channels: int):
        super().__init__()
        c = int(in_channels)
        if c <= 0:
            raise ValueError(f"in_channels must be positive, got {c}")

        k_ll = torch.tensor([[0.25, 0.25], [0.25, 0.25]])
        k_lh = torch.tensor([[0.25, -0.25], [0.25, -0.25]])
        k_hl = torch.tensor([[0.25, 0.25], [-0.25, -0.25]])
        k_hh = torch.tensor([[0.25, -0.25], [-0.25, 0.25]])

        filters = torch.stack([k_ll, k_lh, k_hl, k_hh], dim=0).unsqueeze(1)  # (4,1,2,2)
        self.register_buffer("filters", filters.repeat(c, 1, 1, 1))          # (4C,1,2,2)
        self.in_channels = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (B,C,H,W), got shape={tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels but got {x.shape[1]}")

        x_pad = F.pad(x, (0, 1, 0, 1), mode="reflect")
        out = F.conv2d(x_pad, self.filters, stride=1, padding=0, groups=self.in_channels)
        return out.view(x.shape[0], self.in_channels, 4, x.shape[2], x.shape[3])  # (B,C,4,H,W)


class SWTFPN2SourceAttnLogitKD(BaseKDEngine):
    """
    Multi-stage SWT(FPN-like) + logit-boundary 2-source attention KD engine.

    Assumption:
      model(imgs, return_feats=True) -> (logits, feats_tuple) or dict {"logits","feats"}.

    imgs input formats:
      - tensor (same for student/teacher)
      - (student_img, teacher_img)
      - dict: {"student"/"lr": ..., "teacher"/"hr": ...}
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        w_ce_student: float = 1.0,
        w_kd_logit: float = 0.2,
        temperature: float = 2.0,
        ignore_index: int = 255,
        freeze_teacher: bool = True,
        ce_energy_gamma: float = 2.0,

        # ---- multi-stage SWT FPN settings ----
        teacher_stages: Optional[List[int]] = None,           # default: [0,1,2,3]
        stage_weights: Optional[List[float]] = None,          # default: equal
        fpn_fuse: str = "sum",                                # "sum" or "prod"

        # ---- warm-up settings (engine-internal) ----
        warmup_kd_epochs: int = 0,                            # 0 = disable
        warmup_attn_epochs: int = 0,                          # 0 = disable
        warmup_attn_base: str = "logit",                      # "logit" or "swt"

        # ---- numerics ----
        swt_zsigmoid_eps: float = 1e-6,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__(teacher, student)

        self.w_ce_student = float(w_ce_student)
        self.w_kd_logit = float(w_kd_logit)
        self.temperature = float(temperature)
        self.ignore_index = int(ignore_index)
        self.ce_energy_gamma = float(ce_energy_gamma)

        self.swt_zsigmoid_eps = float(swt_zsigmoid_eps)
        self.norm_eps = float(norm_eps)

        self.kl_loss = nn.KLDivLoss(reduction="none")

        self._freeze_teacher = bool(freeze_teacher)
        if self._freeze_teacher:
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()

        # stages
        if teacher_stages is None:
            teacher_stages = [0, 1, 2, 3]
        self.teacher_stages = [int(s) for s in teacher_stages]

        if stage_weights is None:
            stage_weights = [1.0] * len(self.teacher_stages)
        if len(stage_weights) != len(self.teacher_stages):
            raise ValueError("stage_weights length must match teacher_stages length.")
        self.stage_weights = [float(w) for w in stage_weights]

        fpn_fuse = str(fpn_fuse).lower().strip()
        if fpn_fuse not in ("sum", "prod"):
            raise ValueError("fpn_fuse must be 'sum' or 'prod'.")
        self.fpn_fuse = fpn_fuse

        # warm-up state
        self.warmup_kd_epochs = int(warmup_kd_epochs)
        self.warmup_attn_epochs = int(warmup_attn_epochs)
        self.warmup_attn_base = str(warmup_attn_base).lower().strip()
        if self.warmup_attn_base not in ("logit", "swt"):
            raise ValueError("warmup_attn_base must be 'logit' or 'swt'.")
        self._current_epoch = 0

        # SWT cache by (C, device.type, device.index, dtype)
        self._swt_cache: Dict[tuple, HaarSWT] = {}

    # ---- optional hook called from train code (your train already calls this) ----
    def set_epoch(self, epoch: int) -> None:
        self._current_epoch = int(epoch)

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        if self._freeze_teacher:
            self.teacher.eval()
        return self

    # -----------------------------
    # IO helpers
    # -----------------------------
    @staticmethod
    def _parse_inputs(imgs) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(imgs, (tuple, list)) and len(imgs) == 2:
            return imgs[0], imgs[1]
        if isinstance(imgs, dict):
            s = imgs.get("student", imgs.get("lr"))
            t = imgs.get("teacher", imgs.get("hr", s))
            if s is None:
                s = t
            if t is None:
                t = s
            return s, t
        return imgs, imgs

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
            logits, feats = out, ()

        if logits is None:
            raise RuntimeError("Model forward did not return logits.")

        feats_tuple = (feats,) if isinstance(feats, torch.Tensor) else tuple(feats)
        return logits, feats_tuple

    @staticmethod
    def _select_feat(feats: Tuple[torch.Tensor, ...], index: int) -> torch.Tensor:
        if len(feats) == 0:
            raise RuntimeError("Feature tuple is empty. Ensure return_feats=True yields features.")
        if index < 0:
            index = len(feats) + index
        if index < 0 or index >= len(feats):
            raise IndexError(f"stage index {index} out of range (num_feats={len(feats)})")
        return feats[index]

    # -----------------------------
    # map builders
    # -----------------------------
    def _get_swt(self, c: int, device: torch.device, dtype: torch.dtype) -> HaarSWT:
        key = (int(c), device.type, int(device.index) if device.index is not None else -1, dtype)
        if key not in self._swt_cache:
            self._swt_cache[key] = HaarSWT(int(c)).to(device=device, dtype=dtype)
        return self._swt_cache[key]

    def _minmax_norm(self, x: torch.Tensor) -> torch.Tensor:
        eps = self.norm_eps
        xmin = x.amin(dim=[2, 3], keepdim=True)
        xmax = x.amax(dim=[2, 3], keepdim=True)
        denom = (xmax - xmin).clamp(min=eps)
        return (x - xmin) / denom

    def _sobel_grad_mag(self, x: torch.Tensor) -> torch.Tensor:
        eps = self.norm_eps
        kx = torch.tensor([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]], device=x.device, dtype=x.dtype) / 8.0
        ky = torch.tensor([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]], device=x.device, dtype=x.dtype) / 8.0
        kx = kx.view(1, 1, 3, 3)
        ky = ky.view(1, 1, 3, 3)
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + eps)

    @torch.no_grad()
    def _logit_boundary_probgrad(self, t_logits: torch.Tensor) -> torch.Tensor:
        p = F.softmax(t_logits, dim=1)                   # (B,K,H,W)
        p1 = p.max(dim=1, keepdim=True).values           # (B,1,H,W)
        g = self._sobel_grad_mag(p1)                     # (B,1,H,W)
        return self._minmax_norm(g)

    def _swt_importance(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        swt = self._get_swt(feat.shape[1], feat.device, feat.dtype)
        swt_out = swt(feat)  # (B,C,4,h,w)
        lh = swt_out[:, :, 1]
        hl = swt_out[:, :, 2]
        hh = swt_out[:, :, 3]
        raw = (lh.abs() + hl.abs() + hh.abs()).mean(dim=1, keepdim=True)  # (B,1,h,w)

        eps = self.swt_zsigmoid_eps
        mu = raw.mean(dim=[2, 3], keepdim=True)
        sd = raw.std(dim=[2, 3], keepdim=True).clamp(min=eps)
        z = (raw - mu) / sd
        imp = torch.sigmoid(z)  # (0,1)
        return raw, imp

    def _fpn_fuse_importance(
        self,
        stage_imps: List[torch.Tensor],
        stage_weights: List[float],
        out_hw: Tuple[int, int],
    ) -> torch.Tensor:
        imps_resized = []
        for imp, w in zip(stage_imps, stage_weights):
            x = imp
            if x.shape[-2:] != out_hw:
                x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
            x = self._minmax_norm(x)
            imps_resized.append(x * float(w))

        if len(imps_resized) == 0:
            raise RuntimeError("No stage importance maps to fuse.")

        if self.fpn_fuse == "sum":
            fused = torch.zeros_like(imps_resized[0])
            for x in imps_resized:
                fused = fused + x
        else:  # "prod"
            fused = torch.ones_like(imps_resized[0])
            for x in imps_resized:
                fused = fused * (x.clamp(min=self.norm_eps))

        fused = self._minmax_norm(fused)
        return fused

    def _combine_maps(self, logit_boundary: torch.Tensor, swt_fpn_imp: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        if logit_boundary.shape[-2:] != out_hw:
            logit_boundary = F.interpolate(logit_boundary, size=out_hw, mode="bilinear", align_corners=False)
        if swt_fpn_imp.shape[-2:] != out_hw:
            swt_fpn_imp = F.interpolate(swt_fpn_imp, size=out_hw, mode="bilinear", align_corners=False)

        b = self._minmax_norm(logit_boundary)
        e = self._minmax_norm(swt_fpn_imp)
        a = b * e
        a = self._minmax_norm(a)
        return a

    # -----------------------------
    # warm-up helpers
    # -----------------------------
    def _ramp(self, warmup_epochs: int) -> float:
        if warmup_epochs <= 0:
            return 1.0
        # epoch starts at 1 in your training loop
        e = max(0, int(self._current_epoch))
        return float(min(1.0, e / float(warmup_epochs)))

    # -----------------------------
    # losses
    # -----------------------------
    def _weighted_logit_kd(self, s_logits: torch.Tensor, t_logits: torch.Tensor, masks: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        t = self.temperature
        log_p_s = F.log_softmax(s_logits / t, dim=1)
        p_t = F.softmax(t_logits / t, dim=1)
        pixel_kl = self.kl_loss(log_p_s, p_t).sum(dim=1)  # (B,H,W)

        valid = (masks != self.ignore_index).float().unsqueeze(1)  # (B,1,H,W)

        if weight.shape[-2:] != s_logits.shape[-2:]:
            weight = F.interpolate(weight, size=s_logits.shape[-2:], mode="bilinear", align_corners=False)

        w = weight * valid
        denom = w.sum().clamp(min=1e-6)
        loss = (pixel_kl.unsqueeze(1) * w).sum() / denom
        return loss * (t ** 2)

    def _ce_with_attention(self, s_logits: torch.Tensor, masks: torch.Tensor, attn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ce_map = F.cross_entropy(
            s_logits,
            masks,
            ignore_index=self.ignore_index,
            reduction="none",
        )  # (B,H,W)

        valid = (masks != self.ignore_index).float().unsqueeze(1)  # (B,1,H,W)

        if attn.shape[-2:] != s_logits.shape[-2:]:
            attn = F.interpolate(attn, size=s_logits.shape[-2:], mode="bilinear", align_corners=False)

        attn_valid = attn * valid
        denom_valid = valid.sum(dim=[2, 3], keepdim=True).clamp(min=1.0)
        attn_mean = attn_valid.sum(dim=[2, 3], keepdim=True) / denom_valid
        attn_centered = attn - attn_mean

        weight = 1.0 + self.ce_energy_gamma * attn_centered
        weight = torch.clamp(weight, min=0.0)

        ce_map = ce_map.unsqueeze(1)  # (B,1,H,W)
        w = weight * valid
        denom = w.sum().clamp(min=1.0)
        ce_total = (ce_map * w).sum() / denom

        # logging: top20% vs rest
        attn_flat = attn.view(attn.size(0), -1)
        q_high = torch.quantile(attn_flat, 0.8, dim=1, keepdim=True).view(attn.size(0), 1, 1, 1)
        high = (attn >= q_high).float() * valid
        low = (1.0 - high) * valid

        ce_low = (ce_map * low).sum() / low.sum().clamp(min=1.0)
        ce_high = (ce_map * high).sum() / high.sum().clamp(min=1.0)

        return ce_total, ce_low.detach(), ce_high.detach()

    # -----------------------------
    # main entry
    # -----------------------------
    def compute_losses(self, imgs, masks, device=None):
        s_imgs, t_imgs = self._parse_inputs(imgs)

        s_logits, _ = self._forward_with_feats(self.student, s_imgs)

        if self._freeze_teacher:
            with torch.no_grad():
                t_logits, t_feats = self._forward_with_feats(self.teacher, t_imgs)
        else:
            t_logits, t_feats = self._forward_with_feats(self.teacher, t_imgs)

        t_feats = tuple(t_feats)

        out_hw = s_logits.shape[-2:]

        # (A) semantic boundary from logits
        with torch.no_grad():
            logit_boundary = self._logit_boundary_probgrad(t_logits)  # (B,1,Ht,Wt)

        # (B) multi-stage SWT importance
        stage_raws: List[torch.Tensor] = []
        stage_imps: List[torch.Tensor] = []
        for stg in self.teacher_stages:
            feat = self._select_feat(t_feats, stg)
            raw, imp = self._swt_importance(feat)  # (B,1,h,w)
            stage_raws.append(raw)
            stage_imps.append(imp)

        swt_fpn_importance = self._fpn_fuse_importance(stage_imps, self.stage_weights, out_hw=out_hw)  # (B,1,H,W)

        # final attention (target)
        final_attention = self._combine_maps(logit_boundary, swt_fpn_importance, out_hw=out_hw)  # (B,1,H,W)

        # ---- warm-up attention blend (optional) ----
        alpha_attn = self._ramp(self.warmup_attn_epochs)
        if self.warmup_attn_epochs > 0:
            if self.warmup_attn_base == "logit":
                base_map = self._minmax_norm(F.interpolate(logit_boundary, size=out_hw, mode="bilinear", align_corners=False))
            else:
                base_map = self._minmax_norm(swt_fpn_importance)
            warmup_attention = (1.0 - alpha_attn) * base_map + alpha_attn * final_attention
            warmup_attention = self._minmax_norm(warmup_attention)
        else:
            warmup_attention = final_attention

        # losses (use warmup_attention)
        ce_student, ce_low, ce_high = self._ce_with_attention(s_logits, masks, warmup_attention)

        kd_logit_raw = self._weighted_logit_kd(s_logits, t_logits, masks, warmup_attention)

        # ---- warm-up KD weight (optional) ----
        alpha_kd = self._ramp(self.warmup_kd_epochs)
        kd_weight_eff = self.w_kd_logit * alpha_kd
        kd_logit = kd_logit_raw * alpha_kd

        total = self.w_ce_student * ce_student + kd_weight_eff * (kd_logit_raw.detach() * 0.0) + self.w_ce_student * 0.0
        # NOTE:
        # total에 kd_logit을 바로 더하면 됩니다.
        # 그런데 kd_logit_raw를 재사용하는 과정에서 실수 방지를 위해 아래처럼 명확히 작성합니다.
        total = self.w_ce_student * ce_student + (self.w_kd_logit * alpha_kd) * kd_logit_raw

        # attention stats (valid pixels)
        with torch.no_grad():
            valid = (masks != self.ignore_index).float().unsqueeze(1)
            if valid.shape[-2:] != warmup_attention.shape[-2:]:
                valid = F.interpolate(valid, size=warmup_attention.shape[-2:], mode="nearest")

            v_sum = valid.sum().clamp(min=1.0)
            a_mean = (warmup_attention * valid).sum() / v_sum
            a2_mean = ((warmup_attention * warmup_attention) * valid).sum() / v_sum
            a_std = torch.sqrt((a2_mean - a_mean * a_mean).clamp(min=0.0))
            a_min = warmup_attention.masked_fill(valid == 0, 1e9).amin()
            a_max = warmup_attention.masked_fill(valid == 0, -1e9).amax()

        out = {
            "total": total,

            "ce_student": ce_student.detach(),
            "ce_student_low": ce_low,
            "ce_student_high": ce_high,
            "kd_logit": (kd_logit_raw.detach() * alpha_kd),

            "s_logits": s_logits.detach(),
            "t_logits": t_logits.detach(),
            "student_input": s_imgs.detach(),
            "teacher_input": t_imgs.detach(),

            "logit_boundary": logit_boundary.detach(),
            "swt_fpn_importance": swt_fpn_importance.detach(),
            "final_attention": final_attention.detach(),
            "warmup_attention": warmup_attention.detach(),

            "attn_mean": a_mean.detach(),
            "attn_std": a_std.detach(),
            "attn_min": a_min.detach(),
            "attn_max": a_max.detach(),

            # warm-up scalars
            "warmup_alpha_attn": torch.tensor(float(alpha_attn), device=s_logits.device),
            "warmup_alpha_kd": torch.tensor(float(alpha_kd), device=s_logits.device),
            "kd_weight_eff": torch.tensor(float(self.w_kd_logit * alpha_kd), device=s_logits.device),

            "teacher_stages": str(self.teacher_stages),
            "fpn_fuse": self.fpn_fuse,
        }

        # per-stage debug maps (fixed keys)
        for i, stg in enumerate(self.teacher_stages):
            out[f"swt_energy_s{stg}"] = stage_raws[i].detach()
            out[f"swt_importance_s{stg}"] = stage_imps[i].detach()

        return out

    def get_extra_parameters(self):
        return []
