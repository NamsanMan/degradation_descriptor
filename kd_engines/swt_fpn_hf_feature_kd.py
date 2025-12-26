from __future__ import annotations

from typing import Tuple, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


from .base_engine import BaseKDEngine


class HaarSWT(nn.Module):
    """Lightweight 2×2 Haar SWT applied per-channel (groups=C). Output: (B,C,4,H,W)."""

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

        # reflect pad to keep (H,W)
        x_pad = F.pad(x, (0, 1, 0, 1), mode="reflect")
        out = F.conv2d(x_pad, self.filters, stride=1, padding=0, groups=self.in_channels)
        return out.view(x.shape[0], self.in_channels, 4, x.shape[2], x.shape[3])  # (B,C,4,H,W)


class SWTFPNHFFeatureKD(BaseKDEngine):
    """
    Selective High-Frequency Feature KD (SWT->FPN) for HR->LR semantic segmentation.

    - No logit KD.
    - Teacher logits are used ONLY for gating (boundary/confidence), not for KL.
    - Feature KD compares SWT-based HF energy maps (channel-agnostic) between teacher & student.

    Assumption:
      model(imgs, return_feats=True) -> (logits, feats_tuple) or dict {"logits","feats"}.
    imgs input formats:
      - tensor (same for student/teacher)
      - (student_img, teacher_img)
      - dict: {"student"/"lr": ..., "teacher"/"hr": ...}

    Output dict contains "total" and debug tensors for TensorBoard.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        w_ce_student: float = 1.0,
        w_hf_kd: float = 1.0,
        ignore_index: int = 255,
        freeze_teacher: bool = True,

        # multi-stage settings (teacher/student can use different indices)
        teacher_stages: Optional[List[int]] = None,     # default [0,1,2,3]
        student_stages: Optional[List[int]] = None,     # default same as teacher_stages
        stage_weights: Optional[List[float]] = None,    # default equal
        fpn_fuse: str = "sum",                          # "sum" or "prod"

        # HF construction
        hf_mode: str = "abs_mean",                      # "abs_mean" or "l2_mean"
        use_ll: bool = False,                           # False: HF only (LH/HL/HH)

        # gating (where to distill)
        gate_mode: str = "boundary_conf",               # "boundary", "conf", "boundary_conf", "boundary_conf_hf"
        conf_mode: str = "margin",                      # "margin" or "entropy"
        conf_temp: float = 1.0,                         # softmax temperature for confidence
        gate_pow: float = 1.0,                          # exponent to sharpen/soften gate
        gate_min: float = 0.0,                          # clamp lower bound
        gate_max: float = 1.0,                          # clamp upper bound

        # warmup for hf kd weight
        warmup_hf_epochs: int = 0,                      # 0 disable

        # numerics
        norm_eps: float = 1e-6,
        swt_zsigmoid_eps: float = 1e-6,

        # robust loss
        hf_loss: str = "l1",                            # "l1" or "huber"
        huber_delta: float = 1.0,
        # gradient-aligned HF loss
        w_hf_grad: float = 0.0,  # 0 disables
        hf_grad_mode: str = "sobel",  # "sobel" or "diff"
        huber_delta_grad: float = 1.0,
    ) -> None:
        super().__init__(teacher, student)

        self.w_ce_student = float(w_ce_student)
        self.w_hf_kd = float(w_hf_kd)
        self.ignore_index = int(ignore_index)

        self.norm_eps = float(norm_eps)
        self.swt_zsigmoid_eps = float(swt_zsigmoid_eps)

        self.hf_mode = str(hf_mode).lower().strip()
        if self.hf_mode not in ("abs_mean", "l2_mean"):
            raise ValueError("hf_mode must be 'abs_mean' or 'l2_mean'.")
        self.use_ll = bool(use_ll)

        self.gate_mode = str(gate_mode).lower().strip()
        if self.gate_mode not in ("boundary", "conf", "boundary_conf", "boundary_conf_hf"):
            raise ValueError("gate_mode must be one of: boundary, conf, boundary_conf, boundary_conf_hf")
        self.conf_mode = str(conf_mode).lower().strip()
        if self.conf_mode not in ("margin", "entropy"):
            raise ValueError("conf_mode must be 'margin' or 'entropy'.")
        self.conf_temp = float(conf_temp)
        self.gate_pow = float(gate_pow)
        self.gate_min = float(gate_min)
        self.gate_max = float(gate_max)

        self.warmup_hf_epochs = int(warmup_hf_epochs)
        self._current_epoch = 0

        fpn_fuse = str(fpn_fuse).lower().strip()
        if fpn_fuse not in ("sum", "prod"):
            raise ValueError("fpn_fuse must be 'sum' or 'prod'.")
        self.fpn_fuse = fpn_fuse

        if teacher_stages is None:
            teacher_stages = [0, 1, 2, 3]
        if student_stages is None:
            student_stages = list(teacher_stages)
        self.teacher_stages = [int(s) for s in teacher_stages]
        self.student_stages = [int(s) for s in student_stages]
        if len(self.teacher_stages) != len(self.student_stages):
            raise ValueError("teacher_stages and student_stages must have the same length.")

        if stage_weights is None:
            stage_weights = [1.0] * len(self.teacher_stages)
        if len(stage_weights) != len(self.teacher_stages):
            raise ValueError("stage_weights length must match stages length.")
        self.stage_weights = [float(w) for w in stage_weights]

        self._freeze_teacher = bool(freeze_teacher)
        if self._freeze_teacher:
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()

        self.hf_loss = str(hf_loss).lower().strip()
        if self.hf_loss not in ("l1", "huber"):
            raise ValueError("hf_loss must be 'l1' or 'huber'.")
        self.huber_delta = float(huber_delta)

        self.w_hf_grad = float(w_hf_grad)
        self.hf_grad_mode = str(hf_grad_mode).lower().strip()
        if self.hf_grad_mode not in ("sobel", "diff"):
            raise ValueError("hf_grad_mode must be 'sobel' or 'diff'.")
        self.huber_delta_grad = float(huber_delta_grad)

        self._swt_cache: Dict[tuple, HaarSWT] = {}

    def _spatial_grad(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B,1,H,W) -> gx, gy
        if self.hf_grad_mode == "diff":
            gx = x[..., :, 1:] - x[..., :, :-1]
            gy = x[..., 1:, :] - x[..., :-1, :]
            # pad back to H,W
            gx = F.pad(gx, (0, 1, 0, 0), mode="replicate")
            gy = F.pad(gy, (0, 0, 0, 1), mode="replicate")
            return gx, gy
        # sobel (same kernels as boundary)
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
        return gx, gy

    # optional hook
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
    # SWT / maps
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
        p = F.softmax(t_logits, dim=1)                 # (B,K,H,W)
        p1 = p.max(dim=1, keepdim=True).values         # (B,1,H,W)
        g = self._sobel_grad_mag(p1)                   # (B,1,H,W)
        return self._minmax_norm(g)

    @torch.no_grad()
    def _confidence_map(self, t_logits: torch.Tensor) -> torch.Tensor:
        # output: (B,1,H,W) in [0,1] after minmax
        t = max(self.conf_temp, self.norm_eps)
        p = F.softmax(t_logits / t, dim=1)

        if self.conf_mode == "margin":
            # top1 - top2
            top2 = torch.topk(p, k=2, dim=1).values  # (B,2,H,W)
            m = (top2[:, 0:1] - top2[:, 1:2]).clamp(min=0.0)
            return self._minmax_norm(m)
        else:
            # entropy (lower entropy = higher confidence)
            ent = -(p * (p.clamp(min=self.norm_eps)).log()).sum(dim=1, keepdim=True)  # (B,1,H,W)
            ent = self._minmax_norm(ent)
            conf = 1.0 - ent
            return conf.clamp(0.0, 1.0)

    def _hf_energy(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          raw_hf: (B,1,h,w) before sigmoid-z
          imp_hf: (B,1,h,w) in (0,1) after sigmoid(z-score)
        """
        swt = self._get_swt(feat.shape[1], feat.device, feat.dtype)
        swt_out = swt(feat)  # (B,C,4,h,w)
        ll = swt_out[:, :, 0]
        lh = swt_out[:, :, 1]
        hl = swt_out[:, :, 2]
        hh = swt_out[:, :, 3]

        if self.use_ll:
            bands = (ll, lh, hl, hh)
        else:
            bands = (lh, hl, hh)

        if self.hf_mode == "abs_mean":
            raw = torch.zeros_like(bands[0][:, :1])
            for b in bands:
                raw = raw + b.abs().mean(dim=1, keepdim=True)
            raw = raw / float(len(bands))
        else:  # "l2_mean"
            raw = torch.zeros_like(bands[0][:, :1])
            for b in bands:
                raw = raw + (b * b).mean(dim=1, keepdim=True)
            raw = torch.sqrt(raw / float(len(bands)) + self.norm_eps)

        # z-score -> sigmoid for stable (0,1)
        eps = self.swt_zsigmoid_eps
        mu = raw.mean(dim=[2, 3], keepdim=True)
        sd = raw.std(dim=[2, 3], keepdim=True).clamp(min=eps)
        z = (raw - mu) / sd
        imp = torch.sigmoid(z)
        return raw, imp

    def _fpn_fuse(
        self,
        stage_maps: List[torch.Tensor],
        stage_weights: List[float],
        out_hw: Tuple[int, int],
    ) -> torch.Tensor:
        xs = []
        for m, w in zip(stage_maps, stage_weights):
            x = m
            if x.shape[-2:] != out_hw:
                x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
            xs.append(x * float(w))

        if len(xs) == 0:
            raise RuntimeError("No stage maps to fuse.")

        if self.fpn_fuse == "sum":
            fused = torch.zeros_like(xs[0])
            for x in xs:
                fused = fused + x
        else:
            fused = torch.ones_like(xs[0])
            for x in xs:
                fused = fused * x.clamp(min=self.norm_eps)

        return self._minmax_norm(fused)  # keep only ONE normalization at the end

    def _ramp(self, warmup_epochs: int) -> float:
        if warmup_epochs <= 0:
            return 1.0
        e = max(0, int(self._current_epoch))
        return float(min(1.0, e / float(warmup_epochs)))

    def _masked_hf_loss(self, hf_s, hf_t, mask, valid) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        hf_s, hf_t: (B,1,H,W)
        mask: (B,1,H,W) in [0,1]
        valid: (B,1,H,W) {0,1}
        """
        if hf_t.shape[-2:] != hf_s.shape[-2:]:
            hf_t = F.interpolate(hf_t, size=hf_s.shape[-2:], mode="bilinear", align_corners=False)
        if mask.shape[-2:] != hf_s.shape[-2:]:
            mask = F.interpolate(mask, size=hf_s.shape[-2:], mode="bilinear", align_corners=False)
        if valid.shape[-2:] != hf_s.shape[-2:]:
            valid = F.interpolate(valid, size=hf_s.shape[-2:], mode="nearest")

        w = (mask.clamp(self.gate_min, self.gate_max) ** self.gate_pow) * valid
        denom = w.sum().clamp(min=1e-6)

        diff = (hf_s - hf_t)
        if self.hf_loss == "l1":
            per = diff.abs()
        else:
            # Huber / SmoothL1-like with delta
            d = self.huber_delta
            absd = diff.abs()
            per = torch.where(absd < d, 0.5 * (diff * diff) / d, absd - 0.5 * d)

        loss_mag = (per * w).sum() / denom

        # gradient-aligned term (optional)
        if self.w_hf_grad <= 0.0:
            return loss_mag, loss_mag.new_tensor(0.0)

        gsx, gsy = self._spatial_grad(hf_s)
        gtx, gty = self._spatial_grad(hf_t)
        gdiff = torch.cat([gsx - gtx, gsy - gty], dim=1)  # (B,2,H,W)
        # reuse weights (broadcast to 2ch)
        w2 = w.repeat(1, 2, 1, 1)
        denom2 = w2.sum().clamp(min=1e-6)

        if self.hf_loss == "l1":
            per_g = gdiff.abs()
        else:
            d = self.huber_delta_grad
            absd = gdiff.abs()
            per_g = torch.where(absd < d, 0.5 * (gdiff * gdiff) / d, absd - 0.5 * d)

        loss_grad = (per_g * w2).sum() / denom2
        return loss_mag, loss_grad

    # -----------------------------
    # main entry
    # -----------------------------
    def compute_losses(self, imgs, masks, device=None):
        s_imgs, t_imgs = self._parse_inputs(imgs)

        # student forward (need feats)
        s_logits, s_feats = self._forward_with_feats(self.student, s_imgs)

        # teacher forward
        if self._freeze_teacher:
            with torch.no_grad():
                t_logits, t_feats = self._forward_with_feats(self.teacher, t_imgs)
        else:
            t_logits, t_feats = self._forward_with_feats(self.teacher, t_imgs)

        out_hw = s_logits.shape[-2:]
        valid = (masks != self.ignore_index).float().unsqueeze(1)  # (B,1,H,W)

        # -----------------------------
        # Build teacher gates (no grad)
        # -----------------------------
        with torch.no_grad():
            boundary = self._logit_boundary_probgrad(t_logits)  # (B,1,Ht,Wt)
            conf = self._confidence_map(t_logits)               # (B,1,Ht,Wt)
            boundary = F.interpolate(boundary, size=out_hw, mode="bilinear", align_corners=False)
            conf = F.interpolate(conf, size=out_hw, mode="bilinear", align_corners=False)
            boundary = self._minmax_norm(boundary)
            conf = self._minmax_norm(conf)

        # -----------------------------
        # SWT HF maps per stage (teacher & student)
        # -----------------------------
        t_stage_raws: List[torch.Tensor] = []
        t_stage_imps: List[torch.Tensor] = []
        s_stage_raws: List[torch.Tensor] = []
        s_stage_imps: List[torch.Tensor] = []

        for t_stg, s_stg in zip(self.teacher_stages, self.student_stages):
            t_feat = self._select_feat(tuple(t_feats), t_stg)
            s_feat = self._select_feat(tuple(s_feats), s_stg)

            t_raw, t_imp = self._hf_energy(t_feat)  # (B,1,ht,wt)
            s_raw, s_imp = self._hf_energy(s_feat)  # (B,1,hs,ws)

            t_stage_raws.append(t_raw)
            t_stage_imps.append(t_imp)
            s_stage_raws.append(s_raw)
            s_stage_imps.append(s_imp)

        # FPN fuse (importance maps recommended; more stable than raw)
        t_hf_fpn = self._fpn_fuse(t_stage_imps, self.stage_weights, out_hw=out_hw)  # (B,1,H,W)
        s_hf_fpn = self._fpn_fuse(s_stage_imps, self.stage_weights, out_hw=out_hw)  # (B,1,H,W)

        # -----------------------------
        # Final distillation mask M (where to distill)
        # -----------------------------
        with torch.no_grad():
            if self.gate_mode == "boundary":
                M = boundary
            elif self.gate_mode == "conf":
                M = conf
            elif self.gate_mode == "boundary_conf":
                M0 = self._minmax_norm(boundary * conf)
                # soft top-k gating to avoid overly sparse mask
                q = 0.75  # keep top 15%
                thresh = torch.quantile(M0.flatten(1), q, dim=1, keepdim=True)
                thresh = thresh.view(-1, 1, 1, 1)
                M = (M0 / (thresh + self.norm_eps)).clamp(0.0, 1.0)
            else:  # "boundary_conf_hf"
                # teacher HF itself can act as structure prior; but still gated by boundary/conf
                M = self._minmax_norm(boundary * conf * t_hf_fpn)

            M = M.clamp(self.gate_min, self.gate_max)

        # -----------------------------
        # Losses
        # -----------------------------
        ce = F.cross_entropy(s_logits, masks, ignore_index=self.ignore_index)

        alpha_hf = self._ramp(self.warmup_hf_epochs)
        with torch.no_grad():
            #s_support = (s_hf_fpn > 0.05).float()
            #M_eff = M * s_support
            M_eff = M
        hf_mag, hf_grad = self._masked_hf_loss(s_hf_fpn, t_hf_fpn.detach(), M_eff, valid)
        hf_kd = (hf_mag + self.w_hf_grad * hf_grad) * alpha_hf

        total = self.w_ce_student * ce + self.w_hf_kd * hf_kd

        # -----------------------------
        # Stats / debug
        # -----------------------------
        with torch.no_grad():
            v = valid
            if v.shape[-2:] != out_hw:
                v = F.interpolate(v, size=out_hw, mode="nearest")
            v_sum = v.sum().clamp(min=1.0)

            def _masked_stats(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                xm = (x * v).sum() / v_sum
                x2 = ((x * x) * v).sum() / v_sum
                xs = torch.sqrt((x2 - xm * xm).clamp(min=0.0))
                x_min = x.masked_fill(v == 0, 1e9).amin()
                x_max = x.masked_fill(v == 0, -1e9).amax()
                return xm, xs, x_min, x_max

            m_mean, m_std, m_min, m_max = _masked_stats(M)
            t_mean, t_std, t_min, t_max = _masked_stats(t_hf_fpn)
            s_mean, s_std, s_min, s_max = _masked_stats(s_hf_fpn)

        out = {
            "total": total,

            # losses
            "ce_student": ce.detach(),
            "hf_kd": hf_kd.detach(),
            "hf_kd_mag": hf_mag.detach(),
            "hf_kd_grad": hf_grad.detach(),
            "warmup_alpha_hf": torch.tensor(float(alpha_hf), device=s_logits.device),
            "hf_weight_eff": torch.tensor(float(self.w_hf_kd * alpha_hf), device=s_logits.device),

            # inputs/logits (optional)
            "s_logits": s_logits.detach(),
            "t_logits": t_logits.detach(),
            "student_input": s_imgs.detach(),
            "teacher_input": t_imgs.detach(),

            # gates & maps (for TensorBoard images)
            "gate_boundary": boundary.detach(),
            "gate_conf": conf.detach(),
            "hf_mask": M.detach(),

            "t_hf_fpn": t_hf_fpn.detach(),
            "s_hf_fpn": s_hf_fpn.detach(),

            # stats
            "mask_mean": m_mean.detach(),
            "mask_std": m_std.detach(),
            "mask_min": m_min.detach(),
            "mask_max": m_max.detach(),

            "t_hf_mean": t_mean.detach(),
            "t_hf_std": t_std.detach(),
            "t_hf_min": t_min.detach(),
            "t_hf_max": t_max.detach(),

            "s_hf_mean": s_mean.detach(),
            "s_hf_std": s_std.detach(),
            "s_hf_min": s_min.detach(),
            "s_hf_max": s_max.detach(),

            # config echo
            "teacher_stages": str(self.teacher_stages),
            "student_stages": str(self.student_stages),
            "fpn_fuse": self.fpn_fuse,
            "gate_mode": self.gate_mode,
            "hf_mode": self.hf_mode,
            "use_ll": str(self.use_ll),
        }

        # per-stage debug keys (fixed names)
        for i, (t_stg, s_stg) in enumerate(zip(self.teacher_stages, self.student_stages)):
            out[f"t_swt_hf_raw_s{t_stg}"] = t_stage_raws[i].detach()
            out[f"t_swt_hf_imp_s{t_stg}"] = t_stage_imps[i].detach()
            out[f"s_swt_hf_raw_s{s_stg}"] = s_stage_raws[i].detach()
            out[f"s_swt_hf_imp_s{s_stg}"] = s_stage_imps[i].detach()
            out["hf_mask_eff"] = M_eff.detach()
            out["mask_eff_mean"] = (M_eff * valid).sum() / (valid.sum().clamp(min=1.0))

        return out

    def get_extra_parameters(self):
        return []
