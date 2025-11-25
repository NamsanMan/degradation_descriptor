# kd_engines/hsam_hfa.py
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "get_heterogeneous_feature_align_model",
    "HFA_model",
    "HeterogeneousFeatureAlignLoss",
]


class RRB(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        mid = max(1, out_ch // 4)
        self.unify = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.residual = nn.Sequential(
            nn.Conv2d(out_ch, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_ch, kernel_size=3, padding=1, bias=False),
        )
        self.norm = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unify(x)
        res = self.residual(x)
        x = self.norm(x + res)
        return x


class SpatialSelfAttention(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.q = nn.Conv2d(ch, ch, 1, bias=False)
        self.k = nn.Conv2d(ch, ch, 1, bias=False)
        self.v = nn.Conv2d(ch, ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.q(x).view(b, c, -1).transpose(1, 2)  # (B,HW,C)
        k = self.k(x).view(b, c, -1)                  # (B,C,HW)
        v = self.v(x).view(b, c, -1).transpose(1, 2)  # (B,HW,C)
        attn = (q @ k) / (c ** 0.5)
        attn = attn.softmax(dim=-1)
        out = attn @ v
        return out.transpose(1, 2).view(b, c, h, w)


class HeterogeneousFeatureAlignLoss(nn.Module):
    def __init__(
        self,
        aligned_channels: int = 160,
        offset_scale: float = 2.0,
        align_corners: bool = True,
    ) -> None:
        super().__init__()
        self.aligned_channels = int(aligned_channels)
        self.offset_scale = float(offset_scale)
        self.align_corners = bool(align_corners)

        self.teacher_proj: Optional[nn.Conv2d] = None
        self.student_proj: Optional[nn.Conv2d] = None
        self.teacher_rrb: Optional[RRB] = None
        self.student_rrb: Optional[RRB] = None
        self.student_attn: Optional[SpatialSelfAttention] = None

        self.branch1: Optional[nn.Conv2d] = None
        self.branch2: Optional[nn.Conv2d] = None
        self.delta_gen: Optional[nn.Sequential] = None

    def _build_if_needed(self, teacher_c: int, student_c: int, device: torch.device) -> None:
        if self.teacher_proj is not None:
            return
        ch = self.aligned_channels
        self.teacher_proj = nn.Conv2d(teacher_c, ch, kernel_size=1, bias=False).to(device)
        self.student_proj = nn.Conv2d(student_c, ch, kernel_size=1, bias=False).to(device)
        self.teacher_rrb = RRB(ch, ch).to(device)
        self.student_rrb = RRB(ch, ch).to(device)
        self.student_attn = SpatialSelfAttention(ch).to(device)
        self.branch1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False).to(device)
        self.branch2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False).to(device)
        self.delta_gen = nn.Sequential(
            nn.Conv2d(2 * ch, ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 2, kernel_size=3, padding=1, bias=False),
        ).to(device)
        nn.init.zeros_(self.delta_gen[-1].weight)

    @staticmethod
    def _make_base_grid(n: int, h: int, w: int, device, dtype) -> torch.Tensor:
        yy = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xx = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        base = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
        return base.repeat(n, 1, 1, 1)

    def _bilinear_refactor(self, feat: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        n, _, h, w = feat.shape
        base = self._make_base_grid(n, h, w, feat.device, feat.dtype)
        dx, dy = offset[:, 0], offset[:, 1]
        nx = 2.0 * dx / max(w - 1, 1)
        ny = 2.0 * dy / max(h - 1, 1)
        grid = torch.empty_like(base)
        grid[..., 0] = base[..., 0] + nx
        grid[..., 1] = base[..., 1] + ny
        out = F.grid_sample(feat, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return out

    def forward(
        self,
        teacher_feat: torch.Tensor,
        student_feat: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,   # (B, Hs, Ws) or (B,1,Hs,Ws)
    ) -> torch.Tensor:
        device = student_feat.device
        self._build_if_needed(teacher_feat.shape[1], student_feat.shape[1], device)
        assert self.teacher_proj and self.student_proj
        assert self.teacher_rrb and self.student_rrb and self.student_attn
        assert self.branch1 and self.branch2 and self.delta_gen

        target_size = student_feat.shape[-2:]
        t_resized = F.interpolate(teacher_feat, size=target_size, mode="bilinear", align_corners=True)

        t_proj = self.teacher_proj(t_resized)
        s_proj = self.student_proj(student_feat)

        s_proj = self.student_attn(s_proj)

        rc_t = self.teacher_rrb(t_proj)
        rc_s = self.student_rrb(s_proj)

        h3 = rc_t * rc_s
        b1 = F.relu(self.branch1(h3), inplace=True)
        b2 = F.relu(self.branch2(h3), inplace=True)
        hc3 = torch.cat([b1, b2], dim=1)

        raw_offset = self.delta_gen(hc3)
        offset = torch.tanh(raw_offset) * self.offset_scale
        t_re = self._bilinear_refactor(rc_t, offset)

        # ---- void 무시: 유효 위치만 평균 MSE ----
        if valid_mask is None:
            return F.mse_loss(t_re, rc_s)

        if valid_mask.dim() == 3:
            valid_mask = valid_mask.unsqueeze(1)  # (B,1,Hs,Ws)
        valid_mask = valid_mask.to(dtype=rc_s.dtype)
        # student feature 해상도로 resize (nearest)
        valid_mask = F.interpolate(valid_mask, size=rc_s.shape[-2:], mode="nearest")
        se2 = (t_re - rc_s).pow(2).mean(dim=1, keepdim=True)  # (B,1,H,W)
        masked = se2 * valid_mask
        denom = valid_mask.sum().clamp(min=1.0)
        loss = masked.sum() / denom
        return loss


class HFA_model(HeterogeneousFeatureAlignLoss):
    def __init__(self, batchsize: Optional[int] = None, **kwargs) -> None:
        del batchsize
        super().__init__(**kwargs)


def get_heterogeneous_feature_align_model(batchsize: Optional[int] = None, **kwargs) -> HFA_model:
    model = HFA_model(batchsize=batchsize, **kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model
