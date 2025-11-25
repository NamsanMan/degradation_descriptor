# kd_engines/psam_align.py
from typing import Iterator, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["get_GLA_model", "GLA_model", "PSAMAlign"]


def _pair(v):
    return v if isinstance(v, tuple) else (v, v)


class PatchEmbed(nn.Module):
    """PATCH + PROJ (수식 (1))"""
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int, stride: Optional[int] = None) -> None:
        super().__init__()
        patch = _pair(patch_size)
        stride = _pair(patch_size if stride is None else stride)
        self.patch = patch
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch, stride=stride, padding=0, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

    def _pad_like(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        ph, pw = self.patch
        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad_like(x)
        x = self.proj(x)                                # (B, D, Hp, Wp)
        x = x.flatten(2).transpose(1, 2)                # (B, N, D)
        x = self.norm(x)
        return x

    def mask_to_tokens(self, mask: torch.Tensor) -> torch.Tensor:
        """
        spatial mask(B,1,H,W or H,W) -> token-valid mask(B,N) with SAME padding rule.
        A token is valid if any pixel in the patch is valid.
        """
        if mask.dim() == 3:  # (B,H,W)
            mask = mask.unsqueeze(1)
        elif mask.dim() == 2:  # (H,W)
            mask = mask.unsqueeze(0).unsqueeze(0)
        # replicate padding identical to features
        mask = self._pad_like(mask.float())
        # use max-pool equivalent via unfold-like conv: here avg_pool + >0 works too
        ph, pw = self.patch
        with torch.no_grad():
            pooled = F.avg_pool2d(mask, kernel_size=(ph, pw), stride=(ph, pw))
            valid_hw = pooled > 0.0                      # (B,1,Hp,Wp)
            valid = valid_hw.flatten(2).squeeze(1)       # (B,N)
        return valid


class QKAttentionMap(nn.Module):
    """linear → q,k;  softmax(qk^T) (수식 (2),(4))"""
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        q = self.q(tokens)                                # (B,N,D)
        k = self.k(tokens)                                # (B,N,D)
        attn = torch.bmm(q, k.transpose(1, 2))            # (B,N,N)
        attn = attn / (q.size(-1) ** 0.5)
        attn = self.softmax(attn)
        return attn


class PSAMAlign(nn.Module):
    """
    Patch-based Self-attention Alignment Module (PSAM)
    - valid_mask: (B,Hs,Ws) or (B,1,Hs,Ws). True=유효(=void 아님).
    """
    def __init__(self, embed_dim: int = 64, patch_size: int = 8, stride: Optional[int] = None) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.stride = stride if stride is not None else patch_size

        self.student_proj: Optional[PatchEmbed] = None
        self.teacher_proj: Optional[PatchEmbed] = None
        self.student_qk: Optional[QKAttentionMap] = None
        self.teacher_qk: Optional[QKAttentionMap] = None

    def _build_if_needed(self, c_s: int, c_t: int, device: torch.device) -> None:
        if self.student_proj is not None:
            return
        D, P, S = self.embed_dim, self.patch_size, self.stride
        self.student_proj = PatchEmbed(c_s, D, P, S).to(device)
        self.teacher_proj = PatchEmbed(c_t, D, P, S).to(device)
        self.teacher_proj.requires_grad_(False)
        self.student_qk = QKAttentionMap(D).to(device)
        self.teacher_qk = QKAttentionMap(D).to(device)
        self.teacher_qk.requires_grad_(False)

    @torch.no_grad()
    def _mask_pairs(self, proj: PatchEmbed, valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """token valid vector(B,N)-> pair mask(B,N,N) & count"""
        token_valid = proj.mask_to_tokens(valid_mask)            # (B,N) bool
        pair_mask = token_valid.unsqueeze(2) & token_valid.unsqueeze(1)  # (B,N,N)
        valid_pairs = pair_mask.sum(dim=(1, 2)).clamp(min=1)     # (B,)
        return pair_mask, valid_pairs

    def forward(
        self,
        teacher_feat: torch.Tensor,
        student_feat: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = student_feat.device
        self._build_if_needed(student_feat.shape[1], teacher_feat.shape[1], device)
        assert self.student_proj and self.teacher_proj and self.student_qk and self.teacher_qk

        # teacher 해상도 → student 해상도로 정렬
        teacher_feat = F.interpolate(teacher_feat, size=student_feat.shape[-2:], mode="bilinear", align_corners=False)

        s_tokens = self.student_proj(student_feat)            # (B,N,D)
        with torch.no_grad():
            t_tokens = self.teacher_proj(teacher_feat)        # (B,N,D)

        sa_s = self.student_qk(s_tokens)                      # (B,N,N)
        with torch.no_grad():
            sa_t = self.teacher_qk(t_tokens)                  # (B,N,N)

        # ---- 스케일 업을 위한 N (토큰 수) ----
        N = sa_s.size(1)

        # valid_mask가 없는 경우: sum()/N
        if valid_mask is None:
            # sum over all pairs and batches, then divide by N (≈ mean * N)
            loss = F.mse_loss(sa_s, sa_t, reduction="sum") / N
            return loss

        # valid_mask가 있는 경우: 유효 pair만 sum()/N
        pair_mask, valid_pairs = self._mask_pairs(self.student_proj, valid_mask)  # (B,N,N), (B,)
        diff2 = (sa_s - sa_t).pow(2) * pair_mask.to(sa_s.dtype)  # (B,N,N)

        # 배치별: sum over pairs -> /N  (valid_pairs로 나누지 않고, 스케일만 N으로 통일)
        #   - sum()/N 스케일을 유지하려면 valid 쌍의 개수로 정규화하지 않고 N으로 고정 스케일을 적용
        #   - 만약 mask 비율 차이를 상쇄하고 싶으면 아래 주석처럼 valid_pairs로 나누는 버전을 사용
        loss_b = diff2.sum(dim=(1, 2)) / N

        # ※ 대안(마스크 비율 보정): loss_b = diff2.sum(dim=(1,2)) / valid_pairs.clamp(min=1)

        return loss_b.mean()

    # 학생 파라미터만 노출
    def student_parameters(self, recurse: bool = True):
        for m in (self.student_proj, self.student_qk):
            if m is not None:
                yield from m.parameters(recurse=recurse)

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        yield from self.student_parameters(recurse=recurse)


class GLA_model(PSAMAlign):
    def __init__(self, batchsize: Optional[int] = None, **kwargs) -> None:
        del batchsize
        super().__init__(**kwargs)


def get_GLA_model(batchsize: Optional[int] = None, **kwargs) -> GLA_model:
        model = GLA_model(batchsize=batchsize, **kwargs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model
