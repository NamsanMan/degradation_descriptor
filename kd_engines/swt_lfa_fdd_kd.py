"""
SWT + Learnable Frequency Attention + Frequency-Domain Distillation KD Engine.

- Teacher feature에 Haar SWT를 적용해 LF/HF subband를 얻고,
  Learnable Frequency Attention (LFA)로 어떤 HF subband(LH/HL/HH)가 중요한지 학습.
- 이 attention으로부터 energy/attention map을 만들고,
  CE / logit-KD / feature-KD를 spatial하게 가중.
- 동시에 teacher/student feature에 대해 SWT를 적용하여
  LL(저주파) / LH,HL,HH(고주파) 차이를 분리해서 Frequency-Domain Distillation(FDD) 수행.
- Student feature는 단순 1x1 conv 대신, inverted residual 기반 Feature Adaptation Module(FAM)으로
  teacher feature space로 mapping.

주의:
- teacher, student 모델은 `forward(x, return_feats=True)` 인터페이스를 지원해야 함.
- `teacher_stage` / `student_stage`는 feature tuple index (음수 가능, -1=마지막).
"""

from __future__ import annotations

from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_engine import BaseKDEngine


# ------------------------------------------------------------
# 2x2 Haar SWT (stationary, stride 1) per-channel
# ------------------------------------------------------------
class HaarSWT(nn.Module):
    """Lightweight 2×2 Haar SWT applied per-channel.

    입력:  (B, C, H, W)
    출력:  (B, C, 4, H, W)  # 순서: 0=LL, 1=LH, 2=HL, 3=HH
    """

    def __init__(self, in_channels: int):
        super().__init__()
        k_ll = torch.tensor([[0.25, 0.25], [0.25, 0.25]])
        k_lh = torch.tensor([[0.25, -0.25], [0.25, -0.25]])
        k_hl = torch.tensor([[0.25, 0.25], [-0.25, -0.25]])
        k_hh = torch.tensor([[0.25, -0.25], [-0.25, 0.25]])

        filters = torch.stack([k_ll, k_lh, k_hl, k_hh], dim=0).unsqueeze(1)
        # groups = in_channels for per-channel filtering
        self.register_buffer("filters", filters.repeat(in_channels, 1, 1, 1))
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        if x.shape[1] != self.in_channels:
            raise ValueError(f"HaarSWT expected {self.in_channels} channels but got {x.shape[1]}")
        # reflect padding to avoid border artifacts (SWT-like)
        x_pad = F.pad(x, (0, 1, 0, 1), mode="reflect")
        out = F.conv2d(x_pad, self.filters, stride=1, padding=0, groups=self.in_channels)
        # (B, 4*C, H, W) -> (B, C, 4, H, W)
        return out.view(x.shape[0], self.in_channels, 4, x.shape[2], x.shape[3])


# ------------------------------------------------------------
# Feature Adaptation Module (FAM): inverted residual-style
# ------------------------------------------------------------
class FeatureAdaptationModule(nn.Module):
    """
    Teacher(Transformer) feature를 Student(CNN)가 이해하기 쉬운 local-context 형태로
    변환하기 위한 inverted residual block.

    구조:
      Conv1x1 (expand) -> DWConv3x3 -> Conv1x1 (project)
      in/out 채널이 동일하면 residual 연결 사용.
    """

    def __init__(self, in_channels: int, out_channels: int, expansion: float = 2.0):
        super().__init__()
        hidden_dim = max(int(in_channels * expansion), 1)

        self.use_residual = (in_channels == out_channels)

        self.block = nn.Sequential(
            # expand
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),

            # depthwise 3x3
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),

            # project
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            return x + out
        return out


# ------------------------------------------------------------
# Learnable Frequency Attention (LFA) on HF subbands
# ------------------------------------------------------------
class FrequencyAttention(nn.Module):
    """
    LH/HL/HH global descriptor로부터 band-wise scalar weight를 학습.

    입력:
      desc: (B, 3)   # [d_LH, d_HL, d_HH]
    출력:
      w:    (B, 3)   # sigmoid로 [0,1] band별 weight
    """

    def __init__(self, hidden_dim: int = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )

    def forward(self, desc: torch.Tensor) -> torch.Tensor:
        return self.mlp(desc)


# ------------------------------------------------------------
# Main KD Engine
# ------------------------------------------------------------
class SWTLFAFDDKDEngine(BaseKDEngine):
    """
    SWT-aware KD with Learnable Frequency Attention + Frequency-Domain Distillation.

    주요 구성:
      - CE: energy attention 기반 high/low 영역 분리, high 영역 CE scale 조절
      - Logit KD: energy attention으로 spatial weight
      - Feature KD: FAM으로 student feat -> teacher feat space project 후 L2
      - Freq KD: teacher/student feat에 Haar SWT 적용, LL/HF (LH/HL/HH) 분리 distillation
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        w_ce_student: float,
        w_kd_logit: float,
        w_kd_feat: float,
        w_kd_freq: float = 1.0,
        temperature: float = 1.0,
        ignore_index: int = 255,
        teacher_stage: int = -2,
        student_stage: int = -2,
        energy_temperature: float = 1.5,
        freeze_teacher: bool = True,
        high_ce_scale: float = 0.3,
        freq_ll_weight: float = 1.0,
        freq_hf_weight: float = 1.0,
    ) -> None:
        super().__init__(teacher, student)
        self.w_ce_student = float(w_ce_student)
        self.w_kd_logit = float(w_kd_logit)
        self.w_kd_feat = float(w_kd_feat)
        self.w_kd_freq = float(w_kd_freq)

        self.temperature = float(temperature)
        self.ignore_index = int(ignore_index)
        self.teacher_stage = int(teacher_stage)
        self.student_stage = int(student_stage)
        self.energy_temperature = float(energy_temperature)
        self._freeze_teacher = bool(freeze_teacher)

        self.high_ce_scale = float(high_ce_scale)
        self.freq_ll_weight = float(freq_ll_weight)
        self.freq_hf_weight = float(freq_hf_weight)

        # CE / KL
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.kl_loss = nn.KLDivLoss(reduction="none")

        # feature adaptation modules (student -> teacher)
        self.feat_adapters = nn.ModuleDict()

        # frequency attention (LH/HL/HH band scalar weight 학습)
        self.freq_attention = FrequencyAttention(hidden_dim=8)

        # Haar SWT 캐시 (채널 수별로 재사용)
        self._swt_modules: Dict[int, HaarSWT] = {}

        if self._freeze_teacher:
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()

    # --------------------------------------------------------
    # 기본 동작 오버라이드: teacher는 항상 eval
    # --------------------------------------------------------
    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        if self._freeze_teacher:
            self.teacher.eval()
        return self

    # --------------------------------------------------------
    # 유틸 함수들
    # --------------------------------------------------------
    def _forward_with_feats(
        self, model: nn.Module, imgs: torch.Tensor
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

    # --------------------------------------------------------
    # SWT + LFA 관련
    # --------------------------------------------------------
    def _get_swt(self, feat: torch.Tensor) -> HaarSWT:
        c = feat.shape[1]
        if c not in self._swt_modules:
            self._swt_modules[c] = HaarSWT(c).to(feat.device, feat.dtype)
        return self._swt_modules[c]

    def _swt_decompose(self, feat: torch.Tensor):
        """
        feat: (B, C, H, W)
        return: (LL, LH, HL, HH), 각각 (B, C, H, W)
        """
        swt = self._get_swt(feat)
        out = swt(feat)  # (B, C, 4, H, W)
        ll = out[:, :, 0]
        lh = out[:, :, 1]
        hl = out[:, :, 2]
        hh = out[:, :, 3]
        return ll, lh, hl, hh

    @staticmethod
    def _hf_global_descriptor(
        lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor
    ) -> torch.Tensor:
        """
        HF subband 절대값의 global 평균을 이용해 (B,3) descriptor 생성.
        """
        def _mean_abs(x: torch.Tensor) -> torch.Tensor:
            return x.abs().mean(dim=(1, 2, 3))  # (B,)

        d_lh = _mean_abs(lh)
        d_hl = _mean_abs(hl)
        d_hh = _mean_abs(hh)
        desc = torch.stack([d_lh, d_hl, d_hh], dim=1)  # (B,3)
        return desc

    def _energy_and_attention(
        self, t_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        teacher feature로부터:
          - LL/LH/HL/HH subband
          - learnable band weight (w_LH, w_HL, w_HH)
          - energy map / attention map
        를 생성.
        """
        ll, lh, hl, hh = self._swt_decompose(t_feat)  # (B,C,H,W)

        # HF descriptor -> band-wise scalar weight (0~1)
        desc = self._hf_global_descriptor(lh, hl, hh)  # (B,3)
        band_w = self.freq_attention(desc)             # (B,3)

        b = t_feat.size(0)
        w_lh = band_w[:, 0].view(b, 1, 1, 1)
        w_hl = band_w[:, 1].view(b, 1, 1, 1)
        w_hh = band_w[:, 2].view(b, 1, 1, 1)

        # learnable weighted HF energy
        energy = (
            w_lh * lh.abs() +
            w_hl * hl.abs() +
            w_hh * hh.abs()
        ).mean(dim=1, keepdim=True)  # (B,1,H,W)

        # normalize & softmax로 attention map 생성
        norm_energy = energy / (energy.mean(dim=[1, 2, 3], keepdim=True) + 1e-6)
        attn = torch.softmax(norm_energy.flatten(2) * self.energy_temperature, dim=-1)
        attn = attn.view_as(norm_energy)

        # band_w: (B,3)
        return energy.detach(), attn, band_w.detach()

    # --------------------------------------------------------
    # Feature adaptation (student -> teacher)
    # --------------------------------------------------------
    def _maybe_project_student_feat(self, s_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        """
        Student feature를 teacher feature channel에 맞추는 모듈.
        단순 1x1 conv 대신 inverted residual 기반 FAM 사용.
        """
        if s_feat.shape[1] == t_feat.shape[1]:
            return s_feat

        key = f"s{self.student_stage}_t{self.teacher_stage}"
        if key not in self.feat_adapters:
            self.feat_adapters[key] = FeatureAdaptationModule(
                in_channels=s_feat.shape[1],
                out_channels=t_feat.shape[1],
                expansion=2.0,
            )
        adapter = self.feat_adapters[key].to(s_feat.device, dtype=s_feat.dtype)
        return adapter(s_feat)

    # --------------------------------------------------------
    # Losses
    # --------------------------------------------------------
    def _weighted_feat_loss(
        self, s_feat: torch.Tensor, t_feat: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        if s_feat.shape[-2:] != t_feat.shape[-2:]:
            t_feat = F.interpolate(t_feat, size=s_feat.shape[-2:], mode="bilinear", align_corners=True)
        if weight.shape[-2:] != s_feat.shape[-2:]:
            weight = F.interpolate(weight, size=s_feat.shape[-2:], mode="bilinear", align_corners=True)
        diff = (s_feat - t_feat.detach()) ** 2
        weighted = diff * weight
        denom = weight.sum().clamp(min=1e-6)
        return weighted.sum() / denom

    def _weighted_logit_kd(
        self,
        s_logits: torch.Tensor,
        t_logits: torch.Tensor,
        masks: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        log_p_s = F.log_softmax(s_logits / self.temperature, dim=1)
        p_t = F.softmax(t_logits / self.temperature, dim=1)
        pixel_loss = self.kl_loss(log_p_s, p_t).sum(dim=1)  # (B,H,W)

        valid_mask = (masks != self.ignore_index).float().unsqueeze(1)  # (B,1,H,W)

        if weight.shape[-2:] != s_logits.shape[-2:]:
            weight = F.interpolate(weight, size=s_logits.shape[-2:], mode="bilinear", align_corners=True)

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
        CE를 high-attention / low-attention으로 분리.
        high 영역에서는 self.high_ce_scale만큼만 CE를 반영하여
        그 영역은 KD 신호에 더 의존하도록 유도.
        """
        # per-pixel CE: (B,H,W)
        ce_map = F.cross_entropy(
            s_logits,
            masks,
            ignore_index=self.ignore_index,
            reduction="none",
        )  # (B,H,W)

        valid_mask = (masks != self.ignore_index).float().unsqueeze(1)  # (B,1,H,W)

        attn = energy_attn
        if attn.shape[-2:] != s_logits.shape[-2:]:
            attn = F.interpolate(attn, size=s_logits.shape[-2:], mode="bilinear", align_corners=False)

        attn_flat = attn.view(attn.size(0), -1)
        q_high = torch.quantile(attn_flat, 0.8, dim=1, keepdim=True)  # (B,1)
        high_mask = (attn_flat >= q_high).view_as(attn)               # (B,1,H,W)
        low_mask = ~high_mask                                        # bool

        high_mask = high_mask.float() * valid_mask
        low_mask = low_mask.float() * valid_mask

        ce_map = ce_map.unsqueeze(1)  # (B,1,H,W)

        denom_low = low_mask.sum().clamp(min=1.0)
        denom_high = high_mask.sum().clamp(min=1.0)

        ce_low = (ce_map * low_mask).sum() / denom_low
        ce_high = (ce_map * high_mask).sum() / denom_high

        ce_total = ce_low + self.high_ce_scale * ce_high

        return ce_total, ce_low.detach(), ce_high.detach()

    def _frequency_distill_loss(
        self,
        s_feat: torch.Tensor,
        t_feat: torch.Tensor,
        band_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Frequency-Domain Distillation (FDD)

        L_freq = λ_LL * ||LL_s - LL_t||^2 + λ_HF * Σ_k w_k * ||Subband^k_s - Subband^k_t||^2

        - band_weights: (B,3) = [w_LH, w_HL, w_HH] from LFA
        """
        # spatial size align
        if s_feat.shape[-2:] != t_feat.shape[-2:]:
            s_feat = F.interpolate(s_feat, size=t_feat.shape[-2:], mode="bilinear", align_corners=True)

        ll_s, lh_s, hl_s, hh_s = self._swt_decompose(s_feat)
        ll_t, lh_t, hl_t, hh_t = self._swt_decompose(t_feat.detach())

        # mean-squared error per subband
        ll_loss = (ll_s - ll_t).pow(2).mean()

        lh_loss = (lh_s - lh_t).pow(2).mean()
        hl_loss = (hl_s - hl_t).pow(2).mean()
        hh_loss = (hh_s - hh_t).pow(2).mean()

        # band scalar weight: batch-wise 평균 사용
        w_lh = band_weights[:, 0].mean()
        w_hl = band_weights[:, 1].mean()
        w_hh = band_weights[:, 2].mean()

        hf_loss = w_lh * lh_loss + w_hl * hl_loss + w_hh * hh_loss

        return self.freq_ll_weight * ll_loss + self.freq_hf_weight * hf_loss

    # --------------------------------------------------------
    # public API
    # --------------------------------------------------------
    def compute_losses(self, imgs, masks, device) -> Dict[str, Any]:
        # imgs: student/teacher image pair or dict
        s_imgs, t_imgs = self._parse_inputs(imgs)

        s_logits, s_feats = self._forward_with_feats(self.student, s_imgs)

        if self._freeze_teacher:
            with torch.no_grad():
                t_logits, t_feats = self._forward_with_feats(self.teacher, t_imgs)
        else:
            t_logits, t_feats = self._forward_with_feats(self.teacher, t_imgs)

        s_feats = tuple(s_feats)
        t_feats = tuple(t_feats)

        # 선택 feature
        t_feat_sel = self._select_feat(t_feats, self.teacher_stage)
        s_feat_sel = self._select_feat(s_feats, self.student_stage)

        # student feat -> teacher feat space (FAM)
        s_feat_sel_proj = self._maybe_project_student_feat(s_feat_sel, t_feat_sel)

        # teacher feature SWT + LFA 기반 energy / attention
        raw_energy, energy_attn, band_w = self._energy_and_attention(t_feat_sel)

        # CE: high/low attention 분리
        ce_student, ce_low, ce_high = self._ce_with_attention(s_logits, masks, energy_attn)

        # Logit KD (spatial attention)
        kd_logit = self._weighted_logit_kd(s_logits, t_logits, masks, energy_attn)

        # Feature KD (spatial attention)
        kd_feat = self._weighted_feat_loss(s_feat_sel_proj, t_feat_sel, energy_attn)

        # Frequency-domain KD
        kd_freq = self._frequency_distill_loss(s_feat_sel_proj, t_feat_sel, band_w)

        total = (
            self.w_ce_student * ce_student
            + self.w_kd_logit * kd_logit
            + self.w_kd_feat * kd_feat
            + self.w_kd_freq * kd_freq
        )

        return {
            "total": total,
            "ce_student": ce_student.detach(),
            "ce_student_low": ce_low,
            "ce_student_high": ce_high,
            "kd_logit": kd_logit.detach(),
            "kd_feat": kd_feat.detach(),
            "kd_freq": kd_freq.detach(),
            "s_logits": s_logits.detach(),
            "t_logits": t_logits.detach(),
            "student_input": s_imgs.detach(),
            "teacher_input": t_imgs.detach(),
            "swt_energy": raw_energy,           # (B,1,H,W)
            "swt_attention": energy_attn,       # (B,1,H,W)
            "freq_band_weight": band_w,         # (B,3)
        }

    def get_extra_parameters(self):
        # feat_adapters + freq_attention MLP 파라미터를 optimizer에 추가할 수 있도록 반환
        params = list(self.feat_adapters.parameters()) + list(self.freq_attention.parameters())
        return params
