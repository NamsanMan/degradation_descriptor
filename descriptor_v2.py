import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


# ==========================================
# 1. Improved Target Generator (Log-Scale)
# ==========================================
class LogWaveletDegradationTarget:
    """
    [개선점]
    1. FFT 대신 Wavelet 기반 에너지 측정 (CNN 입력과 정합성 향상)
    2. Linear Diff 대신 Log-Scale Diff 사용 (민감도 안정화)
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def _haar_dwt(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        RGB -> Gray -> DWT 분해
        Return: (Low_Energy, Mid_Energy, High_Energy)
        """
        # RGB to Gray
        if x.size(1) == 3:
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        B, C, H, W = x.shape
        # Pad if needed
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, W % 2, 0, H % 2), 'reflect')

        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 0::2, 1::2]
        x2 = x[:, :, 1::2, 0::2]
        x3 = x[:, :, 1::2, 1::2]

        ll = (x0 + x1 + x2 + x3) / 4.0  # Low Freq
        lh = (x0 + x2 - x1 - x3) / 4.0  # Vertical (Mid)
        hl = (x0 + x1 - x2 - x3) / 4.0  # Horizontal (Mid)
        hh = (x0 + x3 - x1 - x2) / 4.0  # Diagonal (High/Noise)

        # 에너지(Energy) 관점에서 통합
        # Low: LL
        # Mid: LH + HL
        # High: HH
        E_low = torch.abs(ll)
        E_mid = torch.sqrt(lh ** 2 + hl ** 2 + self.eps)
        E_high = torch.abs(hh)

        return E_low, E_mid, E_high

    def __call__(self, hr: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        """
        Return: [B, 3] (Low, Mid, High degradation scores 0~1)
        """
        # LR Up-sampling to match HR size (if needed)
        if lr.shape[-2:] != hr.shape[-2:]:
            lr = F.interpolate(lr, size=hr.shape[-2:], mode='bilinear', align_corners=False)

        # 1. DWT 수행
        l_hr, m_hr, h_hr = self._haar_dwt(hr)
        l_lr, m_lr, h_lr = self._haar_dwt(lr)

        # 2. Band별 평균 에너지 계산 (Global Average)
        def get_avg_energy(e_map):
            return e_map.mean(dim=(2, 3), keepdim=True)  # [B, 1, 1, 1]

        avg_l_hr, avg_m_hr, avg_h_hr = map(get_avg_energy, [l_hr, m_hr, h_hr])
        avg_l_lr, avg_m_lr, avg_h_lr = map(get_avg_energy, [l_lr, m_lr, h_lr])

        # 3. Log-Scale Difference (유사 dB)
        # diff = | log(LR) - log(HR) |
        # 값이 0이면 동일, 클수록 차이가 큼 (Blur든 Noise든 에너지 분포가 바뀜)
        def get_score(e_hr, e_lr):
            # log10 사용
            log_hr = torch.log10(e_hr + self.eps)
            log_lr = torch.log10(e_lr + self.eps)
            diff = torch.abs(log_lr - log_hr)

            # Normalization Heuristic:
            # log 차이가 1.0 (10배 에너지 차이) 이상이면 Max Score(1.0)
            return torch.clamp(diff, 0.0, 1.0).view(-1, 1)

        s_low = get_score(avg_l_hr, avg_l_lr)
        s_mid = get_score(avg_m_hr, avg_m_lr)
        s_high = get_score(avg_h_hr, avg_h_lr)

        return torch.cat([s_low, s_mid, s_high], dim=1)  # [B, 3]


# ==========================================
# 2. Wavelet-based Descriptor Network
# ==========================================
def haar_dwt_decompose(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B, 3, H, W] -> Output: [B, 12, H/2, W/2]
    RGB 각 채널별로 4개 subband 생성 -> 총 12채널
    """
    B, C, H, W = x.shape
    if H % 2 != 0 or W % 2 != 0:
        x = F.pad(x, (0, W % 2, 0, H % 2), 'reflect')

    x0 = x[:, :, 0::2, 0::2]
    x1 = x[:, :, 0::2, 1::2]
    x2 = x[:, :, 1::2, 0::2]
    x3 = x[:, :, 1::2, 1::2]

    ll = (x0 + x1 + x2 + x3) / 4.0
    lh = (x0 + x2 - x1 - x3) / 4.0
    hl = (x0 + x1 - x2 - x3) / 4.0
    hh = (x0 + x3 - x1 - x2) / 4.0

    return torch.cat([ll, lh, hl, hh], dim=1)  # dim 1에 concat


class WaveletDescriptorNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_bands: int = 3):
        super().__init__()

        # 입력: 12채널 (RGB 3 * Wavelet 4)
        input_dim = in_channels * 4

        # Lightweight Backbone
        self.features = nn.Sequential(
            # Stage 1: Fusion (12 -> 32)
            nn.Conv2d(input_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Downsample 1
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Downsample 2
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Global Context
            nn.AdaptiveAvgPool2d(1)
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_bands),
            nn.Sigmoid()  # 0~1 Score
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Explicit Frequency Decomposition
        x_dwt = haar_dwt_decompose(x)  # [B, 12, H/2, W/2]

        # 2. CNN Feature Extraction
        feat = self.features(x_dwt)

        # 3. Prediction
        score = self.head(feat)
        return score


# 호환성을 위해 기존 이름으로 alias
DegradationDescriptorNet = WaveletDescriptorNet
FrequencyDegradationTarget = LogWaveletDegradationTarget