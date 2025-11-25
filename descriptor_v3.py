import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


# ==========================================
# 1. Haar SWT Implementation (Conv-based)
# ==========================================
class HaarSWT(nn.Module):
    """
    Stationary Wavelet Transform using fixed Convolution kernels.
    Stride=1 (No downsampling) -> Output size equals Input size.
    """

    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels

        # Haar Filters (2x2)
        # 0.25 scaling to maintain magnitude consistent with averaging
        k_ll = torch.tensor([[0.25, 0.25], [0.25, 0.25]])
        k_lh = torch.tensor([[0.25, -0.25], [0.25, -0.25]])
        k_hl = torch.tensor([[0.25, 0.25], [-0.25, -0.25]])
        k_hh = torch.tensor([[0.25, -0.25], [-0.25, 0.25]])

        # Stack filters: [4, 2, 2]
        filters = torch.stack([k_ll, k_lh, k_hl, k_hh], dim=0)

        # [FIX] 차원 명시적 확장
        # Conv2d Weight Shape: [Out_Channels, In_Channels/Groups, K, K]
        # 목표: 각 입력 채널마다 4개의 필터를 적용 (Grouped Conv)
        # Shape: [4, 1, 2, 2] (4 filters per 1 input channel)
        filters = filters.unsqueeze(1)

        # Repeat for input channels
        # Final Shape: [in_channels * 4, 1, 2, 2]
        # 예: RGB(3) -> [12, 1, 2, 2]
        if in_channels > 1:
            filters = filters.repeat(in_channels, 1, 1, 1)

        self.register_buffer('filters', filters)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        # Padding to maintain same size with 2x2 kernel (Reflect padding)
        # Pad right and bottom
        x_pad = F.pad(x, (0, 1, 0, 1), mode='reflect')

        # Grouped Convolution to apply filters per channel independently
        # groups=C ensures each input channel gets its own set of 4 filters
        out = F.conv2d(x_pad, self.filters, stride=1, groups=C)

        # out: [B, 4*C, H, W]
        return out


# ==========================================
# 2. Improved Target Generator (Log-SWT)
# ==========================================
class LogSWTDegradationTarget:
    """
    SWT 기반으로 Energy를 계산하므로 Shift-Invariant하고 더 정밀함.
    """

    def __init__(self, eps: float = 1e-6, device='cpu'):
        self.eps = eps
        # Target Generator 내부는 Grayscale 변환 후 처리하므로 in_channels=1
        self.swt = HaarSWT(in_channels=1).to(device)
        self.device = device

    def _to_gray(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 3:
            return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        return x

    def _get_swt_energy(self, x: torch.Tensor):
        # x: [B, 1, H, W] -> swt: [B, 4, H, W]
        # Channels: LL(0), LH(1), HL(2), HH(3)
        if x.device != self.swt.filters.device:
            self.swt = self.swt.to(x.device)

        feat = self.swt(x)

        # Energy Calculation
        E_low = torch.abs(feat[:, 0:1])  # LL
        E_mid = torch.sqrt(feat[:, 1:2] ** 2 + feat[:, 2:3] ** 2 + self.eps)  # LH + HL
        E_high = torch.abs(feat[:, 3:4])  # HH

        return E_low, E_mid, E_high

    def __call__(self, hr: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        # Resize LR if needed
        if lr.shape[-2:] != hr.shape[-2:]:
            lr = F.interpolate(lr, size=hr.shape[-2:], mode='bilinear', align_corners=False)

        # 1. Gray & SWT
        gray_hr = self._to_gray(hr)
        gray_lr = self._to_gray(lr)

        l_hr, m_hr, h_hr = self._get_swt_energy(gray_hr)
        l_lr, m_lr, h_lr = self._get_swt_energy(gray_lr)

        # 2. Band별 평균 에너지 계산 (Global Average)
        def get_avg_energy(e_map):
            return e_map.mean(dim=(2, 3), keepdim=True)

        avg_l_hr, avg_m_hr, avg_h_hr = map(get_avg_energy, [l_hr, m_hr, h_hr])
        avg_l_lr, avg_m_lr, avg_h_lr = map(get_avg_energy, [l_lr, m_lr, h_lr])

        # 3. Log-Scale Difference
        def get_score(e_hr, e_lr):
            log_hr = torch.log10(e_hr + self.eps)
            log_lr = torch.log10(e_lr + self.eps)
            diff = torch.abs(log_lr - log_hr)
            return torch.clamp(diff, 0.0, 1.0).view(-1, 1)

        s_low = get_score(avg_l_hr, avg_l_lr)
        s_mid = get_score(avg_m_hr, avg_m_lr)
        s_high = get_score(avg_h_hr, avg_h_lr)

        return torch.cat([s_low, s_mid, s_high], dim=1)


# ==========================================
# 3. SWT Descriptor Network
# ==========================================
class SWTDescriptorNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_bands: int = 3):
        super().__init__()

        self.swt = HaarSWT(in_channels=in_channels)

        # Input: RGB(3) -> SWT -> 12 Channels (3 * 4)
        input_dim = in_channels * 4

        self.features = nn.Sequential(
            # Stage 1: Fusion & Downsample (12 -> 32)
            nn.Conv2d(input_dim, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Stage 2: (32 -> 64)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Stage 3: (64 -> 128)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Stage 4: (128 -> 128)
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1)
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_bands),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. SWT Decomposition
        x_swt = self.swt(x)  # [B, 12, H, W]

        # 2. CNN Extraction
        feat = self.features(x_swt)

        # 3. Score
        return self.head(feat)


# Alias for compatibility
DegradationDescriptorNet = SWTDescriptorNet
FrequencyDegradationTarget = LogSWTDegradationTarget