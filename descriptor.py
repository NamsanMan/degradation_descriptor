# descriptor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class FrequencyDegradationTarget:
    """
    HR / LR 이미지 쌍으로부터
    - low / mid / high 주파수 대역별 에너지 차이를 계산하여
    - [0, 1] 범위의 degradation score를 만드는 모듈.

    사용 용도:
        - 학습 시: HR, LR 쌍으로 target 생성
        - 추론 시: 사용하지 않고, 이미 학습된 네트워크만 사용
    """

    def __init__(
        self,
        num_bands: int = 3,
        low_cut: float = 0.15,
        mid_cut: float = 0.5,
        eps: float = 1e-6,
    ):
        """
        Args:
            num_bands: 현재는 3 (low / mid / high) 기준만 고려.
            low_cut:   0~1 normalized radius에서 low/mid 경계 (예: 0.15)
            mid_cut:   0~1 normalized radius에서 mid/high 경계 (예: 0.5)
            eps:       안정성용 epsilon
        """
        assert num_bands == 3, "현재 구현은 num_bands=3 (low/mid/high)만 지원."
        assert 0.0 < low_cut < mid_cut < 1.0
        self.num_bands = num_bands
        self.low_cut = low_cut
        self.mid_cut = mid_cut
        self.eps = eps

    @staticmethod
    def _to_gray(x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W], C=1 또는 3 assumed.
        return: [B, 1, H, W]
        """
        if x.size(1) == 1:
            return x
        # RGB를 Y (luminance) 로 단순 변환
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray

    def _build_radial_mask(
        self, H: int, W: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        각 주파수 좌표 (u,v)에 대해
          - radius = sqrt(u^2 + v^2) / r_max  ∈ [0,1]
        를 계산하고, low/mid/high 대역 mask를 만든다.
        """
        # freq index: [-H/2 .. H/2-1], [-W/2 .. W/2-1]
        u = torch.fft.fftfreq(H, d=1.0).to(device)  # [-0.5, 0.5) 범위
        v = torch.fft.fftfreq(W, d=1.0).to(device)
        # meshgrid: (H, W)
        vv, uu = torch.meshgrid(v, u, indexing="xy")
        # radius 정규화 (0~0.5 범위, 여기서는 /0.5로 0~1 정규화)
        radius = torch.sqrt(uu ** 2 + vv ** 2) / 0.5  # max ~1

        low_mask = (radius <= self.low_cut).float()
        mid_mask = ((radius > self.low_cut) & (radius <= self.mid_cut)).float()
        high_mask = (radius > self.mid_cut).float()
        return low_mask, mid_mask, high_mask

    def __call__(self, hr: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hr: [B, C, H, W], clean HR image (0~1)
            lr: [B, C, h, w], degraded LR image (0~1)

        Return:
            target: [B, 3]  (low, mid, high 대역의 degradation score)
                    score ∈ [0,1] (0: 거의 무손실, 1: 강한 손실/왜곡)
        """
        B, C, H, W = hr.shape
        device = hr.device

        # 1) LR을 HR 크기로 업샘플
        lr_up = F.interpolate(lr, size=(H, W), mode="bilinear", align_corners=False)

        # 2) gray 변환
        hr_g = self._to_gray(hr)  # [B,1,H,W]
        lr_g = self._to_gray(lr_up)

        # 3) FFT -> amplitude
        # fft2: [B,1,H,W] -> [B,1,H,W]
        F_hr = torch.fft.fft2(hr_g, norm="ortho")
        F_lr = torch.fft.fft2(lr_g, norm="ortho")
        # magnitude
        A_hr = torch.abs(F_hr)  # [B,1,H,W]
        A_lr = torch.abs(F_lr)

        # 4) radial band mask
        low_mask, mid_mask, high_mask = self._build_radial_mask(H, W, device)
        # [H, W] -> [1,1,H,W] for broadcast
        low_mask = low_mask.view(1, 1, H, W)
        mid_mask = mid_mask.view(1, 1, H, W)
        high_mask = high_mask.view(1, 1, H, W)

        # 5) band별 에너지
        def band_energy(A: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            # A: [B,1,H,W], mask: [1,1,H,W]
            m = mask
            num = (A ** 2 * m).sum(dim=(2, 3))  # [B,1]
            den = m.sum(dim=(2, 3)) + self.eps
            return num / den  # [B,1]

        E_hr_low = band_energy(A_hr, low_mask)
        E_hr_mid = band_energy(A_hr, mid_mask)
        E_hr_high = band_energy(A_hr, high_mask)

        E_lr_low = band_energy(A_lr, low_mask)
        E_lr_mid = band_energy(A_lr, mid_mask)
        E_lr_high = band_energy(A_lr, high_mask)

        # 6) band별 degradation score = |E_lr - E_hr| / (E_hr + eps)
        def band_score(E_hr: torch.Tensor, E_lr: torch.Tensor) -> torch.Tensor:
            diff = torch.abs(E_lr - E_hr)
            return torch.clamp(diff / (E_hr + self.eps), 0.0, 1.0)  # [B,1]

        s_low = band_score(E_hr_low, E_lr_low)
        s_mid = band_score(E_hr_mid, E_lr_mid)
        s_high = band_score(E_hr_high, E_lr_high)

        # 7) [B,3]로 concat
        target = torch.cat([s_low, s_mid, s_high], dim=1)  # [B,3]
        return target


class DegradationDescriptorNet(nn.Module):
    """
    LR 이미지를 입력 받아,
    (low, mid, high) 주파수 대역의 degradation score ∈ [0,1]^3
    를 예측하는 작은 CNN 모듈.
    """

    def __init__(self, in_channels: int = 3, num_bands: int = 3):
        super().__init__()
        self.num_bands = num_bands

        # 단순한 lightweight CNN backbone
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # global avg pooling 후 FC
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # [B,128,1,1]
            nn.Flatten(),                  # [B,128]
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_bands),
            nn.Sigmoid(),                  # [0,1] 범위 score
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W] (LR or any degraded image, 0~1)
        return: [B, num_bands]  (low/mid/high degradation scores)
        """
        feat = self.features(x)
        scores = self.head(feat)
        return scores  # [B,3], 각 요소 ∈ [0,1]


def interpret_scores(scores: torch.Tensor) -> Dict[str, float]:
    """
    scores: [3] 또는 [1,3] 텐서
    간단한 해석용 helper.
    """
    if scores.dim() == 2:
        scores = scores[0]
    return {
        "low_freq_deg": float(scores[0].item()),
        "mid_freq_deg": float(scores[1].item()),
        "high_freq_deg": float(scores[2].item()),
    }
