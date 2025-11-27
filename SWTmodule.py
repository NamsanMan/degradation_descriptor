import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. Haar SWT Implementation (Base)
# ==========================================
class HaarSWT(nn.Module):
    """
    Stationary Wavelet Transform using fixed Convolution kernels.
    Input: [B, 3, H, W] -> Output: [B, 12, H, W] (For RGB)
    Channels per input: LL, LH, HL, HH
    """

    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels

        # Haar Filters (2x2)
        # Note: 0.5 scaling is often used for energy conservation in standard WT,
        # but 0.25 is fine if consistent. Keeping user's 0.25 setting.
        k_ll = torch.tensor([[0.25, 0.25], [0.25, 0.25]])
        k_lh = torch.tensor([[0.25, -0.25], [0.25, -0.25]])
        k_hl = torch.tensor([[0.25, 0.25], [-0.25, -0.25]])
        k_hh = torch.tensor([[0.25, -0.25], [-0.25, 0.25]])

        # Stack filters: [4, 1, 2, 2]
        filters = torch.stack([k_ll, k_lh, k_hl, k_hh], dim=0).unsqueeze(1)

        # Repeat for input channels to use grouped convolution
        if in_channels > 1:
            filters = filters.repeat(in_channels, 1, 1, 1)

        self.register_buffer('filters', filters)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        # Reflect padding to maintain spatial resolution (Stationary property)
        x_pad = F.pad(x, (0, 1, 0, 1), mode='reflect')

        # Grouped Conv: Each input channel is convolved with its own 4 filters
        out = F.conv2d(x_pad, self.filters, stride=1, groups=C)
        return out


# ==========================================
# 2. SWT Frequency Attention Module (New Core)
# ==========================================
class SWTFrequencyAttention(nn.Module):
    """
    Generates a Spatial Attention Map based on frequency components.
    Instead of a global vector, this outputs a [B, 1, H, W] map.
    """

    def __init__(self,
                 student_channels: int,
                 rgb_channels: int = 3,
                 reduction_ratio: int = 4):
        super().__init__()

        self.swt = HaarSWT(in_channels=rgb_channels)

        # Input dim: RGB(3) * 4 subbands = 12 channels
        swt_feat_dim = rgb_channels * 4

        # 1. Structure Analysis Network (Lightweight CNN)
        # Learns the relationship between HH (Noise) and LH/HL (Structure)
        self.attention_net = nn.Sequential(
            # Compression: 12 -> Intermediate
            nn.Conv2d(swt_feat_dim, swt_feat_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(swt_feat_dim * 2),
            nn.ReLU(inplace=True),

            # Bottleneck for context aggregation
            nn.Conv2d(swt_feat_dim * 2, swt_feat_dim, kernel_size=1),
            nn.BatchNorm2d(swt_feat_dim),
            nn.ReLU(inplace=True),

            # Generate Attention Map (Single Channel Spatial Map)
            nn.Conv2d(swt_feat_dim, 1, kernel_size=3, padding=1),
            nn.Sigmoid()

        )

        # 2. Alignment Layer (Optional)
        # If student feature channels need specific scaling, use a channel-wise scale
        self.channel_scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(student_channels, student_channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(student_channels // reduction_ratio, student_channels, 1),
            nn.Sigmoid()
        )

        # [Critical] Bias Tuning based on Dataset Statistics
        # 데이터셋 Mean Ratio가 1.3 정도이므로, Sigmoid 입력값이 0 근처가 되도록 Bias 조정
        # 이렇게 하면 Ratio 변화에 대해 Gradient가 가장 활발하게 흐름 (Vanishing Gradient 방지)
        for m in self.attention_net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # 마지막 레이어는 약간의 음수 Bias를 주어 초기에는 "Noise 억제(Low Attention)" 쪽으로 유도하다가
        # 확실한 구조(Structure)가 발견될 때만 활성화되도록 설정 (Conservative Strategy)
        if isinstance(self.attention_net[-2], nn.Conv2d):  # 마지막 Conv (Sigmoid 직전)
            nn.init.constant_(self.attention_net[-2].bias, -1.0)

    def forward(self, raw_img, student_feat):
        """
        Args:
            raw_img: Original Input Image [B, 3, H_orig, W_orig]
            student_feat: Feature map from Student Encoder [B, C, H_feat, W_feat]
        """
        # 1. Extract Frequency Features (Full Resolution)
        # swt_out: [B, 12, H_orig, W_orig]
        swt_out = self.swt(raw_img)

        # 2. Downsample SWT features to match Student Feature resolution
        # Usually Student features are 1/4, 1/8, etc.
        target_h, target_w = student_feat.shape[2], student_feat.shape[3]

        if swt_out.shape[2:] != (target_h, target_w):
            # Bilinear interpolation is safe for coefficients
            swt_resized = F.interpolate(swt_out, size=(target_h, target_w),
                                        mode='bilinear', align_corners=False)
        else:
            swt_resized = swt_out

        # 3. Generate Spatial Attention Map
        # spatial_attn: [B, 1, H_feat, W_feat]
        # High value (near 1) -> Salient Structure (Keep/Enhance)
        # Low value (near 0) -> Noise dominance (Suppress)
        spatial_attn = self.attention_net(swt_resized)

        # 4. Generate Channel Attention (Optional refinement)
        # This helps to select feature channels sensitive to edges vs texture
        channel_attn = self.channel_scale(student_feat)

        # 5. Dual Modulation
        # Apply spatial attention to emphasize structure,
        # Apply channel attention to select relevant kernels
        refined_feat = student_feat * spatial_attn * channel_attn

        # Residual Connection (Important for gradient flow)
        return refined_feat + student_feat


# ==========================================
# 3. Integration Example (Dummy Model)
# ==========================================
class RobustSegmentationStudent(nn.Module):
    """
    Example of how to integrate SWTFrequencyAttention into a CNN/Transformer Student.
    """

    def __init__(self, num_classes=19):
        super().__init__()

        # Mock Encoder layers (e.g., ResNet or MiT blocks)
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.ReLU())  # 1/2
        self.layer1 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())  # 1/2
        self.downsample = nn.MaxPool2d(2)  # 1/4
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU())  # 1/4

        # *** SWT Adapter Injection ***
        # We inject it at a stage where spatial info is still rich (e.g., 1/4 scale)
        self.swt_adapter = SWTFrequencyAttention(student_channels=128, rgb_channels=3)

        self.head = nn.Conv2d(128, num_classes, 1)

    def forward(self, x):
        # x: Raw Image [B, 3, H, W]

        # Encoder Flow
        feat = self.stem(x)  # [B, 64, H/2, W/2]
        feat = self.layer1(feat)
        feat = self.downsample(feat)  # [B, 64, H/4, W/4]
        feat = self.layer2(feat)  # [B, 128, H/4, W/4]

        # *** Apply SWT Attention ***
        # Pass both Raw Image (for frequency analysis) and Current Features
        feat_refined = self.swt_adapter(x, feat)

        # ... Rest of the network ...
        out = self.head(feat_refined)
        return out