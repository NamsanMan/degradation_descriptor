# d3p.py (Dynamic Convolution Version)
from typing import List, Sequence, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

# Descriptor V3 (SWT)
from descriptor_v3 import DegradationDescriptorNet
from config import DATA, MODEL

DEFAULT_ENCODER_NAME = "resnet50"
DEFAULT_ENCODER_WEIGHTS = "imagenet"
DEFAULT_IN_CHANNELS = 3
DEFAULT_NUM_CLASSES = DATA.NUM_CLASSES
DEFAULT_STAGE_INDICES: Tuple[int, ...] = (0, 1, 2, 3, 4)

AUTO_PAD_STRIDE = 16
PAD_MODE = "replicate"


# ==========================================
# [NEW] Dynamic Convolution DAS Block
# ==========================================
class DynamicDASBlock(nn.Module):
    """
    Descriptor 점수에 따라 Conv Weight를 동적으로 생성하는 레이어 (CondConv)
    - FiLM(값 조절) 대신 필터(Weight) 자체를 바꿔서 근본적인 대응 수행.
    - 예: 노이즈가 심하면 Smoothing 필터, 블러가 심하면 Sharpening 필터 자동 합성.
    """

    def __init__(self, in_channels: int, descriptor_dim: int = 3, num_experts: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.num_experts = num_experts  # 전문가 개수 (예: Clean용, Noise용, Blur용)

        # 1. Routing Function (Descriptor -> Expert Weights)
        self.routing_fc = nn.Sequential(
            nn.Linear(descriptor_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, num_experts),
            nn.Softmax(dim=1)  # 확률값 (합이 1)
        )

        # 2. Expert Weights (Static Kernels)
        # 1x1 Conv를 사용하여 채널 간의 관계를 동적으로 재설정
        # Shape: [num_experts, in_channels, in_channels, 1, 1]
        # 주의: 파라미터 수가 너무 커지지 않게 1x1 Conv 사용 권장
        self.weight = nn.Parameter(
            torch.Tensor(num_experts, in_channels, in_channels, 1, 1)
        )

        # Initialization
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        scores: [B, 3]
        """
        B, C, H, W = x.shape

        # 1. Routing Weights 계산
        # routing_w: [B, num_experts]
        routing_w = self.routing_fc(scores)

        # 2. Dynamic Kernel 합성
        # weights: [B, C_out, C_in, 1, 1]
        # (B, K) x (K, C, C, 1, 1) -> (B, C, C, 1, 1)
        # einsum을 쓰면 깔끔하지만, 직관적인 구현을 위해 reshape 사용

        # [B, K, 1, 1, 1, 1] * [1, K, C, C, 1, 1] -> Sum over K
        routing_w = routing_w.view(B, self.num_experts, 1, 1, 1, 1)
        expert_weights = self.weight.unsqueeze(0)

        # 배치별 동적 커널 생성
        dynamic_weight = (routing_w * expert_weights).sum(dim=1)  # [B, C, C, 1, 1]

        # 3. Apply Conv per Sample (Efficient Group Conv)
        # Pytorch는 배치별 커널 적용을 직접 지원하지 않으므로 Group Conv 트릭 사용

        # Input을 [1, B*C, H, W]로 변형
        x_grouped = x.view(1, B * C, H, W)

        # Weight를 [B*C, C, 1, 1]로 변형 (Group=B)
        # Conv2d의 weight shape는 [Out, In/Groups, k, k]
        w_grouped = dynamic_weight.view(B * C, C, 1, 1)

        # Group Conv 수행 (각 샘플이 자신의 커널로 컨볼루션됨)
        out = F.conv2d(x_grouped, w_grouped, stride=1, padding=0, groups=B)

        # 원래 형태로 복구
        out = out.view(B, C, H, W)

        # Residual Connection (안전장치)
        return x + out


class DeepLabV3PlusWrapper(nn.Module):
    def __init__(
            self,
            base_model: smp.DeepLabV3Plus,
            stage_indices: Sequence[int] = DEFAULT_STAGE_INDICES,
            num_classes: int = DEFAULT_NUM_CLASSES,
            use_das: bool = False,
            descriptor_path: str = None
    ) -> None:
        super().__init__()
        self.encoder = base_model.encoder
        self.decoder = base_model.decoder
        self.segmentation_head = base_model.segmentation_head

        self.stage_indices: Tuple[int, ...] = tuple(stage_indices)
        self.num_classes = int(num_classes)

        # === DAS Module Setup ===
        self.use_das = use_das
        if self.use_das:
            print(f"▶ [d3p] Enabling DAS Mode (Dynamic Convolution Version)")
            if descriptor_path is None:
                raise ValueError("descriptor_path must be provided when use_das=True")

            self.descriptor = DegradationDescriptorNet(in_channels=3, num_bands=3)
            try:
                ckpt = torch.load(descriptor_path, map_location='cpu', weights_only=False)
            except TypeError:
                ckpt = torch.load(descriptor_path, map_location='cpu')

            state_dict = ckpt.get("model_state", ckpt)
            self.descriptor.load_state_dict(state_dict)
            self.descriptor.eval()
            for param in self.descriptor.parameters():
                param.requires_grad = False

            # Dynamic Conv는 파라미터가 많으므로 필요한 곳(2, 5)에만 적용 권장
            # 하지만 일반화를 위해 전체 적용하되, num_experts를 작게 유지
            enc_out_ch = getattr(self.encoder, "out_channels", [])
            self.das_blocks = nn.ModuleList()

            for i, ch in enumerate(enc_out_ch):
                if ch > 0:
                    # [Change] DynamicDASBlock 사용
                    self.das_blocks.append(DynamicDASBlock(in_channels=ch, descriptor_dim=3))
                else:
                    self.das_blocks.append(nn.Identity())
        else:
            self.descriptor = None
            self.das_blocks = None

        # Property setup
        enc_out_ch = getattr(self.encoder, "out_channels", None)
        if isinstance(enc_out_ch, (list, tuple)):
            self._feat_channels = [
                enc_out_ch[i] for i in self.stage_indices if 0 <= i < len(enc_out_ch)
            ]
        else:
            self._feat_channels = None

    @property
    def feat_channels(self) -> Union[List[int], None]:
        return self._feat_channels

    def _pad_to_stride(self, x: torch.Tensor, stride: int = AUTO_PAD_STRIDE):
        B, C, H, W = x.shape
        Ht = math.ceil(H / stride) * stride
        Wt = math.ceil(W / stride) * stride
        pad_h = Ht - H
        pad_w = Wt - W
        if pad_h == 0 and pad_w == 0:
            return x, (H, W), (0, 0)
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode=PAD_MODE)
        return x_pad, (H, W), (pad_h, pad_w)

    def _crop_spatial(self, t: torch.Tensor, H: int, W: int) -> torch.Tensor:
        return t[..., :H, :W]

    def forward(
            self,
            x: torch.Tensor,
            return_feats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:

        x_pad, (H_orig, W_orig), (pad_h, pad_w) = self._pad_to_stride(x, AUTO_PAD_STRIDE)

        # 1) Descriptor Inference (No Normalization Needed for Softmax)
        deg_scores = None
        if self.use_das:
            with torch.no_grad():
                deg_scores = self.descriptor(x_pad)  # [B, 3]

        # 2) Encoder Forward
        features: List[torch.Tensor] = self.encoder(x_pad)

        # 3) Dynamic Feature Modulation
        if self.use_das and deg_scores is not None:
            new_features = []
            for i, feat in enumerate(features):
                if i < len(self.das_blocks):
                    # Dynamic Conv 실행
                    feat = self.das_blocks[i](feat, deg_scores)
                new_features.append(feat)
            features = new_features

        # 4) Decoder & Head
        dec_out: torch.Tensor = self.decoder(features)
        logits_pad: torch.Tensor = self.segmentation_head(dec_out)

        logits = self._crop_spatial(logits_pad, H_orig, W_orig)

        if not return_feats:
            return logits

        Hp, Wp = x_pad.shape[-2], x_pad.shape[-1]
        feats_out: List[torch.Tensor] = []
        for i in self.stage_indices:
            if i >= len(features): continue
            f = features[i]
            fh, fw = f.shape[-2], f.shape[-1]
            sh = max(1, Hp // fh)
            sw = max(1, Wp // fw)
            Hf = math.ceil(H_orig / sh)
            Wf = math.ceil(W_orig / sw)
            f = self._crop_spatial(f, Hf, Wf)
            feats_out.append(f)

        return logits, feats_out


# --- Factory Functions ---

def create_model(
        encoder_name: str = DEFAULT_ENCODER_NAME,
        encoder_weights: Union[str, None] = DEFAULT_ENCODER_WEIGHTS,
        in_channels: int = DEFAULT_IN_CHANNELS,
        classes: int = DEFAULT_NUM_CLASSES,
        stage_indices: Sequence[int] = DEFAULT_STAGE_INDICES,
        use_das: bool = False,
        descriptor_path: str = None,
        **kwargs,
) -> DeepLabV3PlusWrapper:
    base = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        **kwargs,
    )
    return DeepLabV3PlusWrapper(
        base_model=base,
        stage_indices=stage_indices,
        num_classes=classes,
        use_das=use_das,
        descriptor_path=descriptor_path
    )