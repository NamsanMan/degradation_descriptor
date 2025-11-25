# d3p.py (Final Optimized & Generalized with Score Normalization)
from typing import List, Sequence, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

# Descriptor V3 (SWT)
# descriptor_v3.py가 프로젝트 루트에 있어야 합니다.
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
# DAS Block: Soft Gating (Safety-First)
# ==========================================
class DASBlock(nn.Module):
    """
    Model-Agnostic Feature Modulator
    """

    def __init__(self, in_channels: int, descriptor_dim: int = 3):
        super().__init__()
        self.in_channels = in_channels

        self.mlp = nn.Sequential(
            nn.Linear(descriptor_dim, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels * 2),
        )

        # [Initialization]
        # 학습 초기에는 Identity(Scale=1, Shift=0)에 가깝게 동작하도록
        # weight를 0 근처로 초기화함 (약간 키워서 반응성 확보)
        with torch.no_grad():
            self.mlp[-1].weight.data.normal_(mean=0.0, std=0.01)  # 0.001 -> 0.01
            self.mlp[-1].bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        params = self.mlp(scores)
        gamma_raw, beta_raw = torch.split(params, self.in_channels, dim=1)

        gamma_raw = gamma_raw.view(-1, self.in_channels, 1, 1)
        beta_raw = beta_raw.view(-1, self.in_channels, 1, 1)

        # Soft Gating Range
        ALPHA = 0.3  # 0.2 -> 0.3 (반응성 확대)
        BETA_LIMIT = 0.5

        gamma = 1.0 + ALPHA * torch.tanh(gamma_raw)
        beta = BETA_LIMIT * torch.tanh(beta_raw)

        return x * gamma + beta


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
            print(f"▶ [d3p] Enabling DAS Mode (General: All Stages + Score Norm)")
            if descriptor_path is None:
                raise ValueError("descriptor_path must be provided when use_das=True")

            # 1. Load Descriptor (V3 SWT)
            self.descriptor = DegradationDescriptorNet(in_channels=3, num_bands=3)

            # Load weights
            try:
                ckpt = torch.load(descriptor_path, map_location='cpu', weights_only=False)
            except TypeError:
                ckpt = torch.load(descriptor_path, map_location='cpu')

            state_dict = ckpt.get("model_state", ckpt)
            self.descriptor.load_state_dict(state_dict)

            # Freeze Descriptor
            self.descriptor.eval()
            for param in self.descriptor.parameters():
                param.requires_grad = False

            # 2. Create DAS Blocks for ALL Encoder Outputs
            enc_out_ch = getattr(self.encoder, "out_channels", [])
            self.das_blocks = nn.ModuleList()

            for i, ch in enumerate(enc_out_ch):
                if ch > 0:
                    self.das_blocks.append(DASBlock(in_channels=ch, descriptor_dim=3))
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

    # [FIXED] forward function in DeepLabV3PlusWrapper
    def forward(
            self,
            x: torch.Tensor,
            return_feats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:

        x_pad, (H_orig, W_orig), (pad_h, pad_w) = self._pad_to_stride(x, AUTO_PAD_STRIDE)

        # 1) Descriptor Inference & Safer Normalization
        deg_scores = None
        if self.use_das:
            with torch.no_grad():
                raw_scores = self.descriptor(x_pad)  # [B, 3], Range: 0~1

                # [CRITICAL FIX]
                # Z-score 정규화는 배치 내 분산이 0일 때 Gradient 폭발 위험이 있음.
                # 대신 0.5를 중심으로 단순 Shift만 수행하거나, Min-Max 스케일링을 사용.

                # 전략: Centering only (안전함)
                # 0~1 범위를 -1~1 범위로 매핑하여 MLP가 양수/음수를 모두 보게 함.
                deg_scores = (raw_scores - 0.5) * 2.0

                # 만약 분산을 키우고 싶다면 상수를 곱하되, 나눗셈은 피함.
                # deg_scores = deg_scores * 2.0 # (선택사항: 민감도 2배 증가)

        # 2) Encoder Forward
        features: List[torch.Tensor] = self.encoder(x_pad)

        # 3) Feature Modulation
        if self.use_das and deg_scores is not None:
            new_features = []
            for i, feat in enumerate(features):
                if i < len(self.das_blocks):
                    # 안전한 deg_scores가 들어감
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