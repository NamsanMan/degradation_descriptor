# d3p.py (with SWT Frequency Attention)
from typing import List, Sequence, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from models.swt_attention import SWTFrequencyAttention  # ← SWT 모듈
from config import DATA

DEFAULT_ENCODER_NAME     = "mobilenet_v2"
DEFAULT_ENCODER_WEIGHTS  = "imagenet"
DEFAULT_IN_CHANNELS      = 3
DEFAULT_NUM_CLASSES      = DATA.NUM_CLASSES

# 주의: smp encoder.features() 의 index 범위에 맞게 사용
#  - 일반적으로 0~4 (5개 stage)
#  - stage_indices 안에 잘못된 index가 있어도 아래 코드에서 자동으로 skip되도록 수정함.
DEFAULT_STAGE_INDICES: Tuple[int, ...] = (0, 1, 2, 3, 4)

# 내부 패딩 설정
AUTO_PAD_STRIDE = 16          # DeepLabV3+ 기본 output stride = 16
PAD_MODE = "replicate"        # 'replicate' or 'reflect' 권장


class DeepLabV3PlusWrapper(nn.Module):
    def __init__(
        self,
        base_model: smp.DeepLabV3Plus,
        stage_indices: Sequence[int] = DEFAULT_STAGE_INDICES,
        num_classes: int = DEFAULT_NUM_CLASSES,
        use_swt: bool = False,              # ← SWT 사용 여부 (기본 False, 실험 시 True)
        swt_stage_idx: int = 2,             # ← SWT를 적용할 encoder stage index
    ) -> None:
        super().__init__()
        self.encoder = base_model.encoder
        self.decoder = base_model.decoder
        self.segmentation_head = base_model.segmentation_head

        self.stage_indices: Tuple[int, ...] = tuple(stage_indices)
        self.num_classes = int(num_classes)

        # encoder 출력 채널 정보
        enc_out_ch = getattr(self.encoder, "out_channels", None)
        if isinstance(enc_out_ch, (list, tuple)):
            self._feat_channels = [
                enc_out_ch[i] for i in self.stage_indices if 0 <= i < len(enc_out_ch)
            ]
        else:
            self._feat_channels = None

        # === SWT Frequency Attention 설정 ===
        self.use_swt = bool(use_swt)
        self.swt_attn: Union[SWTFrequencyAttention, None] = None
        self.swt_stage_idx: int = swt_stage_idx

        if self.use_swt and isinstance(enc_out_ch, (list, tuple)):
            # stage index 보정 (범위 밖이면 마지막 stage로 보정)
            if self.swt_stage_idx < 0 or self.swt_stage_idx >= len(enc_out_ch):
                self.swt_stage_idx = len(enc_out_ch) - 1

            swt_ch = enc_out_ch[self.swt_stage_idx]
            self.swt_attn = SWTFrequencyAttention(
                student_channels=swt_ch,
                rgb_channels=3,
                reduction_ratio=4,
            )
        else:
            self.use_swt = False  # encoder 정보가 없으면 SWT 비활성화

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
        # 마지막 2차원만 크롭
        return t[..., :H, :W]

    def _denorm_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        data_loader에서 ImageNet mean/std로 정규화된 입력을
        대략 [0,1] 스케일의 RGB로 되돌리는 helper.
        """
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return x * std + mean

    def forward(
        self,
        x: torch.Tensor,
        return_feats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:

        # 1) 내부에서 stride 배수로 자동 패딩
        x_pad, (H_orig, W_orig), (pad_h, pad_w) = self._pad_to_stride(x, AUTO_PAD_STRIDE)

        # 2) SWT용 raw 이미지 준비 (de-normalize)
        raw_for_swt = None
        if self.use_swt and self.swt_attn is not None:
            with torch.no_grad():
                # [0,1] 근처로 클램핑해서 SWT에 입력
                raw_for_swt = self._denorm_input(x_pad).clamp(0.0, 1.0)

        # 3) encoder forward
        features: List[torch.Tensor] = self.encoder(x_pad)

        # 4) SWT Attention 적용 (선택된 stage 한 곳)
        if self.use_swt and self.swt_attn is not None and raw_for_swt is not None:
            idx = self.swt_stage_idx
            if 0 <= idx < len(features):
                features[idx] = self.swt_attn(raw_for_swt, features[idx])

        # 5) decoder & head
        dec_out: torch.Tensor = self.decoder(features)
        logits_pad: torch.Tensor = self.segmentation_head(dec_out)

        # 6) logits는 원본 H×W로 크롭 (CE/KD 바로 사용 가능)
        logits = self._crop_spatial(logits_pad, H_orig, W_orig)

        if not return_feats:
            return logits

        # 7) 스테이지 feat도 원본 공간에 대응되도록 크롭
        Hp, Wp = x_pad.shape[-2], x_pad.shape[-1]
        feats_out: List[torch.Tensor] = []
        for i in self.stage_indices:
            if i < 0 or i >= len(features):
                continue  # out-of-range 방지
            f = features[i]
            fh, fw = f.shape[-2], f.shape[-1]
            # stage stride 추정 (보통 정수: 4/8/16/32)
            sh = max(1, Hp // fh)
            sw = max(1, Wp // fw)
            Hf = math.ceil(H_orig / sh)
            Wf = math.ceil(W_orig / sw)
            f = self._crop_spatial(f, Hf, Wf)
            feats_out.append(f)

        return logits, feats_out


def get_backbone_channels(
    encoder_name: str = DEFAULT_ENCODER_NAME,
    encoder_weights: Union[str, None] = DEFAULT_ENCODER_WEIGHTS,
    in_channels: int = DEFAULT_IN_CHANNELS,
    stage_indices: Sequence[int] = DEFAULT_STAGE_INDICES,
) -> List[int]:
    enc = smp.encoders.get_encoder(
        encoder_name,
        in_channels=in_channels,
        depth=5,
        weights=encoder_weights,
    )
    out_ch = getattr(enc, "out_channels", [])
    return [out_ch[i] for i in stage_indices if 0 <= i < len(out_ch)]


def create_model(
    encoder_name: str = DEFAULT_ENCODER_NAME,
    encoder_weights: Union[str, None] = DEFAULT_ENCODER_WEIGHTS,
    in_channels: int = DEFAULT_IN_CHANNELS,
    classes: int = DEFAULT_NUM_CLASSES,
    stage_indices: Sequence[int] = DEFAULT_STAGE_INDICES,
    use_swt: bool = False,             # ← 실험 시 True로 켜기
    swt_stage_idx: int = 2,            # ← SWT 적용 stage (encoder.features index)
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
        use_swt=use_swt,
        swt_stage_idx=swt_stage_idx,
    )
