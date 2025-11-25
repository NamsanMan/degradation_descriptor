# models/segformer_wrapper.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation
from config import DATA

_SOURCES = {
    "segformerb0": "nvidia/mit-b0",
    "segformerb1": "nvidia/mit-b1",
    "segformerb2": "nvidia/mit-b2",
    "segformerb3": "nvidia/mit-b3",
    "segformerb4": "nvidia/mit-b4",
    "segformerb5": "nvidia/mit-b5",
}

class SegFormerWrapper(nn.Module):
    def __init__(self, name: str, num_classes: int = DATA.NUM_CLASSES):
        super().__init__()
        name = name.lower()
        assert name in _SOURCES, f"Unknown SegFormer name: {name}"
        src = _SOURCES[name]

        cfg = SegformerConfig.from_pretrained(src)
        cfg.num_labels = num_classes
        cfg.id2label = {i: n for i, n in enumerate(DATA.CLASS_NAMES)}
        cfg.label2id = {n: i for i, n in enumerate(DATA.CLASS_NAMES)}
        cfg.output_hidden_states = True  # 항상 stage feats 반환한다 >> KD시 필요

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            src, config=cfg, ignore_mismatched_sizes=True
        )
        self._last_encoder_feats = None

    # --------- 내부 유틸 ---------
    def _reshape_hidden_state(self, feat: torch.Tensor, idx: int, hw):
        """SegFormer encoder hidden state (N, HW, C) → (N, C, H, W)."""
        if feat.dim() == 4:
            return feat

        if feat.dim() != 3:
            raise RuntimeError(
                f"SegFormer hidden state must be 3D/4D, got shape {tuple(feat.shape)}"
            )

        b, n, c = feat.shape
        h_in, w_in = hw

        ratios = getattr(self.model.config, "downsample_ratios", None)
        strides = getattr(self.model.config, "encoder_stride", None)
        if ratios and idx < len(ratios):
            stride = ratios[idx]
        elif strides and idx < len(strides):
            stride = strides[idx]
        else:
            stride = 2 ** (idx + 2)  # 합리적인 fallback (4,8,16,32,...)

        h = max(1, math.ceil(h_in / stride))
        w = max(1, math.ceil(w_in / stride))
        if h * w != n:
            # token 수와 맞지 않으면 가능한 정사각형 근사 사용
            h = max(1, int(round(math.sqrt(n))))
            w = max(1, n // h)
            if h * w != n:
                raise RuntimeError(
                    f"Cannot reshape hidden state of length {n} to 2D map (stride idx {idx})"
                )

        feat = feat.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return feat

    def _collect_encoder_feats(self, out, input_hw):
        feats = getattr(out, "encoder_hidden_states", None)
        if feats is None:
            feats = getattr(out, "hidden_states", None)
        if feats is None:
            return None
        feats = feats[-4:]
        reshaped = tuple(
            self._reshape_hidden_state(feat, idx, input_hw) for idx, feat in enumerate(feats)
        )
        self._last_encoder_feats = reshaped
        return reshaped

    def get_last_encoder_features(self):
        """최근 forward에서 추출한 encoder stage feature를 반환."""
        return self._last_encoder_feats

    def forward(self, x, return_feats: bool = False):
        # 위에서 이미 output_hidden_states=True 이므로 인자 없이 호출
        out = self.model(pixel_values=x, return_dict=True)
        logits = F.interpolate(out.logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

        feats = self._collect_encoder_feats(out, x.shape[-2:])

        # KD용 feature 반환 >> return_feats = True 일때 logit과 4개의 텐서로 4개의 feature map 반환
        if return_feats:
            if feats is None:
                raise RuntimeError("SegFormer hidden_states를 얻지 못했습니다.")
            return logits, feats
        return logits
