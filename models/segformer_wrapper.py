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
        # encoder hidden states 항상 반환 (stage feature + patch embedding용)
        cfg.output_hidden_states = True

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            src, config=cfg, ignore_mismatched_sizes=True
        )

        # 마지막 forward에서 사용된 stage별 feature / embedding 저장용
        self._last_encoder_feats = None   # Tuple[Tensor(B,C,H,W), ...] 길이 4
        self._last_encoder_embeds = None  # Tuple[Tensor(B,N,C), ...]   길이 4

    # --------- 내부 유틸 ---------
    def _reshape_hidden_state(self, feat: torch.Tensor, idx: int, hw):
        """
        SegFormer encoder hidden state (N, HW, C) → (N, C, H, W)
        - feat: (B, HW, C) 또는 (B, C, H, W)
        - idx: stage index (0~3)
        - hw: 입력 이미지 해상도 (H_in, W_in)
        """
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
            # fallback (4, 8, 16, 32, ...)
            stride = 2 ** (idx + 2)

        h = max(1, math.ceil(h_in / stride))
        w = max(1, math.ceil(w_in / stride))
        if h * w != n:
            # token 수와 맞지 않으면 가능한 한 정사각형에 가깝게 reshape
            h = max(1, int(round(math.sqrt(n))))
            w = max(1, n // h)
            if h * w != n:
                raise RuntimeError(
                    f"Cannot reshape hidden state of length {n} to 2D map (stride idx {idx})"
                )

        feat = feat.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return feat

    def _collect_encoder_feats_and_embeds(self, out, input_hw):
        """
        encoder hidden states에서
          - stage 4개 raw embedding: (B, N, C) -> embeds
          - stage 4개 feature map:  (B, C, H, W) -> feats
        를 동시에 추출해서 저장.
        """
        feats = getattr(out, "encoder_hidden_states", None)
        if feats is None:
            feats = getattr(out, "hidden_states", None)
        if feats is None:
            self._last_encoder_feats = None
            self._last_encoder_embeds = None
            return None, None

        # hidden_states는 (B, HW, C) 또는 (B, C, H, W) 형태의 stage들이 들어있는 tuple/list
        # SegFormer의 마지막 4개 stage만 사용
        feats = feats[-4:]

        # patch embedding (B, N, C) 그대로 사용
        embeds = tuple(f if f.dim() == 3 else f.flatten(2).transpose(1, 2) for f in feats)

        # feature map (B, C, H, W)로 reshape
        reshaped = tuple(
            self._reshape_hidden_state(feat, idx, input_hw) for idx, feat in enumerate(feats)
        )

        self._last_encoder_feats = reshaped
        self._last_encoder_embeds = embeds
        return reshaped, embeds

    def get_last_encoder_features(self):
        """최근 forward에서 추출한 encoder stage feature (B,C,H,W) 튜플을 반환."""
        return self._last_encoder_feats

    def get_last_encoder_embeddings(self):
        """최근 forward에서 추출한 encoder stage patch embedding (B,N,C) 튜플을 반환."""
        return self._last_encoder_embeds

    def forward(
        self,
        x: torch.Tensor,
        return_feats: bool = False,
        return_embeds: bool = False,
    ):
        """
        기본 동작:
          - return_feats=False, return_embeds=False: logits만 반환 (기존과 동일)
          - return_feats=True, return_embeds=False: (logits, feats)
          - return_feats=True, return_embeds=True: (logits, feats, embeds)
          - return_feats=False, return_embeds=True: (logits, embeds)
        """
        # HuggingFace SegFormer: output_hidden_states=True 설정되어 있음
        out = self.model(pixel_values=x, return_dict=True)

        # decoder output (B, num_classes, H_dec, W_dec) -> 입력 해상도로 upsample
        logits = F.interpolate(
            out.logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        feats, embeds = self._collect_encoder_feats_and_embeds(out, x.shape[-2:])

        # 기존 코드와의 호환: return_feats=True만 쓰는 경우 (logits, feats) 반환
        if not return_feats and not return_embeds:
            return logits

        if return_feats and not return_embeds:
            if feats is None:
                raise RuntimeError("SegFormer hidden_states를 얻지 못했습니다.")
            return logits, feats

        if not return_feats and return_embeds:
            if embeds is None:
                raise RuntimeError("SegFormer hidden_states를 얻지 못했습니다.")
            return logits, embeds

        # 둘 다 True 인 경우
        if feats is None or embeds is None:
            raise RuntimeError("SegFormer hidden_states를 얻지 못했습니다.")
        return logits, feats, embeds
