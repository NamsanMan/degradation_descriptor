import torch
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation
import config  # config.py 파일 import


# ──────────────────────────────────────────────────────────────────
# 1. 모델 아키텍처 정의 (Model Architecture Definition)
# ──────────────────────────────────────────────────────────────────

SEGFORMER_SOURCE = "nvidia/mit-b4"

class SegFormerWrapper(torch.nn.Module):
    def __init__(self, num_classes=config.DATA.NUM_CLASSES):
        super().__init__()
        # 1) Hugging Face feature extractor 로드
        cfg = SegformerConfig.from_pretrained(SEGFORMER_SOURCE)
        # 2) 클래스 수 재설정
        cfg.num_labels = num_classes
        cfg.id2label = {i: name for i, name in enumerate(config.DATA.CLASS_NAMES)}
        cfg.label2id = {name: i for i, name in enumerate(config.DATA.CLASS_NAMES)}
        """
        # 허깅페이스에서 불러오는 모델의 가중치를 다운로드/로딩하지 않는다. 모델(인코더 + 디코더/헤드)만 생성하고 모든 파라미터는 초기화. 즉 모든 pretrain 옵션을 불러오지 않는다
        self.model = SegformerForSemanticSegmentation(config=cfg)
        """
        # 허깅페이스에서 불러오는 체크포인트의 가중치를 불러옴. pretrain 가중치를 불러온다
        # SegformerForSemanticSegmentation가 디코더/헤드 "아키텍쳐"를 생성한다
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            SEGFORMER_SOURCE,
            config=cfg,
            ignore_mismatched_sizes=True
        )

    def forward(self, x, return_feats: bool=False):
        """
        return:
          - return_feats=False: upsampled logits (B, C, H, W)
          - return_feats=True : (upsampled logits, features[4]) where features are tuples of 4 tensors
                                 each (B, C_i, H_i, W_i) from encoder stages.
        """
        # x: (B,3,H,W) 정규화된 tensor
        out = self.model(pixel_values=x, output_hidden_states=return_feats, return_dict=True)         # (B, num_classes, h', w')
        # 입력 크기 (H,W)로 bilinear upsample
        logits = F.interpolate(out.logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if return_feats:
            # huggingface는 hidden_states(=encoder stage별 feature) 튜플을 제공합니다.
            feats = out.hidden_states  # tuple length 4, each (B, C_i, H_i, W_i)
            return logits, feats
        else:
            return logits

