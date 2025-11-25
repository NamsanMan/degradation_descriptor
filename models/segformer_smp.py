"""
나중에 혹시 CNN-Tansformer Hybrid network 사용해 보고 싶을때 해보려고 만들어놓음
"""

import segmentation_models_pytorch as smp

from config import DATA

ENCODER_NAME = "mit_b0"
ENCODER_WEIGHTS = "imagenet"


def create_segformer_smp(encoder_name = ENCODER_NAME, encoder_weights = ENCODER_WEIGHTS, in_channels = 3, classes = DATA.NUM_CLASSES):
    model = smp.Segformer(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes
    )
    return model