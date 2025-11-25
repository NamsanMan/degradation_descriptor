import torch
import math
import torch.nn.functional as F
from models.d3p import create_model, AUTO_PAD_STRIDE, PAD_MODE
from models.segformer_wrapper import SegFormerWrapper


def check_segformer_encoder_stages():
    """
    SegFormerWrapper 모델을 생성하고 encoder의 stage별
    출력 feature map의 크기를 확인합니다.
    """
    print("--- SegFormer 모델 생성 및 인코더 분석 시작 ---")

    # 1. SegFormerWrapper 클래스를 직접 사용하여 모델을 생성합니다.
    #    name 인자에 원하는 모델 버전('segformerb0', 'segformerb1' 등)을 전달합니다.
    try:
        model = SegFormerWrapper(name="segformerb1")
    except ImportError as e:
        print(f"Error: {e}")
        print("config.py를 찾을 수 없습니다. SegFormerWrapper 초기화에 config가 필요할 수 있습니다.")
        return

    model.eval()  # 추론 모드로 설정

    # 2. 동일한 크기 (1, 3, 360, 480)의 더미 입력 데이터를 생성합니다.
    dummy_input = torch.randn(1, 3, 360, 480)
    print(f"\n[1] 더미 입력(Input) 생성 완료")
    print(f"    - 크기: {dummy_input.shape}")

    # 3. 모델의 forward에 입력을 통과시켜 logits와 feature 리스트를 얻습니다.
    #    SegFormer는 내부 패딩이 필요 없으며, return_feats=True로 특징을 얻습니다.
    with torch.no_grad():
        # model.encoder()가 아닌, model()을 직접 호출해야 합니다.
        logits, features = model(dummy_input, return_feats=True)

    # 4. 결과 분석 및 출력
    print(f"\n[2] 인코더(SegFormer MiT-B1) 분석 결과")
    print("-" * 50)
    # SegFormerWrapper는 encoder의 마지막 4개 stage 출력을 반환합니다.
    print(f"총 Stage(Feature Map) 개수: {len(features)}")
    print("각 Stage 별 출력 Feature Map 크기:")

    for i, f in enumerate(features):
        # Stage 번호를 1부터 시작하도록 i+1로 표기
        print(f"  - Stage {i + 1}: {list(f.shape)}")
    print("-" * 50)

def check_encoder_stages():
    """
    DeepLabV3PlusWrapper 모델을 생성하고 encoder의 stage 수와
    각 stage별 출력 feature map의 크기를 확인합니다.
    """
    print("--- 모델 생성 및 인코더 분석 시작 ---")

    # 1. d3p.py의 create_model 함수를 사용하여 모델을 생성합니다.
    #    기본 설정 (encoder_name='mobilenet_v2', etc.)을 사용합니다.
    model = create_model()
    model.eval()  # 추론 모드로 설정

    # 2. 명시된 크기 (1, 3, 360, 480)의 더미 입력 데이터를 생성합니다.
    #    (Batch, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, 360, 480)
    print(f"\n[1] 원본 더미 입력(Input) 생성 완료")
    print(f"    - 크기: {dummy_input.shape}")

    # 3. 모델의 forward 패스와 동일하게 내부 패딩을 적용합니다.
    #    Encoder는 패딩된 입력을 받으므로, 정확한 출력을 위해 이 과정을 거칩니다.
    B, C, H, W = dummy_input.shape
    Ht = math.ceil(H / AUTO_PAD_STRIDE) * AUTO_PAD_STRIDE
    Wt = math.ceil(W / AUTO_PAD_STRIDE) * AUTO_PAD_STRIDE
    pad_h = Ht - H
    pad_w = Wt - W
    padded_input = F.pad(dummy_input, (0, pad_w, 0, pad_h), mode=PAD_MODE)
    print(f"\n[2] 내부 패딩(Internal Padding) 적용 완료")
    print(f"    - Stride '{AUTO_PAD_STRIDE}'의 배수로 패딩됨")
    print(f"    - 패딩 후 크기: {padded_input.shape}")

    # 4. 모델의 encoder에 패딩된 입력을 통과시켜 feature 리스트를 얻습니다.
    #    torch.no_grad()를 사용하여 불필요한 연산을 방지합니다.
    with torch.no_grad():
        # Encoder를 직접 호출합니다.
        features = model.encoder(padded_input)

    # 5. 결과 분석 및 출력
    print(f"\n[3] 인코더({model.encoder.__class__.__name__}) 분석 결과")
    print("-" * 50)
    # segmentation_models_pytorch의 MobileNetV2Encoder는 depth=5일 때
    # 원본 입력 + 5개의 다운샘플링된 feature map, 총 6개를 반환합니다.
    print(f"총 Stage(Feature Map) 개수: {len(features)}")
    print("각 Stage 별 출력 Feature Map 크기:")

    for i, f in enumerate(features):
        print(f"  - Stage {i}: {list(f.shape)}")
    print("-" * 50)
if __name__ == "__main__":
    check_segformer_encoder_stages()
    check_encoder_stages()