import torch
import numpy as np
from tqdm import tqdm

# ================================================================
#               사용자 설정 (이 부분을 수정하세요)
# ================================================================
# 'deeplab' 또는 'segformer' 중 하나를 선택하여 분석할 모델을 지정합니다.
MODEL_TO_PROFILE = 'segformer'  # <--- 여기를 'segformer'로 바꿔서 다시 실행

# 훈련 시와 유사한 배치 사이즈를 사용합니다.
BATCH_SIZE = 4
# ================================================================

# 모델과 관련 부품들을 불러옵니다.
if MODEL_TO_PROFILE == 'deeplab':
    # DeepLabV3+의 훈련 스크립트에서 모델, 손실 함수, 옵티마이저를 가져옵니다.
    from no_use.train import model, criterion, optimizer

    print("DeepLabV3+ (ResNet50) 모델을 프로파일링합니다.")
elif MODEL_TO_PROFILE == 'segformer':
    # SegFormer의 훈련 스크립트에서 모델, 손실 함수, 옵티마이저를 가져옵니다.
    # 파일 이름이 다르다면 `train_transformers`를 실제 파일 이름으로 바꿔주세요.
    from train_transformers import model, criterion, optimizer

    print("SegFormer (MiT-B1) 모델을 프로파일링합니다.")
else:
    raise ValueError("MODEL_TO_PROFILE은 'deeplab' 또는 'segformer'여야 합니다.")


def profile_training_step(warmup_steps=10, measure_steps=50):
    """훈련의 한 스텝을 4단계로 나누어 각 단계의 소요 시간을 측정합니다."""

    assert torch.cuda.is_available(), "CUDA(GPU)가 필요합니다."
    device = torch.device("cuda")

    # 모델을 훈련 모드로 설정하고 GPU로 보냅니다.
    model.train().to(device)

    # 더미 입력 데이터와 타겟 데이터를 생성합니다.
    # 실제 데이터로더의 이미지 크기와 클래스 수에 맞춰주세요.
    input_size = (BATCH_SIZE, 3, 368, 480)
    num_classes = 12
    inputs = torch.randn(input_size, device=device, dtype=torch.float32)
    targets = torch.randint(0, num_classes, (input_size[0], input_size[2], input_size[3]), device=device,
                            dtype=torch.long)

    # 정확한 시간 측정을 위한 CUDA 이벤트를 생성합니다.
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # 각 단계별 시간을 저장할 리스트를 초기화합니다.
    forward_times = []
    loss_times = []
    backward_times = []
    optimizer_times = []

    print(f"준비 운동(Warm-up)을 {warmup_steps}번 실행합니다...")
    for _ in range(warmup_steps):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"본 측정을 {measure_steps}번 실행합니다...")
    for _ in tqdm(range(measure_steps)):
        # 1. Forward Pass 시간 측정
        torch.cuda.synchronize()
        starter.record()
        outputs = model(inputs)
        ender.record()
        torch.cuda.synchronize()
        forward_times.append(starter.elapsed_time(ender))

        # 2. Loss 계산 시간 측정
        torch.cuda.synchronize()
        starter.record()
        loss = criterion(outputs, targets)
        ender.record()
        torch.cuda.synchronize()
        loss_times.append(starter.elapsed_time(ender))

        # 3. Backward Pass 시간 측정
        torch.cuda.synchronize()
        starter.record()
        loss.backward()
        ender.record()
        torch.cuda.synchronize()
        backward_times.append(starter.elapsed_time(ender))

        # 4. Optimizer 업데이트 시간 측정
        torch.cuda.synchronize()
        starter.record()
        optimizer.step()
        ender.record()
        torch.cuda.synchronize()
        optimizer_times.append(starter.elapsed_time(ender))

        # 다음 스텝을 위해 그래디언트를 초기화합니다.
        optimizer.zero_grad()

    # 결과 분석 및 출력
    fwd_ms = np.mean(forward_times)
    loss_ms = np.mean(loss_times)
    bwd_ms = np.mean(backward_times)
    optim_ms = np.mean(optimizer_times)
    total_ms = fwd_ms + loss_ms + bwd_ms + optim_ms

    print("\n" + "=" * 50)
    print(f"             {MODEL_TO_PROFILE.upper()} 훈련 스텝 시간 분석")
    print("=" * 50)
    print(f"  Forward Pass       : {fwd_ms:8.3f} ms ({fwd_ms / total_ms * 100:5.1f} %)")
    print(f"  Loss Calculation   : {loss_ms:8.3f} ms ({loss_ms / total_ms * 100:5.1f} %)")
    print(f"  Backward Pass      : {bwd_ms:8.3f} ms ({bwd_ms / total_ms * 100:5.1f} %)")
    print(f"  Optimizer Step     : {optim_ms:8.3f} ms ({optim_ms / total_ms * 100:5.1f} %)")
    print("-" * 50)
    print(f"  Total Step Time    : {total_ms:8.3f} ms (100.0 %)")
    print("=" * 50)


if __name__ == "__main__":
    profile_training_step()