import os
from pathlib import Path
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# descriptor.py에서 필요한 클래스 임포트
from descriptor import DegradationDescriptorNet, FrequencyDegradationTarget, interpret_scores

# =========================
# 1. 경로 및 설정
# =========================

# 학습된 모델 가중치 경로 (main_descriptor.py의 SAVE_PATH와 일치)
CKPT_PATH = Path(r"D:\LAB\result_files\des_module2\descriptor_net.pth")

# Test 데이터셋 경로 (A_set 기준)
HR_TEST_DIR = Path(r"D:\LAB\datasets\project_use\CamVid_12_2Fold_v4\B_set\test\images")
LR_TEST_DIR = Path(r"D:\LAB\datasets\project_use\CamVid_12_2Fold_LR_x4_Bilinear\B_set\test\images")

# 결과 저장 경로
OUT_DIR = Path(r"D:\LAB\result_files\des_module2\test_evaluation_vis")

# 입력 크기 (학습 시와 동일해야 함)
INPUT_SIZE: Tuple[int, int] = (360, 480)

# 시각화할 샘플 수 (0이면 전체 수행하지만 시각화 저장은 안 함, -1이면 전체 저장)
VIS_SAMPLES: int = 20

# =========================
# 2. 유틸리티 함수
# =========================

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(IMG_EXTENSIONS)


def load_checkpoint_model(ckpt_path: Path, device: torch.device) -> DegradationDescriptorNet:
    print(f"[Load] Loading model from {ckpt_path} ...")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=device)

    model = DegradationDescriptorNet(in_channels=3, num_bands=3).to(device)
    state_dict = ckpt.get("model_state", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_transform(size: Tuple[int, int]):
    return transforms.Compose([
        transforms.Resize(size, interpolation=Image.BILINEAR),
        transforms.ToTensor()
    ])


def save_comparison_overlay(
        img_tensor: torch.Tensor,
        pred: torch.Tensor,
        target: torch.Tensor,
        out_path: Path
):
    """
    LR 이미지 위에 Pred vs GT 점수를 오버레이하여 저장
    """
    # Tensor -> Numpy Image
    img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    plt.figure(figsize=(8, 6))
    plt.imshow(img_np)
    plt.axis("off")

    # 텍스트 포맷팅
    # Yellow: Prediction, Cyan: Ground Truth
    info_text = (
        f"PRED | L:{pred[0]:.3f} M:{pred[1]:.3f} H:{pred[2]:.3f}\n"
        f"GT   | L:{target[0]:.3f} M:{target[1]:.3f} H:{target[2]:.3f}\n"
        f"Diff | L:{abs(pred[0] - target[0]):.3f} M:{abs(pred[1] - target[1]):.3f} H:{abs(pred[2] - target[2]):.3f}"
    )

    plt.text(
        10, 25, info_text,
        fontsize=12, color="white", fontfamily="monospace", fontweight="bold",
        bbox=dict(facecolor="black", alpha=0.6, edgecolor="none")
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close()


# =========================
# 3. 메인 평가 루프
# =========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # 1. 모델 및 GT 생성기 로드
    model = load_checkpoint_model(CKPT_PATH, device)
    # 학습 때 사용한 파라미터와 동일해야 공정한 평가 가능
    target_builder = FrequencyDegradationTarget(num_bands=3, low_cut=0.15, mid_cut=0.5)

    # 2. 파일 리스트 매칭
    if not HR_TEST_DIR.exists() or not LR_TEST_DIR.exists():
        raise RuntimeError(f"Test directories not found.\nHR: {HR_TEST_DIR}\nLR: {LR_TEST_DIR}")

    hr_files = sorted([f for f in os.listdir(HR_TEST_DIR) if is_image_file(f)])
    lr_files = sorted([f for f in os.listdir(LR_TEST_DIR) if is_image_file(f)])

    # 교집합 확인
    common_files = sorted(list(set(hr_files).intersection(set(lr_files))))
    print(f"[Data] Found {len(common_files)} test pairs.")

    if len(common_files) == 0:
        print("[Error] No matching files found. Check filenames.")
        return

    # 3. 평가 실행
    tf = build_transform(INPUT_SIZE)

    mae_sum = torch.zeros(3, device=device)  # Mean Absolute Error Accumulator
    mse_sum = torch.zeros(3, device=device)  # Mean Squared Error Accumulator

    print("\n[Start] Evaluation on Test Set...")

    with torch.no_grad():
        for idx, fname in enumerate(common_files):
            # 이미지 로드
            hr_path = HR_TEST_DIR / fname
            lr_path = LR_TEST_DIR / fname

            hr_img = Image.open(hr_path).convert("RGB")
            lr_img = Image.open(lr_path).convert("RGB")

            hr_tensor = tf(hr_img).unsqueeze(0).to(device)  # [1,3,H,W]
            lr_tensor = tf(lr_img).unsqueeze(0).to(device)  # [1,3,H,W]

            # GT 계산 (Target)
            target = target_builder(hr_tensor, lr_tensor)  # [1,3]

            # 모델 예측 (Prediction)
            pred = model(lr_tensor)  # [1,3]

            # Error 계산
            diff = torch.abs(pred - target)
            mae_sum += diff[0]
            mse_sum += diff[0] ** 2

            # 시각화 (설정된 개수만큼만 저장)
            if idx < VIS_SAMPLES:
                out_path = OUT_DIR / f"eval_{fname}"
                save_comparison_overlay(lr_tensor[0], pred[0].cpu(), target[0].cpu(), out_path)
                print(f"  [{idx + 1}/{len(common_files)}] Processed & Saved: {fname}")
            elif idx % 50 == 0:
                print(f"  [{idx + 1}/{len(common_files)}] Processed...")

    # 4. 최종 결과 출력
    n_samples = len(common_files)
    mae_avg = (mae_sum / n_samples).cpu().numpy()
    mse_avg = (mse_sum / n_samples).cpu().numpy()
    rmse_avg = np.sqrt(mse_avg)

    print("\n" + "=" * 40)
    print("   FINAL TEST EVALUATION REPORT   ")
    print("=" * 40)
    print(f"Total Samples : {n_samples}")
    print(f"Weights Path  : {CKPT_PATH.name}")
    print("-" * 40)
    print(f"{'Band':<10} | {'MAE (L1)':<10} | {'RMSE':<10}")
    print("-" * 40)
    print(f"{'Low':<10} | {mae_avg[0]:.5f}    | {rmse_avg[0]:.5f}")
    print(f"{'Mid':<10} | {mae_avg[1]:.5f}    | {rmse_avg[1]:.5f}")
    print(f"{'High':<10} | {mae_avg[2]:.5f}    | {rmse_avg[2]:.5f}")
    print("-" * 40)
    print(f"Avg All    | {mae_avg.mean():.5f}    | {rmse_avg.mean():.5f}")
    print("=" * 40)
    print(f"[Visual Results] Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()