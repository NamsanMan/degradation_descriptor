# debug_pdm_save.py
from pathlib import Path
from PIL import Image
import os
import torch
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import config
# PDMPatchMix, TrainAugmentation을 data_loader에서 가져옴
from data_loader import PDMPatchMix, TrainAugmentation

# 역정규화 (모델 입력 텐서를 시각화용 RGB로 복원)
def denormalize_to_pil(t: torch.Tensor) -> Image.Image:
    # t: (3,H,W), normalized
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    x = t * std + mean
    x = x.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)

def save_side_by_side(img_left: Image.Image, img_right: Image.Image, out_path: Path):
    # 좌: 원본(또는 기준), 우: 변환본
    w = max(img_left.width, img_right.width)
    h = max(img_left.height, img_right.height)
    canvas = Image.new("RGB", (w * 2, h), (0, 0, 0))
    canvas.paste(img_left, (0, 0))
    canvas.paste(img_right, (w, 0))
    canvas.save(out_path)

def main(n_samples: int = 16):
    # 입력/출력 경로
    img_dir = config.DATA.TRAIN_IMG_DIR
    lbl_dir = config.DATA.TRAIN_LABEL_DIR
    out_root = Path(r"D:\LAB\result_files\analysis_results\30pdm_debug")
    out_A = out_root / "A_PDM_only"               # (A) PDM만 적용
    out_B = out_root / "B_model_input_view"       # (B) 전체 TrainAugmentation 후(역정규화)
    out_root.mkdir(parents=True, exist_ok=True); out_A.mkdir(exist_ok=True); out_B.mkdir(exist_ok=True)

    # 파일 목록
    files = [f for f in os.listdir(img_dir) if f.lower().endswith(".png")]
    files.sort()
    files = files[:n_samples]

    # PDM 변환기 & TrainAugmentation (기존 설정 재사용)
    pdm = PDMPatchMix(config.PDM)
    train_tf = TrainAugmentation(
        size=config.DATA.INPUT_RESOLUTION,
        pdm_transform=PDMPatchMix(config.PDM)  # 동일 설정으로 모델 입력과 동일한 경로 확인
    )

    for fname in files:
        stem = Path(fname).stem
        img_path = Path(img_dir) / fname
        lbl_path = Path(lbl_dir) / fname  # 라벨 파일명 동일 가정

        # 원본(HR)
        img_hr = Image.open(img_path).convert("RGB")
        # (A) PDM만 적용 (리사이즈/기타 증강 전) + 원본과 나란히 저장
        img_pdm = pdm(img_hr.copy())
        save_side_by_side(
            img_left=img_hr, img_right=img_pdm,
            out_path=out_A / f"{stem}_A_hr_vs_pdm.png"
        )

        # (B) 실제 모델 입력 경로: TrainAugmentation 전체 적용 → 역정규화하여 저장
        #    (GT를 함께 넣어야 하므로 라벨을 읽어 전달)
        if lbl_path.exists():
            mask = Image.open(lbl_path)
        else:
            # 라벨이 없다면 더미로 동일 해상도 제로 배열
            mask = Image.fromarray(np.zeros((img_hr.height, img_hr.width), dtype=np.uint8))

        img_tensor, _ = train_tf(img_hr.copy(), mask)  # 최종 모델 입력과 동일한 변환(정규화 포함)
        img_vis = denormalize_to_pil(img_tensor)       # 역정규화로 보기 좋게 변환
        img_vis.save(out_B / f"{stem}_B_model_input.png")

    print(f"[OK] Saved:\n - (A) PDM-only -> {out_A}\n - (B) model-input-view -> {out_B}")

if __name__ == "__main__":
    # 샘플 수는 필요에 따라 조정
    main(n_samples=16)
