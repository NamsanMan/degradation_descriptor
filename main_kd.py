"""


Train 부터 test 후 시각화 까지 end-to-end를 위한 main.py


"""

import os
import random
import torch
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image

import config

def set_seed(seed):
    """
    재현성을 위해 시드를 고정하는 함수
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    # CuDNN 결정론적 연산 활성화
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"▶ Seed is fixed to {seed}")

set_seed(config.GENERAL.SEED)

import train_kd
import data_loader
import evaluate

def decode_segmap(label_mask):
    """
    label_mask: 2D numpy array (H×W), 값은 [0..n_classes-1]
    return: 3D numpy array (H×W×3), dtype=uint8
    """
    # class_colors[label_mask] 형태로 브로드캐스트 매핑
    return config.DATA.CLASS_COLORS[label_mask]

def main():
    # 랜덤 이미지 시각화 할때는 seed 고정 영향 안받게 함
    visual_random = random.Random()

    if config.TRAIN.USE_CHECKPOINT:
        ##### train 안하고 checkpoint만 로드할 때 ######
        checkpoint_name = 'best_model.pth'
        best_ckpt = config.GENERAL.BASE_DIR / checkpoint_name
        if not best_ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {best_ckpt}")
    else:
        # 1) 학습 수행
        # train.py의 run_training을 호출하여, 가장 좋은 모델 체크포인트 경로를 반환받음
        best_ckpt = train_kd.run_training(num_epochs=config.TRAIN.EPOCHS)

    ##### train 안하고 checkpoint만 로드할 때 ######
    checkpoint_name = 'best_model.pth'
    best_ckpt = config.GENERAL.BASE_DIR / checkpoint_name
    if not best_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {best_ckpt}")

    # 2) test를 위해 베스트 체크포인트 로드(student) (모델 정보와 epoch 정보만 불러온다, 학습 재개를 위한다면 코드 수정필요)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train_kd.student
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    epoch_num = ckpt["epoch"]
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # 3) 테스트셋 전체에 대해 mIoU / Pixel Accuracy 계산
    metrics = evaluate.evaluate_all(model, data_loader.test_loader, device)
    test_miou = metrics["mIoU"]
    test_pa   = metrics["PixelAcc"]
    print(f"▶ Loaded model from epoch {epoch_num}, Test mIoU: {test_miou:.4f}, Test Pixel Acc: {test_pa:.4f}")

    per_cls_iou = metrics["per_class_iou"]
    df_test_iou = pd.DataFrame({
        "Class": config.DATA.CLASS_NAMES,
        "IoU": per_cls_iou
    })
    test_iou_path = config.GENERAL.LOG_DIR / f"test_iou_epoch_{epoch_num}.csv"
    df_test_iou.to_csv(test_iou_path, index=False)

    # 3.2) 테스트셋 confusion matrix 계산 및 저장
    cm_test = metrics["confusion_matrix"]
    plt.figure(figsize=(8, 6))
    plt.imshow(cm_test, interpolation='nearest')
    plt.title(f"Test Confusion Matrix (Epoch {epoch_num})")
    plt.colorbar()
    tick_marks = np.arange(config.DATA.NUM_CLASSES)
    plt.xticks(tick_marks, config.DATA.CLASS_NAMES, rotation=45, ha='right')
    plt.yticks(tick_marks, config.DATA.CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    test_cm_path = config.GENERAL.LOG_DIR / f"test_confusion_matrix_epoch_{epoch_num}.png"
    plt.savefig(str(test_cm_path), bbox_inches="tight")
    plt.close()

    # 4) 결과 파일에 기록
    results_txt = config.GENERAL.BASE_DIR / "results.txt"
    os.makedirs(results_txt.parent, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(results_txt, "a", encoding="utf-8") as f:
        f.write(f"[{now}] Test mIoU: {test_miou:.4f}, Test PA: {test_pa:.4f}\n")

    # 5) 랜덤 5장 시각화 (기존 로직 유지)
    input_dir  = config.DATA.TEST_IMG_DIR
    mask_dir   = config.DATA.TEST_LABEL_DIR
    output_dir = config.GENERAL.BASE_DIR / "images"
    os.makedirs(output_dir, exist_ok=True)

    all_imgs = [f for f in os.listdir(input_dir) if f.endswith(".png")]
    k = min(5, len(all_imgs))
    for filename in visual_random.sample(all_imgs, k):
        img_path  = os.path.join(str(input_dir), filename)
        img       = Image.open(img_path).convert('RGB')
        mask_path = os.path.join(str(mask_dir), filename)
        mask      = Image.open(mask_path)

        img_t, mask_t  = data_loader.SegmentationTransform(config.DATA.INPUT_RESOLUTION)(img, mask)

        #예측 진행
        with torch.no_grad():
            pred_logits = model(img_t.unsqueeze(0).to(device))
            pred_idx     = pred_logits.argmax(dim=1).squeeze().cpu().numpy()

        img_np = np.array(img)
        mask_idx = mask_t.numpy()

        # 디코딩
        mask_rgb = decode_segmap(mask_idx)
        pred_rgb = decode_segmap(pred_idx)

        H, W, _ = img_np.shape
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(img_np)
        axes[0].set_title(f"Original ({W}x{H})")
        axes[0].axis("off")
        axes[0].grid(False)

        axes[1].imshow(mask_rgb)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
        axes[1].grid(False)

        axes[2].imshow(pred_rgb)
        axes[2].set_title("Prediction")
        axes[2].axis("off")
        axes[2].grid(False)

        plt.tight_layout()  # 서브플롯 간 간격을 자동으로 조절합니다.

        save_path = output_dir / f"viz_{Path(filename).stem}.png"
        fig.savefig(str(save_path), bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {save_path}")

if __name__ == "__main__":
    main()
