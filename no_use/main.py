import os
import random
import torch
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image

import train
import data_loader

class_names = train.class_names
log_dir     = train.log_dir

class_colors = np.array([
    [128, 128, 128],  # Sky
    [128,   0,   0],  # Building
    [192, 192, 128],  # Pole
    [128,  64, 128],  # Road
    [  0,   0, 192],  # Sidewalk
    [128, 128,   0],  # Tree
    [192, 128, 128],  # SignSymbol
    [ 64,  64, 128],  # Fence
    [ 64,   0, 128],  # Car
    [ 64,  64,   0],  # Pedestrian
    [  0, 128, 192],  # Bicyclist
    [  0,   0,   0],  # Void
], dtype=np.uint8)

def decode_segmap(label_mask):
    """
    label_mask: 2D numpy array (H×W), 값은 [0..n_classes-1]
    return: 3D numpy array (H×W×3), dtype=uint8
    """
    # class_colors[label_mask] 형태로 브로드캐스트 매핑
    return class_colors[label_mask]

def main():
    # 1) 학습 수행
    # train.py의 run_training을 호출하여, 가장 좋은 모델 체크포인트 경로를 반환받음
    best_ckpt = train.run_training(num_epochs=train.train_epoch)

    ##### train 안하고 checkpoint만 로드할 때 ######
    """
    checkpoint_name = 'best_model.pth'
    best_ckpt = train_transformers.base_dir / checkpoint_name
    if not best_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {best_ckpt}")
    """

    # 2) 베스트 체크포인트 로드
    model  = train.model
    device = train.device
    ckpt = torch.load(best_ckpt, map_location=device)
    best_epoch = ckpt["epoch"]
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # 3) 테스트셋 전체에 대해 mIoU / Pixel Accuracy 계산
    test_miou = train.compute_dataset_iou(data_loader.test_loader)
    test_pa   = train.compute_dataset_pa(data_loader.test_loader)
    print(f"▶ Loaded model from epoch {best_epoch}, Test mIoU: {test_miou:.4f}, Test Pixel Acc: {test_pa:.4f}")

    per_cls_iou = train.compute_per_class_iou(data_loader.test_loader)
    df_test_iou = pd.DataFrame({
        "Class": class_names,
        "IoU": per_cls_iou
    })
    test_iou_path = log_dir / f"test_iou_epoch_{best_epoch}.csv"
    df_test_iou.to_csv(test_iou_path, index=False)

    # 3.2) 테스트셋 confusion matrix 계산 및 저장
    cm_test = train.compute_confusion_matrix(data_loader.test_loader)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm_test, interpolation='nearest')
    plt.title(f"Test Confusion Matrix (Epoch {best_epoch})")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    test_cm_path = log_dir / f"test_confusion_matrix_epoch_{best_epoch}.png"
    plt.savefig(str(test_cm_path), bbox_inches="tight")
    plt.close()

    # 4) 결과 파일에 기록
    results_txt = train.base_dir / "results.txt"
    os.makedirs(results_txt.parent, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(results_txt, "a", encoding="utf-8") as f:
        f.write(f"[{now}] Test mIoU: {test_miou:.4f}, Test PA: {test_pa:.4f}\n")

    # 5) 랜덤 5장 시각화 (기존 로직 유지)
    input_dir  = data_loader.test_dir / "images"
    mask_dir   = data_loader.test_dir / "labels"
    output_dir = train.base_dir / "images"
    os.makedirs(output_dir, exist_ok=True)

    all_imgs = [f for f in os.listdir(input_dir) if f.endswith(".png")]
    for filename in random.sample(all_imgs, 5):
        img_path  = os.path.join(str(input_dir), filename)
        img       = Image.open(img_path).convert('RGB')
        mask_path = os.path.join(str(mask_dir), filename)
        mask      = Image.open(mask_path)

        img_t, mask_t  = data_loader.SegmentationTransform(data_loader.input_resolution)(img, mask)

        #예측 진행
        with torch.no_grad():
            pred_logits = model(img_t.unsqueeze(0).to(device))
            pred_idx     = pred_logits.argmax(dim=1).squeeze().cpu().numpy()

        img_np = np.array(img)
        mask_idx = mask_t.numpy()

        # 디코딩
        mask_rgb = decode_segmap(mask_idx)
        pred_rgb = decode_segmap(pred_idx)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_np)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(mask_rgb)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(pred_rgb)
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        save_path = output_dir / f"viz_{Path(filename).stem}.png"
        fig.savefig(str(save_path), bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {save_path}")

if __name__ == "__main__":
    main()
