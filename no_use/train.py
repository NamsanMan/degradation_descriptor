import os

import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch
import data_loader
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from timm.scheduler.poly_lr import PolyLRScheduler
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import copy
from fvcore.nn import FlopCountAnalysis
from datetime import datetime
import segmentation_models_pytorch as smp

# 보기 싫은 로그 숨김
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 훈련 돌릴 epoch 설정
train_epoch = 100
warmup_epoch = 5

class_names = [
    "Sky", "Building", "Pole", "Road", "Sidewalk",
    "Tree", "SignSymbol", "Fence", "Car",
    "Pedestrian", "Bicyclist", "Void"
]

# model 설정
ENCODER_NAME = "resnet50"
ENCODER_WEIGHTS = "imagenet"
model = smp.DeepLabV3Plus(
    encoder_name=ENCODER_NAME,
    encoder_weights=ENCODER_WEIGHTS,
    in_channels=3,
    classes=12
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=11)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)
warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.1,
    end_factor=1.0,
    total_iters=warmup_epoch
)

# 저장 위치 설정
base_dir = Path(r"E:\LAB\result_files\test_results\temp")
log_dir = base_dir / "log"
log_dir.mkdir(parents=True, exist_ok=True)

summary_txt  = log_dir / "training_summary.txt"
save_plot    = log_dir / "training_progress.png"
os.makedirs(os.path.dirname(str(save_plot)), exist_ok=True)

# 모델의 parameter, FLOPs수 계산 >> batch size와 관련없는 per-sample FLOPs를 계산한다
model_cpu = copy.deepcopy(model).cpu().eval()
H,W = data_loader.input_resolution
dummy = torch.randn(1, 3, H, W, dtype=next(model_cpu.parameters()).dtype)
with torch.no_grad():
    flops = FlopCountAnalysis(model_cpu, dummy)
    total_flops = flops.total()
    params = sum(p.numel() for p in model_cpu.parameters())

backbone_class  = model_cpu.encoder.__class__.__name__
backbone_params = sum(p.numel() for p in model_cpu.encoder.parameters())
head_params     = params - backbone_params    # decoder + segmentation_head 포함


# 1epoch당 학습 방법 설정 후 loss값 반환
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(loader, ascii=True, dynamic_ncols=True, leave=True, desc="Training"):
        imgs, masks = imgs.to(device), masks.to(device)
        #logit tensor를 리턴
        preds = model(imgs)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# 검증방법 설정 후 loss값 반환
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, ascii=True, dynamic_ncols=True, leave=True, desc="Validation"):
            imgs, masks = imgs.to(device), masks.to(device)
            # FCN 스타일의 리턴
            #preds = model(imgs)['out']
            # logit tensor를 리턴
            preds = model(imgs)
            total_loss += criterion(preds, masks).item()
    return total_loss / len(loader)

# 픽셀 정확도 & mIoU 평가 및 지표 계산
def compute_dataset_iou(loader):
    """
    dataset-wise mIoU: 전체 intersection/union 누적 후 클래스별 IoU 평균
    """
    intersection = np.zeros(12, dtype=np.int64)
    union        = np.zeros(12, dtype=np.int64)

    model.eval()
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds       = torch.argmax(model(imgs), dim=1)
            for cls in range(12):
                if cls == 11:  # ignore Void
                    continue
                p_inds = (preds == cls)
                t_inds = (masks == cls)
                intersection[cls] += (p_inds & t_inds).sum().item()
                union       [cls] += (p_inds | t_inds).sum().item()

    ious = [intersection[c]/union[c]
            for c in range(12)
            if c!=11 and union[c]>0]
    return float(np.mean(ious))

def compute_dataset_pa(loader):
    """
    dataset-wise Pixel Accuracy: 전체 올바른 픽셀/전체 유효 픽셀
    """
    correct = 0
    total   = 0
    model.eval()
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds       = torch.argmax(model(imgs), dim=1)
            valid       = (masks != 11)
            correct    += ((preds==masks)&valid).sum().item()
            total      += valid.sum().item()
    return correct/total

def compute_per_class_iou(loader, num_classes=12, ignore_index=11):
    """
    클래스별 IoU를 계산해 numpy array로 반환
    """
    intersection = np.zeros(num_classes, dtype=np.int64)
    union        = np.zeros(num_classes, dtype=np.int64)

    model.eval()
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = torch.argmax(model(imgs), dim=1)
            for cls in range(num_classes):
                if cls == ignore_index:
                    continue
                p_inds = (preds == cls)
                t_inds = (masks == cls)
                intersection[cls] += (p_inds & t_inds).sum().item()
                union       [cls] += (p_inds | t_inds).sum().item()

    ious = []
    for cls in range(num_classes):
        if cls == ignore_index or union[cls] == 0:
            ious.append(np.nan)
        else:
            ious.append(intersection[cls] / union[cls])
    return np.array(ious)

def compute_confusion_matrix(loader, num_classes=12, ignore_index=11):
    """
    전체 데이터에 대해 (true, pred) confusion matrix 반환
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = torch.argmax(model(imgs), dim=1)
            preds_np = preds.cpu().numpy().flatten()
            masks_np = masks.cpu().numpy().flatten()
            valid = masks_np != ignore_index
            preds_np = preds_np[valid]
            masks_np = masks_np[valid]
            cm_batch = confusion_matrix(masks_np, preds_np, labels=list(range(num_classes)))
            cm += cm_batch
    return cm

def plot_progress(epochs, train_losses, val_losses):
    plt.figure(figsize=(8,6))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses,   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True)
    plt.savefig(str(save_plot), bbox_inches="tight")
    plt.close()



def write_summary(init=False, best_epoch=None, best_miou=None):
    """
    init=True: config만 기록
    else: best 모델 info 덮어쓰기
    """
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("=== Training Configuration ===\n")
        f.write(f"Dataset path  : {data_loader.data_dir}\n")

        # Model / Backbone
        f.write(f"Model         : {model.__class__.__name__}\n")
        f.write(f"Backbone      : {ENCODER_NAME} [{backbone_class}]\n")
        f.write(f"Pretrained    : {ENCODER_WEIGHTS}\n\n")

        # Optimizer / Scheduler
        og = optimizer.param_groups[0]
        f.write(f"Optimizer     : {optimizer.__class__.__name__}\n")
        f.write(f"  lr           : {og['lr']}\n")
        f.write(f"  weight_decay : {og.get('weight_decay')}\n")
        f.write(f"Scheduler     : {scheduler.__class__.__name__}\n")
        f.write(f"Batch size    : {data_loader.train_loader.batch_size}\n\n")

        # Params / FLOPs
        f.write(f"#Parameters (total)      : {params/1e6:.2f} M\n")
        f.write(f"  ├─ backbone            : {backbone_params/1e6:.2f} M\n")
        f.write(f"  └─ head+decoder        : {head_params/1e6:.2f} M\n")
        f.write(f"FLOPs(per sample, {H}x{W}) : {total_flops/1e9:.2f} G\n\n")

        if init:
            f.write("=== Best Model (to be updated) ===\n")
            f.write("epoch        : N/A\nbest_val_mIoU: N/A\n\n")
        else:
            f.write("=== Best Model ===\n")
            f.write(f"epoch        : {best_epoch}\n")
            f.write(f"best_val_mIoU: {best_miou:.4f}\n\n")

def write_timing(start_dt, end_dt, path=summary_txt):
    elapsed = end_dt - start_dt
    total_sec = int(elapsed.total_seconds())
    hh = total_sec // 3600
    mm = (total_sec % 3600) // 60
    ss = total_sec % 60
    with open(path, "a", encoding="utf-8") as f:  # append
        f.write("=== Timing ===\n")
        f.write(f"Start : {start_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End   : {end_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total : {hh:02d}:{mm:02d}:{ss:02d} (H:M:S)\n\n")

# 학습 진행 및 잘 되고있나 성능평가
def run_training(num_epochs):
    # 초기 summary 파일 생성
    write_summary(init=True)
    start_dt = datetime.now()
    print(f"Started at : {start_dt:%Y-%m-%d %H:%M:%S}")

    best_miou   = 0.0
    best_epoch  = 0
    best_ckpt   = base_dir / "best_model.pth"

    train_losses, val_losses = [], []

    for epoch in range(1, num_epochs+1):
        tr_loss = train_one_epoch(model, data_loader.train_loader, criterion, optimizer)
        vl_loss = validate(model, data_loader.val_loader, criterion)

        # scheduler.step for ReduceLROnPlateau needs val loss
        if epoch <= warmup_epoch:
            warmup.step()
        else:
            scheduler.step(vl_loss)

        miou = compute_dataset_iou(data_loader.val_loader)
        pa   = compute_dataset_pa(data_loader.val_loader)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        print(f"[{epoch}/{num_epochs}] "
              f"train_loss={tr_loss:.4f}, val_loss={vl_loss:.4f}, "
              f"val_mIoU={miou:.4f},  PA={pa:.4f}")

        # best model 갱신
        if miou > best_miou:
            best_miou  = miou
            best_epoch = epoch

            # 1) per-class IoU 계산 및 CSV 저장
            per_cls_iou = compute_per_class_iou(data_loader.val_loader)
            df_iou = pd.DataFrame({
                "Class": class_names,
                "IoU": per_cls_iou
            })
            iou_path = log_dir / f"iou_epoch_{epoch}.csv"
            df_iou.to_csv(iou_path, index=False)

            # 2) confusion matrix 계산 및 시각화 저장
            cm = compute_confusion_matrix(data_loader.val_loader)
            plt.figure(figsize=(8,6))
            plt.imshow(cm, interpolation='nearest')
            plt.title(f"Confusion Matrix (Epoch {epoch})")
            plt.colorbar()
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45, ha='right')
            plt.yticks(tick_marks, class_names)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            cm_path = log_dir / f"confusion_matrix_epoch_{epoch}.png"
            plt.savefig(str(cm_path), bbox_inches="tight")
            plt.close()
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_val_mIoU": best_miou
            }, best_ckpt)
            print(f"▶ New best val_mIoU at epoch {epoch}: {miou:.4f} → {best_ckpt}")
            write_summary(init=False, best_epoch=best_epoch, best_miou=best_miou)

        # 10 epoch마다 plot 저장
        if epoch % 10 == 0:
            plot_progress(list(range(1, epoch+1)), train_losses, val_losses)

    end_dt = datetime.now()
    elapsed = end_dt - start_dt
    total_sec = int(elapsed.total_seconds())
    hh = total_sec // 3600
    mm = (total_sec % 3600) // 60
    ss = total_sec % 60

    # NEW: 콘솔 출력
    print(f"Started at : {start_dt:%Y-%m-%d %H:%M:%S}")
    print(f"Finished at: {end_dt:%Y-%m-%d %H:%M:%S}")
    print(f"Total time : {hh:02d}:{mm:02d}:{ss:02d} (H:M:S)")

    # NEW: summary_txt에 append
    write_timing(start_dt, end_dt, summary_txt)

    print(f"Training complete. Best epoch: {best_epoch}, Best val_mIoU: {best_miou:.4f}")
    return best_ckpt

if __name__ == "__main__":
    run_training(train_epoch)