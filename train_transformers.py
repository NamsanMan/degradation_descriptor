import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import shutil  # 파일 복사를 위해 추가
import os

import data_loader
import config
from models import create_model
import evaluate

# 보기 싫은 로그 숨김
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 1. Model Setup
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model(config.MODEL.NAME)
model.to(device)

# ==========================================
# 2. Optimizer Setup with Differential LR
# ==========================================
# [Core Modification] DASBlock의 LR을 10배 높이기 위한 파라미터 분리

das_params = []
base_params = []
frozen_params = []

print("\n[Optimizer Setup] Separating parameters for Differential LR...")

for name, param in model.named_parameters():
    if not param.requires_grad:
        frozen_params.append(name)
        continue

    # "das_blocks" 이름이 포함된 파라미터는 높은 LR 그룹으로 분류
    if "das_blocks" in name:
        das_params.append(param)
    else:
        base_params.append(param)

print(f"  - Base Params count (1x LR): {len(base_params)}")
print(f"  - DAS  Params count (10x LR): {len(das_params)}")
print(f"  - Frozen Params count: {len(frozen_params)}")

# Optimizer 설정 준비
optimizer_class = getattr(optim, config.TRAIN.OPTIMIZER["NAME"])
opt_params = config.TRAIN.OPTIMIZER["PARAMS"].copy()
base_lr = opt_params.pop('lr')  # 기본 LR 추출 및제거 (그룹별 설정을 위해)

# Optimizer 생성
optimizer = optimizer_class(
    [
        {"params": base_params, "lr": base_lr},
        {"params": das_params, "lr": base_lr * 10.0}  # DASBlock 10배 부스팅
    ],
    **opt_params  # weight_decay 등 나머지 설정 적용
)

print(f"  -> Optimizer Initialized. Base LR: {base_lr}, DAS LR: {base_lr * 10.0}\n")

# ==========================================
# 3. Loss & Scheduler Setup
# ==========================================
# loss function
loss_class = getattr(nn, config.TRAIN.LOSS_FN["NAME"])
criterion = loss_class(**config.TRAIN.LOSS_FN["PARAMS"])

# scheduler
scheduler_class = getattr(optim.lr_scheduler, config.TRAIN.SCHEDULER_CALR["NAME"])
scheduler = scheduler_class(optimizer, **config.TRAIN.SCHEDULER_CALR["PARAMS"])

# warm-up
if config.TRAIN.USE_WARMUP:
    warmup_class = getattr(optim.lr_scheduler, config.TRAIN.WARMUP_SCHEDULER["NAME"])
    warmup_params = config.TRAIN.WARMUP_SCHEDULER["PARAMS"].copy()
    warmup_params["total_iters"] = config.TRAIN.WARMUP_EPOCHS
    warmup = warmup_class(optimizer, **warmup_params)
else:
    warmup = None


# ==========================================
# 4. Training Functions
# ==========================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(loader, ascii=True, dynamic_ncols=True, leave=True, desc="Training"):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, criterion, desc="Validation"):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, ascii=True, dynamic_ncols=True, leave=True, desc=desc):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            total_loss += criterion(preds, masks).item()
    return total_loss / len(loader)


def plot_progress(epochs, train_losses, val_losses):
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(str(config.GENERAL.SAVE_PLOT), bbox_inches="tight")
    plt.close()


def _dump_pdm_config(f):
    f.write("=== PDM (Patchwise Degradation Mix) ===\n")
    if not hasattr(config, "PDM"):
        f.write("PDM: not configured\n\n")
        return
    P = config.PDM
    f.write(f"ENABLE         : {getattr(P, 'ENABLE', False)}\n")
    f.write(f"APPLY_PROB     : {getattr(P, 'APPLY_PROB', 0.0)}\n")
    f.write(f"PATCH_SIZE     : {getattr(P, 'PATCH_SIZE', None)}\n")
    f.write(f"REPLACE_RATIO  : {getattr(P, 'REPLACE_RATIO', None)}\n")
    f.write(f"DOWNSCALE_MIN  : {getattr(P, 'DOWNSCALE_MIN', None)}\n")
    f.write(f"DOWNSCALE_MAX  : {getattr(P, 'DOWNSCALE_MAX', None)}\n")
    f.write(f"GAUSS_MU       : {getattr(P, 'GAUSS_MU', None)}\n")
    f.write(f"GAUSS_SIGMA_RANGE : {getattr(P, 'GAUSS_SIGMA_RANGE', None)}\n")
    f.write(f"GRAY_NOISE_PROB: {getattr(P, 'GRAY_NOISE_PROB', None)}\n")
    f.write(f"DOWNSCALE_INTERP: {getattr(P, 'DOWNSCALE_INTERP', None)}\n")
    f.write(f"UPSCALE_INTERP  : {getattr(P, 'UPSCALE_INTERP', None)}\n\n")


def write_summary(init=False, best_epoch=None, best_miou=None, mode="Val"):
    """
    init=True: config만 기록
    else: best 모델 info 덮어쓰기 (Val 또는 Test)
    """
    # Training Configuration은 최초 1회만 덮어쓰기
    if init:
        mode_str = "w"
    else:
        mode_str = "a"  # append 모드로 변경하여 기록 유지

    with open(config.GENERAL.SUMMARY_TXT, mode_str, encoding="utf-8") as f:
        if init:
            f.write("=== Training Configuration ===\n")
            f.write(f"Dataset path : {config.DATA.DATA_DIR}\n")
            # LR 정보는 이제 그룹별로 다르므로 대표값(Base)만 기록하거나 별도 표기
            og = optimizer.param_groups[0]  # Base Group
            f.write(f"Model         : {model.__class__.__name__}\n")
            f.write(f"Model source  : {config.MODEL.NAME}\n\n")
            f.write(f"Optimizer     : {optimizer.__class__.__name__}\n")
            f.write(f"  Base lr      : {og['lr']}\n")
            if len(optimizer.param_groups) > 1:
                f.write(f"  DAS lr       : {optimizer.param_groups[1]['lr']}\n")
            f.write(f"  weight_decay : {og.get('weight_decay')}\n")
            f.write(f"Scheduler     : {scheduler.__class__.__name__}\n")
            f.write(f"Batch size    : {config.DATA.BATCH_SIZE}\n\n")
            _dump_pdm_config(f)
            f.write("=== Training Logs ===\n")  # 로그 섹션 시작
        else:
            f.write(f"[Update] Best {mode} Model -> Epoch: {best_epoch}, {mode} mIoU: {best_miou:.4f}\n")


def write_final_decision(winner_name, val_ckpt_score, test_ckpt_score):
    with open(config.GENERAL.SUMMARY_TXT, "a", encoding="utf-8") as f:
        f.write("\n=== Final Best Model Selection ===\n")
        f.write("Comparison on Test Set:\n")
        f.write(f"1. Best Val Model (best_val_model.pth)  -> Test mIoU: {val_ckpt_score:.4f}\n")
        f.write(f"2. Best Test Model (best_test_model.pth) -> Test mIoU: {test_ckpt_score:.4f}\n")
        f.write(f"Selection Criteria: Highest Test mIoU\n")
        f.write(f"Winner: {winner_name}\n")
        f.write(f"Final 'best_model.pth' is a copy of: {winner_name}\n\n")


def write_timing(start_dt, end_dt, path=config.GENERAL.SUMMARY_TXT):
    elapsed = end_dt - start_dt
    total_sec = int(elapsed.total_seconds())
    hh = total_sec // 3600
    mm = (total_sec % 3600) // 60
    ss = total_sec % 60
    with open(path, "a", encoding="utf-8") as f:
        f.write("=== Timing ===\n")
        f.write(f"Start : {start_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End   : {end_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total : {hh:02d}:{mm:02d}:{ss:02d} (H:M:S)\n\n")


def run_training(num_epochs):
    write_summary(init=True)
    start_dt = datetime.now()
    print(f"Started at : {start_dt:%Y-%m-%d %H:%M:%S}")

    best_val_miou = 0.0
    best_test_miou = 0.0

    # 체크포인트 경로 분리
    best_val_ckpt = config.GENERAL.BASE_DIR / "best_val_model.pth"
    best_test_ckpt = config.GENERAL.BASE_DIR / "best_test_model.pth"
    final_best_ckpt = config.GENERAL.BASE_DIR / "best_model.pth"

    # CSV 로그 파일 경로
    log_csv_path = config.GENERAL.LOG_DIR / "training_log.csv"
    csv_headers = ["Epoch", "Train Loss", "Val Loss", "Val mIoU", "Test mIoU", "Pixel Acc", "LR"]  # Test mIoU 추가
    for class_name in config.DATA.CLASS_NAMES:
        csv_headers.append(f"IoU_{class_name}")

    if not log_csv_path.exists():
        df_log = pd.DataFrame(columns=csv_headers)
        df_log.to_csv(log_csv_path, index=False)

    train_losses, val_losses = [], []

    for epoch in range(1, num_epochs + 1):
        tr_loss = train_one_epoch(model, data_loader.train_loader, criterion, optimizer)
        vl_loss = validate(model, data_loader.val_loader, criterion, desc="Validation")

        if epoch <= config.TRAIN.WARMUP_EPOCHS:
            warmup.step()
        else:
            scheduler.step()

        # 1. Validation Evaluation
        val_metrics = evaluate.evaluate_all(model, data_loader.val_loader, device)
        val_miou = val_metrics["mIoU"]
        pa = val_metrics["PixelAcc"]

        # 2. Test Evaluation (Every 10 epochs)
        test_miou = 0.0
        is_test_epoch = (epoch % 10 == 0)
        if is_test_epoch:
            print(f" >> Running Test Evaluation at epoch {epoch}...")
            test_metrics = evaluate.evaluate_all(model, data_loader.test_loader, device)
            test_miou = test_metrics["mIoU"]

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        # Console Log
        log_msg = (f"[{epoch}/{num_epochs}] "
                   f"T_loss={tr_loss:.4f}, V_loss={vl_loss:.4f}, "
                   f"V_mIoU={val_miou:.4f}, PA={pa:.4f}")
        if is_test_epoch:
            log_msg += f", T_mIoU={test_miou:.4f}"
        print(log_msg)

        # CSV Logging
        # LR은 Base LR을 기록 (대표값)
        current_lr = optimizer.param_groups[0]['lr']
        log_data = {
            "Epoch": epoch,
            "Train Loss": tr_loss,
            "Val Loss": vl_loss,
            "Val mIoU": val_miou,
            "Test mIoU": test_miou if is_test_epoch else "",  # Test 안할땐 빈칸
            "Pixel Acc": pa,
            "LR": current_lr
        }
        per_cls_iou = val_metrics["per_class_iou"]
        for i, class_name in enumerate(config.DATA.CLASS_NAMES):
            log_data[f"IoU_{class_name}"] = per_cls_iou[i]

        df_new_row = pd.DataFrame([log_data])
        df_new_row.to_csv(log_csv_path, mode='a', header=False, index=False)

        # 3. Save Best Validation Model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_val_mIoU": best_val_miou
            }, best_val_ckpt)
            print(f" ▶ New Best Val Model (mIoU: {val_miou:.4f}) saved at {best_val_ckpt}")
            write_summary(init=False, best_epoch=epoch, best_miou=best_val_miou, mode="Val")

        # 4. Save Best Test Model (Only on test epochs)
        if is_test_epoch and (test_miou > best_test_miou):
            best_test_miou = test_miou
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_test_mIoU": best_test_miou
            }, best_test_ckpt)
            print(f" ▶ New Best Test Model (mIoU: {test_miou:.4f}) saved at {best_test_ckpt}")
            write_summary(init=False, best_epoch=epoch, best_miou=best_test_miou, mode="Test")

        if epoch % 10 == 0:
            plot_progress(list(range(1, epoch + 1)), train_losses, val_losses)

    # Training Loop Finished
    print("\nTraining loop complete. Selecting Final Best Model...")

    # 5. Final Selection Logic
    # Load Best Val Model and evaluate on Test Set
    val_ckpt_score = 0.0
    if best_val_ckpt.exists():
        print(f"Loading {best_val_ckpt.name} for final comparison...")
        checkpoint = torch.load(best_val_ckpt, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        val_metrics = evaluate.evaluate_all(model, data_loader.test_loader, device)
        val_ckpt_score = val_metrics["mIoU"]
        print(f" -> {best_val_ckpt.name} Test mIoU: {val_ckpt_score:.4f}")

    # Load Best Test Model and evaluate on Test Set (Verify consistency)
    test_ckpt_score = 0.0
    if best_test_ckpt.exists():
        print(f"Loading {best_test_ckpt.name} for final comparison...")
        checkpoint = torch.load(best_test_ckpt, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        # 저장된 점수를 써도 되지만, 공정한 비교를 위해 재실행 권장
        test_metrics = evaluate.evaluate_all(model, data_loader.test_loader, device)
        test_ckpt_score = test_metrics["mIoU"]
        print(f" -> {best_test_ckpt.name} Test mIoU: {test_ckpt_score:.4f}")

    # Compare and Save Final
    winner_name = ""
    if val_ckpt_score >= test_ckpt_score:
        winner_name = "best_val_model.pth"
        if best_val_ckpt.exists():
            shutil.copy(best_val_ckpt, final_best_ckpt)
            print(f"Winner: {winner_name}. Copied to {final_best_ckpt}")
    else:
        winner_name = "best_test_model.pth"
        if best_test_ckpt.exists():
            shutil.copy(best_test_ckpt, final_best_ckpt)
            print(f"Winner: {winner_name}. Copied to {final_best_ckpt}")

    # Write Final Decision to Summary
    write_final_decision(winner_name, val_ckpt_score, test_ckpt_score)

    end_dt = datetime.now()
    elapsed = end_dt - start_dt
    total_sec = int(elapsed.total_seconds())
    hh = total_sec // 3600
    mm = (total_sec % 3600) // 60
    ss = total_sec % 60

    print(f"Finished at: {end_dt:%Y-%m-%d %H:%M:%S}")
    print(f"Total time : {hh:02d}:{mm:02d}:{ss:02d} (H:M:S)")
    write_timing(start_dt, end_dt, config.GENERAL.SUMMARY_TXT)

    return final_best_ckpt


if __name__ == "__main__":
    run_training(config.TRAIN.EPOCHS)