import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import data_loader
import config
from models import create_model
import evaluate

from kd_engines import create_kd_engine

# 보기 싫은 로그 숨김
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
LOSS_KEY_DISPLAY_OVERRIDES = {
    "total": "Total Loss",
    "student_total": "Student Total Loss",
    "disc": "Discriminator Loss",
    "kd_pca": "PCA Loss",
    "kd_gl": "GL Loss",
    "mvg": "MVG Loss",
}

SCALAR_LOSS_KEYS: List[str] = []
LOSS_KEY_TO_HEADER: Dict[str, str] = {}
LOSS_HEADER_ORDER: List[str] = []


def _is_scalar_loss_value(value) -> bool:
    if isinstance(value, torch.Tensor):
        return value.dim() == 0
    return isinstance(value, (int, float))


def _loss_value_to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def _display_name_for_loss(key: str) -> str:
    if key in LOSS_KEY_DISPLAY_OVERRIDES:
        return LOSS_KEY_DISPLAY_OVERRIDES[key]
    pretty = key.replace('_', ' ').title()
    if "loss" not in key.lower():
        pretty = f"{pretty} Loss"
    return pretty


def _norm_losses(loss_dict):
    """
    엔진이 {'student_total','disc'} 또는 {'total'}를 반환할 수 있음.
    반환: student_loss, disc_loss, loss_dict
    """
    if "student_total" in loss_dict:
        s = loss_dict["student_total"]
    else:
        s = loss_dict["total"]  # 구형 엔진 호환

    d = loss_dict.get("disc", s.new_tensor(0.0))
    return s, d, loss_dict


# model 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
teacher = create_model(config.KD.TEACHER_NAME).to(device)
student = create_model(config.KD.STUDENT_NAME).to(device)
model = student

if config.KD.FREEZE_TEACHER:
    try:
        ckpt_path = Path(config.TEACHER_CKPT)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if "model_state" in ckpt:
                teacher.load_state_dict(ckpt["model_state"])
                print(f"▶ Successfully loaded pretrained teacher weights from: {ckpt_path}")
            else:
                teacher.load_state_dict(ckpt)
                print(f"▶ Successfully loaded pretrained teacher weights from: {ckpt_path} (direct state dict)")
        else:
            print(f"⚠️ WARNING: Teacher checkpoint not found at {ckpt_path}. Using ImageNet pretrained weights.")
    except Exception as e:
        print(f"⚠️ WARNING: Failed to load teacher checkpoint. Error: {e}. Using ImageNet pretrained weights.")

# ── KD 엔진 구성 ───────────────────────────────────
kd_engine = create_kd_engine(config.KD, teacher, student).to(device)

# --- build KD projections (dry-run) ---
imgs0, masks0 = next(iter(data_loader.train_loader))
with torch.no_grad():
    dry_run_out = kd_engine.compute_losses(imgs0.to(device, non_blocking=True),
                                           masks0.to(device, non_blocking=True),
                                           device)
# key 검증만
_norm_losses(dry_run_out)

SCALAR_LOSS_KEYS = [
    key for key, value in dry_run_out.items() if _is_scalar_loss_value(value)
]
if "student_total" in SCALAR_LOSS_KEYS:
    SCALAR_LOSS_KEYS.remove("student_total")
    SCALAR_LOSS_KEYS.insert(0, "student_total")
elif "total" in SCALAR_LOSS_KEYS:
    SCALAR_LOSS_KEYS.remove("total")
    SCALAR_LOSS_KEYS.insert(0, "total")

LOSS_KEY_TO_HEADER = {key: _display_name_for_loss(key) for key in SCALAR_LOSS_KEYS}
LOSS_HEADER_ORDER = [LOSS_KEY_TO_HEADER[key] for key in SCALAR_LOSS_KEYS]

print("▶ Tracking losses:", ", ".join(LOSS_KEY_TO_HEADER.values()))

# ── 옵티마이저/스케줄러 ─────────────────────────────
if hasattr(kd_engine, "get_student_parameters"):
    student_params = kd_engine.get_student_parameters()
else:
    student_params = list(student.parameters())
    if hasattr(kd_engine, "get_extra_parameters"):
        student_params += list(kd_engine.get_extra_parameters())
opt_s = optim.AdamW(student_params, **config.TRAIN.OPTIMIZER["PARAMS"])

if hasattr(kd_engine, "get_disc_parameters"):
    disc_params = kd_engine.get_disc_parameters()
else:
    disc_params = []
opt_d = None
if len(disc_params) > 0:
    lr_d = getattr(config.TRAIN, "LR_D", config.TRAIN.OPTIMIZER["PARAMS"]["lr"])
    opt_d = optim.AdamW(disc_params, lr=lr_d, weight_decay=config.TRAIN.OPTIMIZER["PARAMS"].get("weight_decay", 0.0))

scheduler_class = getattr(optim.lr_scheduler, config.TRAIN.SCHEDULER_CALR["NAME"])
scheduler = scheduler_class(opt_s, **config.TRAIN.SCHEDULER_CALR["PARAMS"])

# warm-up
if config.TRAIN.USE_WARMUP:
    warmup_class = getattr(optim.lr_scheduler, config.TRAIN.WARMUP_SCHEDULER["NAME"])
    warmup_params = config.TRAIN.WARMUP_SCHEDULER["PARAMS"].copy()
    warmup_params["total_iters"] = config.TRAIN.WARMUP_EPOCHS
    warmup = warmup_class(opt_s, **warmup_params)
else:
    warmup = None


def train_one_epoch_kd(kd_engine, loader, opt_s, opt_d, device):
    kd_engine.train()
    if not SCALAR_LOSS_KEYS:
        raise RuntimeError("No scalar losses registered from KD engine dry-run.")

    epoch_losses = {key: 0.0 for key in SCALAR_LOSS_KEYS}
    pbar = tqdm(loader, ascii=True, dynamic_ncols=True, leave=True, desc="Training")

    for imgs, masks in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # 1) Discriminator step
        if opt_d is not None:
            opt_d.zero_grad(set_to_none=True)
            out_d = kd_engine.compute_losses(imgs, masks, device)
            _, loss_mad, _ = _norm_losses(out_d)
            loss_mad.backward()
            opt_d.step()
        else:
            loss_mad = torch.tensor(0.0, device=imgs.device)

        # 2) Student step
        opt_s.zero_grad(set_to_none=True)
        out_s = kd_engine.compute_losses(imgs, masks, device)
        loss_student, _, out_s = _norm_losses(out_s)
        loss_student.backward()
        opt_s.step()

        # 손실 로깅
        for key in epoch_losses:
            value = out_s.get(key)
            if value is None or not _is_scalar_loss_value(value):
                continue
            epoch_losses[key] += _loss_value_to_float(value)

        postfix = {}
        for key in SCALAR_LOSS_KEYS:
            value = out_s.get(key)
            if value is None or not _is_scalar_loss_value(value):
                continue
            postfix_label = LOSS_KEY_TO_HEADER.get(key, key)
            postfix[postfix_label] = f'{_loss_value_to_float(value):.3f}'
        postfix["MAD"] = f"{loss_mad.item():.3f}"
        if postfix:
            pbar.set_postfix(postfix)

    num_batches = len(loader)
    avg_losses = {key: val / num_batches for key, val in epoch_losses.items()}
    return avg_losses


def validate_student(student_model, loader, criterion):
    student_model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, ascii=True, dynamic_ncols=True, leave=True, desc="Validation"):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = student_model(imgs)
            total_loss += criterion(preds, masks).item()
    return total_loss / len(loader)


def plot_progress(epochs, train_losses, val_losses):
    plt.figure(figsize=(8,6))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True)
    plt.savefig(str(config.GENERAL.SAVE_PLOT), bbox_inches="tight")
    plt.close()


def write_summary(init=False, best_epoch=None, best_miou=None):
    with open(config.GENERAL.SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("=== Training Configuration ===\n")
        f.write(f"Dataset path : {config.DATA.DATA_DIR}\n")
        og = opt_s.param_groups[0]
        f.write(f"Student Model: {student.__class__.__name__}  (source: {config.KD.STUDENT_NAME})\n")
        f.write(f"Teacher Model: {teacher.__class__.__name__}  (source: {config.KD.TEACHER_NAME})\n\n")
        f.write(f"Teacher Freeze: {config.KD.FREEZE_TEACHER}\n")
        f.write(f"Optimizer(S)  : {opt_s.__class__.__name__}\n")
        f.write(f"  lr           : {og['lr']}\n")
        f.write(f"  weight_decay : {og.get('weight_decay')}\n")
        if opt_d is not None:
            f.write(f"Optimizer(D)  : {opt_d.__class__.__name__}\n")
        f.write(f"Scheduler     : {scheduler.__class__.__name__}\n")
        f.write(f"Batch size    : {config.DATA.BATCH_SIZE}\n\n")
        f.write("=== Knowledge Distillation Configuration ===\n")
        f.write(f"Engine NAME        : {config.KD.ENGINE_NAME}\n")
        f.write(f"Teacher Source CKPT: {config.TEACHER_CKPT}\n\n")
        engine_name = config.KD.ENGINE_NAME
        current_engine_params = config.KD.ALL_ENGINE_PARAMS.get(engine_name, {})
        f.write(f"--- Parameters for '{engine_name}' engine ---\n")
        if not current_engine_params:
            f.write("No parameters found for this engine.\n")
        else:
            for key, value in current_engine_params.items():
                f.write(f"{key:<25} : {value}\n")
        f.write("\n")

        if init:
            f.write("=== Best Model (to be updated) ===\n")
            f.write("epoch     : N/A\nbest_val_mIoU : N/A\n\n")
        else:
            f.write("=== Best Model ===\n")
            f.write(f"epoch     : {best_epoch}\n")
            f.write(f"best_val_mIoU : {best_miou:.4f}\n\n")


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

    best_miou = 0.0
    best_epoch = 0
    best_ckpt = config.GENERAL.BASE_DIR / "best_model.pth"

    log_csv_path = config.GENERAL.LOG_DIR / "training_log.csv"
    loss_headers = LOSS_HEADER_ORDER if LOSS_HEADER_ORDER else ["Student Total Loss"]
    csv_headers = ["Epoch", *loss_headers, "Val Loss", "Val mIoU", "Pixel Acc", "LR"]
    for class_name in config.DATA.CLASS_NAMES:
        csv_headers.append(f"IoU_{class_name}")

    if log_csv_path.exists():
        try:
            existing_cols = list(pd.read_csv(log_csv_path, nrows=0).columns)
            if len(existing_cols) > 0:
                csv_headers = existing_cols
        except Exception:
            pass
    else:
        pd.DataFrame(columns=csv_headers).to_csv(log_csv_path, index=False)

    train_losses, val_losses = [], []
    loss_class = getattr(nn, config.TRAIN.LOSS_FN["NAME"])
    criterion = loss_class(**config.TRAIN.LOSS_FN["PARAMS"])

    for epoch in range(1, num_epochs + 1):
        tr_losses_dict = train_one_epoch_kd(kd_engine, data_loader.train_loader, opt_s, opt_d, device)
        if "student_total" in tr_losses_dict:
            tr_loss = tr_losses_dict["student_total"]
        else:
            tr_loss = tr_losses_dict["total"]

        vl_loss = validate_student(student, data_loader.val_loader, criterion)
        metrics = evaluate.evaluate_all(model, data_loader.val_loader, device)
        miou = metrics["mIoU"]
        pa = metrics["PixelAcc"]

        if epoch <= config.TRAIN.WARMUP_EPOCHS:
            warmup.step()
        else:
            scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        print(f"[{epoch}/{num_epochs}] "
              f"train_loss={tr_loss:.4f}, val_loss={vl_loss:.4f}, "
              f"val_mIoU={miou:.4f},  PA={pa:.4f}")

        current_lr = opt_s.param_groups[0]['lr']
        log_data = {"Epoch": epoch}
        for key in SCALAR_LOSS_KEYS:
            header_name = LOSS_KEY_TO_HEADER.get(key, key)
            log_data[header_name] = tr_losses_dict.get(key, float("nan"))

        log_data.update({
            "Val Loss": vl_loss,
            "Val mIoU": miou,
            "Pixel Acc": pa,
            "LR": current_lr,
        })
        per_cls_iou = metrics["per_class_iou"]
        for i, class_name in enumerate(config.DATA.CLASS_NAMES):
            log_data[f"IoU_{class_name}"] = per_cls_iou[i]

        df_new_row = pd.DataFrame([log_data]).reindex(columns=csv_headers)
        df_new_row.to_csv(log_csv_path, mode='a', header=False, index=False)

        if miou > best_miou:
            best_miou = miou
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state": student.state_dict(),
                "teacher_state": teacher.state_dict(),
                "opt_s_state": opt_s.state_dict(),
                "opt_d_state": opt_d.state_dict() if opt_d else None,
                "scheduler_state": scheduler.state_dict(),
                "best_val_mIoU": best_miou
            }, best_ckpt)
            print(f"▶ New best val_mIoU at epoch {epoch}: {miou:.4f} → {best_ckpt}")
            write_summary(init=False, best_epoch=best_epoch, best_miou=best_miou)

        if epoch % 10 == 0:
            plot_progress(list(range(1, epoch + 1)), train_losses, val_losses)

    end_dt = datetime.now()
    write_timing(start_dt, end_dt, config.GENERAL.SUMMARY_TXT)

    elapsed = end_dt - start_dt
    total_sec = int(elapsed.total_seconds())
    hh = total_sec // 3600
    mm = (total_sec % 3600) // 60
    ss = total_sec % 60

    print(f"\nTraining complete.")
    print(f"Started at : {start_dt:%Y-%m-%d %H:%M:%S}")
    print(f"Finished at: {end_dt:%Y-%m-%d %H:%M:%S}")
    print(f"Total time : {hh:02d}:{mm:02d}:{ss:02d} (H:M:S)")
    print(f"Best epoch: {best_epoch}, Best val_mIoU: {best_miou:.4f}")

    return best_ckpt


if __name__ == "__main__":
    run_training(config.TRAIN.EPOCHS)
