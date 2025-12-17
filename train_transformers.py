import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

import data_loader
import config
from models import create_model
import evaluate

from torch.utils.tensorboard import SummaryWriter  # ← TensorBoard

# 보기 싫은 로그 숨김
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────
# model 설정
# ──────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model(config.MODEL.NAME)
model.to(device)

# loss function
loss_class = getattr(nn, config.TRAIN.LOSS_FN["NAME"])
criterion = loss_class(**config.TRAIN.LOSS_FN["PARAMS"])

# optimizer
optimizer_class = getattr(optim, config.TRAIN.OPTIMIZER["NAME"])
optimizer = optimizer_class(model.parameters(), **config.TRAIN.OPTIMIZER["PARAMS"])

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

# ──────────────────────────────────
# TensorBoard & SWT Attention Hook
# ──────────────────────────────────

# TensorBoard log dir
tb_log_dir = config.GENERAL.LOG_DIR / "tb"
tb_log_dir.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir=str(tb_log_dir))

# SWT attention map 캐시
swt_attn_cache = {"map": None}


def _register_swt_hook_if_available():
    """
    model.swt_attn.attention_net 에 forward hook을 달아서
    spatial attention map (Sigmoid 이후)을 swt_attn_cache["map"]에 저장.
    """
    if not hasattr(model, "swt_attn"):
        print("▶ SWT attention module not found on model (no logging for SWT).")
        return

    swt_attn = getattr(model, "swt_attn")
    if swt_attn is None:
        print("▶ model.swt_attn is None (no logging for SWT).")
        return

    if not hasattr(swt_attn, "attention_net"):
        print("▶ model.swt_attn has no 'attention_net' (no logging for SWT).")
        return

    def _swt_hook(module, inputs, output):
        # output: [B, 1, H, W] (Sigmoid 후 spatial attention map)
        try:
            swt_attn_cache["map"] = output.detach().cpu()
        except Exception:
            swt_attn_cache["map"] = None

    swt_attn.attention_net.register_forward_hook(_swt_hook)
    print("▶ SWT attention forward-hook registered for TensorBoard logging.")


_register_swt_hook_if_available()


# ──────────────────────────────────
# Helper: ImageNet de-normalization
# ──────────────────────────────────
def denorm_imagenet(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B,3,H,W], ImageNet mean/std로 정규화된 텐서
    return: [B,3,H,W], [0,1] 범위로 클램핑된 텐서
    """
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = x * std + mean
    return x.clamp(0.0, 1.0)


# ──────────────────────────────────
# 1epoch당 학습
# ──────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, epoch,
                    writer=None, swt_cache=None, global_step=0):
    model.train()
    total_loss = 0.0

    for batch_idx, (imgs, masks) in enumerate(
        tqdm(loader, ascii=True, dynamic_ncols=True, leave=True, desc=f"Training {epoch}")
    ):
        # NOTE:
        #   JointKDTrainAugmentation 이 활성화된 경우, DataLoader는
        #   ((student_batch, teacher_batch), mask_batch) 형태를 반환한다.
        #   기존 코드는 imgs가 Tensor라고 가정하고 .to()를 바로 호출했기 때문에
        #   리스트/튜플이면 AttributeError가 발생했다.
        #   → 학생 입력만 사용하되, student 텐서를 안전하게 꺼내서 GPU로 옮긴다.
        if isinstance(imgs, (list, tuple)):
            imgs = imgs[0]
        imgs, masks = imgs.to(device), masks.to(device)

        # Forward
        preds = model(imgs)
        loss = criterion(preds, masks)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # ─── TensorBoard logging (per-iteration) ───
        if writer is not None:
            # iteration-wise loss
            writer.add_scalar("Loss/train_iter", loss.item(), global_step)

            # SWT attention stats
            if swt_cache is not None and swt_cache.get("map") is not None:
                attn = swt_cache["map"]  # [B,1,H,W] on CPU
                attn_mean = attn.mean().item()
                attn_std = attn.std().item()
                writer.add_scalar("SWT/attn_mean_iter", attn_mean, global_step)
                writer.add_scalar("SWT/attn_std_iter", attn_std, global_step)

                # 첫 배치에 대해 epoch 단위로 attention map / 원본 이미지 시각화
                if batch_idx == 0:
                    # attention map [1,H,W] 형태로 정규화
                    attn_vis = attn[0, 0:1]  # [1,H,W]
                    # input과 크기 다르면 upsample
                    if attn_vis.shape[-2:] != imgs.shape[-2:]:
                        attn_vis = F.interpolate(
                            attn_vis.unsqueeze(0),
                            size=imgs.shape[-2:],
                            mode="bilinear",
                            align_corners=False
                        )[0]
                    attn_vis = attn_vis - attn_vis.min()
                    attn_vis = attn_vis / (attn_vis.max() + 1e-6)

                    writer.add_image(f"SWT/attn_epoch", attn_vis, epoch)

                    # 대응되는 RGB 이미지 (de-normalize)
                    rgb_vis = denorm_imagenet(imgs[0:1].detach().cpu())  # [1,3,H,W]
                    writer.add_image(f"SWT/rgb_epoch", rgb_vis[0], epoch)

        global_step += 1

    return total_loss / len(loader), global_step


# ──────────────────────────────────
# 검증
# ──────────────────────────────────
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, ascii=True, dynamic_ncols=True, leave=True, desc="Validation"):
            if isinstance(imgs, (list, tuple)):
                imgs = imgs[0]
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            total_loss += criterion(preds, masks).item()
    return total_loss / len(loader)


# ──────────────────────────────────
# Loss curve 저장
# ──────────────────────────────────
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


# NEW: PDM 설정 덤프 헬퍼
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


def write_summary(init=False, best_epoch=None, best_miou=None):
    """
    init=True: config만 기록
    else: best 모델 info 덮어쓰기
    """
    with open(config.GENERAL.SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("=== Training Configuration ===\n")
        f.write(f"Dataset path : {config.DATA.DATA_DIR}\n")
        og = optimizer.param_groups[0]
        f.write(f"Model         : {model.__class__.__name__}\n")
        f.write(f"Model source  : {config.MODEL.NAME}\n\n")
        f.write(f"Optimizer     : {optimizer.__class__.__name__}\n")
        f.write(f"  lr           : {og['lr']}\n")
        f.write(f"  weight_decay : {og.get('weight_decay')}\n")
        f.write(f"Scheduler     : {scheduler.__class__.__name__}\n")
        f.write(f"Batch size    : {config.DATA.BATCH_SIZE}\n\n")

        # NEW: PDM 설정 요약 기록
        _dump_pdm_config(f)

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
    with open(path, "a", encoding="utf-8") as f:  # append
        f.write("=== Timing ===\n")
        f.write(f"Start : {start_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End   : {end_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total : {hh:02d}:{mm:02d}:{ss:02d} (H:M:S)\n\n")


# ──────────────────────────────────
# 학습 루프
# ──────────────────────────────────
def run_training(num_epochs):
    # 초기 summary 파일 생성
    write_summary(init=True)
    start_dt = datetime.now()
    print(f"Started at : {start_dt:%Y-%m-%d %H:%M:%S}")

    best_miou = 0.0
    best_epoch = 0
    best_ckpt = config.GENERAL.BASE_DIR / "best_model.pth"

    # CSV 로그 파일 경로 설정 및 헤더 생성
    log_csv_path = config.GENERAL.LOG_DIR / "training_log.csv"
    csv_headers = [
        "Epoch", "Train Loss", "VAL Loss",
        "Val mIoU", "Pixel Acc", "LR"
    ]
    # 클래스별 IoU 헤더 추가
    for class_name in config.DATA.CLASS_NAMES:
        csv_headers.append(f"IoU_{class_name}")

    if not log_csv_path.exists():
        df_log = pd.DataFrame(columns=csv_headers)
        df_log.to_csv(log_csv_path, index=False)

    train_losses, val_losses = [], []
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        # training
        tr_loss, global_step = train_one_epoch(
            model, data_loader.train_loader, criterion, optimizer,
            epoch, writer=writer, swt_cache=swt_attn_cache, global_step=global_step
        )
        # validation
        vl_loss = validate(model, data_loader.val_loader, criterion)

        # scheduler
        if warmup is not None and epoch <= config.TRAIN.WARMUP_EPOCHS:
            warmup.step()
        else:
            scheduler.step()

        # metric (mIoU, pixel acc, per-class IoU)
        metrics = evaluate.evaluate_all(model, data_loader.val_loader, device)
        miou = metrics["mIoU"]
        pa = metrics["PixelAcc"]

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        print(f"[{epoch}/{num_epochs}] "
              f"train_loss={tr_loss:.4f}, val_loss={vl_loss:.4f}, "
              f"val_mIoU={miou:.4f},  PA={pa:.4f}")

        current_lr = optimizer.param_groups[0]['lr']

        # ─── TensorBoard: epoch-wise logging ───
        if writer is not None:
            writer.add_scalar("Loss/train_epoch", tr_loss, epoch)
            writer.add_scalar("Loss/val_epoch", vl_loss, epoch)
            writer.add_scalar("Metric/mIoU", miou, epoch)
            writer.add_scalar("Metric/PixelAcc", pa, epoch)
            writer.add_scalar("LR/learning_rate", current_lr, epoch)

            # per-class IoU도 option으로 기록
            per_cls_iou = metrics["per_class_iou"]
            for i, class_name in enumerate(config.DATA.CLASS_NAMES):
                writer.add_scalar(f"IoU/{class_name}", per_cls_iou[i], epoch)

        # CSV 파일에 성능 지표 기록
        log_data = {
            "Epoch": epoch,
            "TRAIN Loss": tr_loss,
            "VAL Loss": vl_loss,
            "Val mIoU": miou,
            "Pixel Acc": pa,
            "LR": current_lr
        }
        per_cls_iou = metrics["per_class_iou"]
        for i, class_name in enumerate(config.DATA.CLASS_NAMES):
            log_data[f"IoU_{class_name}"] = per_cls_iou[i]

        df_new_row = pd.DataFrame([log_data])
        df_new_row.to_csv(log_csv_path, mode='a', header=False, index=False)

        # best model 갱신
        if miou > best_miou:
            best_miou = miou
            best_epoch = epoch

            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_mIoU": best_miou
            }, best_ckpt)
            print(f"▶ New best val_mIoU at epoch {epoch}: {miou:.4f} → {best_ckpt}")
            write_summary(init=False, best_epoch=best_epoch, best_miou=best_miou)

        # 10 epoch마다 loss curve plot 저장
        if epoch % 10 == 0:
            plot_progress(list(range(1, epoch + 1)), train_losses, val_losses)

    end_dt = datetime.now()
    elapsed = end_dt - start_dt
    total_sec = int(elapsed.total_seconds())
    hh = total_sec // 3600
    mm = (total_sec % 3600) // 60
    ss = total_sec % 60

    print(f"Started at : {start_dt:%Y-%m-%d %H:%M:%S}")
    print(f"Finished at: {end_dt:%Y-%m-%d %H:%M:%S}")
    print(f"Total time : {hh:02d}:{mm:02d}:{ss:02d} (H:M:S)")

    write_timing(start_dt, end_dt, config.GENERAL.SUMMARY_TXT)

    print(f"Training complete. Best epoch: {best_epoch}, Best val_mIoU: {best_miou:.4f}")
    return best_ckpt


if __name__ == "__main__":
    ckpt_path = run_training(config.TRAIN.EPOCHS)
    # TensorBoard writer 정리
    writer.close()