import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from torch_poly_lr_decay import PolynomialLRDecay  # ← PolyLR (cmpark0126)
from torch.amp import autocast, GradScaler    # ← AMP

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
    "ce_student": "CE Student Loss",
    "ce_teacher": "CE Teacher Loss",
    "kd_logit": "KD Logit Loss",
    "kd_feat": "KD Feature Loss",
    "pca_total": "PCA Total Loss",
    "pca_attn": "PCA Attention Loss",
    "pca_v": "PCA Value Loss",
    "gl_loss": "GL Loss",
    "hf_kd": "HF Feature KD Loss",
    "warmup_alpha_hf": "HF Warmup Alpha",
    "hf_weight_eff": "HF KD Weight (Eff)",
    "mask_mean": "HF Mask Mean",
    "mask_std": "HF Mask Std",
    "mask_min": "HF Mask Min",
    "mask_max": "HF Mask Max",
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
            # Teacher 체크포인트가 'model_state' 키를 가지고 있는지 확인하고 로드
            if "model_state" in ckpt:
                teacher.load_state_dict(ckpt["model_state"])        # "model_state"만 로드 하면서 .pth의 내용중 가중치만 불러옴
                print(f"▶ Successfully loaded pretrained teacher weights from: {ckpt_path}")
            else:
                # 키가 다른 경우를 대비하여 직접 로드 시도
                teacher.load_state_dict(ckpt)
                print(f"▶ Successfully loaded pretrained teacher weights from: {ckpt_path} (direct state dict)")

        else:
            print(f"⚠️ WARNING: Teacher checkpoint not found at {ckpt_path}. Using ImageNet pretrained weights.")
    except Exception as e:
        print(f"⚠️ WARNING: Failed to load teacher checkpoint. Error: {e}. Using ImageNet pretrained weights.")

    # Freeze the teacher so only the student updates during KD.
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()


# ── KD 엔진 구성 ───────────────────────────────────
kd_engine = create_kd_engine(config.KD, teacher, student).to(device)

def _move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(o, device) for o in obj)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj

# --- build KD projections (dry-run) so their params are included in optimizer ---
imgs0, masks0 = next(iter(data_loader.train_loader))
imgs0 = _move_to_device(imgs0, device)
masks0 = masks0.to(device, non_blocking=True)
with torch.no_grad():
    dry_run_out = kd_engine.compute_losses(imgs0,
                                           masks0,
                                           device)

if "total" not in dry_run_out:
    raise KeyError("KD engine must return a 'total' loss entry.")

SCALAR_LOSS_KEYS = [
    key for key, value in dry_run_out.items() if _is_scalar_loss_value(value)
]

if "total" in SCALAR_LOSS_KEYS:
    SCALAR_LOSS_KEYS.remove("total")
    SCALAR_LOSS_KEYS.insert(0, "total")
else:
    SCALAR_LOSS_KEYS.insert(0, "total")

LOSS_KEY_TO_HEADER = {key: _display_name_for_loss(key) for key in SCALAR_LOSS_KEYS}

LOSS_HEADER_ORDER = [LOSS_KEY_TO_HEADER[key] for key in SCALAR_LOSS_KEYS]

print("▶ Tracking losses:", ", ".join(LOSS_KEY_TO_HEADER.values()))

# ── 옵티마이저/스케줄러 ─────────────────────────────
params = []
params += list(student.parameters())
params += list(kd_engine.get_extra_parameters())        # KD에서 projection에 이용되는 1x1 conv의 파라미터 추가
if not config.KD.FREEZE_TEACHER and config.KD.ENGINE_PARAMS.get('w_ce_teacher', 0.0) > 0.0:
    params += list(teacher.parameters())

optimizer_class = getattr(optim, config.TRAIN.OPTIMIZER["NAME"])
optimizer = optimizer_class(params, **config.TRAIN.OPTIMIZER["PARAMS"])

# -------------------------
# Scheduler: Warmup(epoch) + PolyLR(iter-update)
# -------------------------
ACCUM_STEPS = int(getattr(config.TRAIN, "ACCUM_STEPS", 1))
if ACCUM_STEPS < 1:
    raise ValueError(f"ACCUM_STEPS must be >= 1, got {ACCUM_STEPS}")

GRAD_CLIP_NORM = float(getattr(config.TRAIN, "GRAD_CLIP_NORM", 1.0))
USE_AMP = bool(getattr(config.TRAIN, "USE_AMP", True))
AMP_DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
scaler = GradScaler(enabled=bool(USE_AMP and AMP_DEVICE_TYPE == "cuda"))

# warm-up
if config.TRAIN.USE_WARMUP:
    warmup_class = getattr(optim.lr_scheduler, config.TRAIN.WARMUP_SCHEDULER["NAME"])
    # LinearLR의 total_iters 파라미터는 따로 계산하여 추가해줍니다.
    warmup_params = config.TRAIN.WARMUP_SCHEDULER["PARAMS"].copy()
    warmup_params["total_iters"] = config.TRAIN.WARMUP_EPOCHS
    warmup = warmup_class(optimizer, **warmup_params)
else:
    warmup = None

iters_per_epoch = len(data_loader.train_loader)
num_epochs = int(config.TRAIN.EPOCHS)
warmup_epochs = int(config.TRAIN.WARMUP_EPOCHS) if config.TRAIN.USE_WARMUP else 0
effective_epochs = max(num_epochs - warmup_epochs, 1)
total_optimizer_updates = (effective_epochs * iters_per_epoch + ACCUM_STEPS - 1) // ACCUM_STEPS

poly_params = getattr(config.TRAIN, "POLY_SCHEDULER", None)
POLY_POWER = float(poly_params.get("power", 0.9) if poly_params else 0.9)
POLY_END_LR = float(poly_params.get("end_learning_rate", 1e-6) if poly_params else 1e-6)

poly_scheduler = PolynomialLRDecay(
    optimizer,
    max_decay_steps=total_optimizer_updates,
    end_learning_rate=POLY_END_LR,
    power=POLY_POWER,
)

def _mask_to_grid(mask_tensor: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert integer masks (B,H,W) to a grid for TensorBoard logging."""
    if mask_tensor.dim() == 4 and mask_tensor.size(1) == 1:
        mask_tensor = mask_tensor.squeeze(1)
    mask_float = mask_tensor.float() / max(num_classes - 1, 1)
    mask_float = mask_float.unsqueeze(1)  # (B,1,H,W)
    return vutils.make_grid(mask_float, normalize=False)


def _log_segmentation_comparison(writer, out, masks, epoch, global_step, num_classes):
    """Log student/teacher inputs and predictions for qualitative KD inspection."""
    student_in = out.get("student_input")
    teacher_in = out.get("teacher_input")
    s_logits = out.get("s_logits")
    t_logits = out.get("t_logits")

    if student_in is not None:
        grid = vutils.make_grid(student_in, normalize=True, scale_each=True)
        writer.add_image("train/student_input", grid, global_step=global_step)

    if teacher_in is not None:
        grid = vutils.make_grid(teacher_in, normalize=True, scale_each=True)
        writer.add_image("train/teacher_input", grid, global_step=global_step)

    if s_logits is not None:
        s_pred = torch.argmax(s_logits, dim=1)
        grid = _mask_to_grid(s_pred, num_classes)
        writer.add_image("train/student_pred", grid, global_step=global_step)

    if t_logits is not None:
        t_pred = torch.argmax(t_logits, dim=1)
        grid = _mask_to_grid(t_pred, num_classes)
        writer.add_image("train/teacher_pred", grid, global_step=global_step)

    if masks is not None:
        gt_grid = _mask_to_grid(masks, num_classes)
        writer.add_image("train/ground_truth", gt_grid, global_step=global_step)

    if s_logits is not None and t_logits is not None:
        s_pred = torch.argmax(s_logits, dim=1)
        t_pred = torch.argmax(t_logits, dim=1)
        disagreement = (s_pred != t_pred).float().unsqueeze(1)
        diff_grid = vutils.make_grid(disagreement, normalize=True)
        writer.add_image("train/student_teacher_disagreement", diff_grid, global_step=global_step)


# 1epoch당 학습 방법 설정 후 loss값 반환
def train_one_epoch_kd(
    kd_engine,
    loader,
    optimizer,
    device,
    writer=None,
    epoch: int = 0,
    global_step_start: int = 0,
):
    kd_engine.train()
    if hasattr(kd_engine, "set_epoch"):
        try:
            kd_engine.set_epoch(epoch)
        except Exception:
            pass

    if not SCALAR_LOSS_KEYS:
        raise RuntimeError("No scalar losses registered from KD engine dry-run.")

    epoch_losses = {key: 0.0 for key in SCALAR_LOSS_KEYS}

    pbar = tqdm(loader, ascii=True, dynamic_ncols=True, leave=True, desc="Training")
    optimizer.zero_grad(set_to_none=True)
    opt_update_step = 0  # PolyLR step counter (optimizer update count within this epoch loop)

    for batch_idx, (imgs, masks) in enumerate(pbar):
        imgs = _move_to_device(imgs, device)
        masks = masks.to(device, non_blocking=True)

        # AMP forward (KD engine 내부에서 teacher/student forward 포함)
        amp_on = bool(USE_AMP and AMP_DEVICE_TYPE == "cuda")
        with autocast(AMP_DEVICE_TYPE, enabled=amp_on):
            out = kd_engine.compute_losses(imgs, masks, device)
            total_loss = out.get("total")
            if total_loss is None:
                raise KeyError("KD engine output does not contain 'total' loss.")
            if not isinstance(total_loss, torch.Tensor):
                raise TypeError("'total' loss must be a torch.Tensor for backpropagation.")

            loss_scaled = total_loss / ACCUM_STEPS

        if amp_on:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        do_update = ((batch_idx + 1) % ACCUM_STEPS == 0) or ((batch_idx + 1) == len(loader))
        if do_update:
            # grad clip: AMP면 unscale 후 clip
            if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
                if amp_on:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(kd_engine.parameters(), max_norm=GRAD_CLIP_NORM)

            if amp_on:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            # PolyLR: warmup epoch 동안에는 step 금지
            if (epoch > warmup_epochs) and (poly_scheduler is not None):
                poly_scheduler.step(opt_update_step + 1)
                opt_update_step += 1

        # 누적 loss (epoch 평균용)
        for key in epoch_losses:
            value = out.get(key)
            if value is None or not _is_scalar_loss_value(value):
                continue
            epoch_losses[key] += _loss_value_to_float(value)

        if writer is not None:
            global_step = global_step_start + batch_idx

            # 1) scalar logging (매 step)
            for key in SCALAR_LOSS_KEYS:
                value = out.get(key)
                if value is None or not _is_scalar_loss_value(value):
                    continue
                writer.add_scalar(f"train/{key}", _loss_value_to_float(value), global_step)

            # 2) image/hist logging (epoch당 1회: batch_idx==0)
            if batch_idx == 0:
                def _p(k):
                    v = out.get(k, None)
                    if v is None:
                        print(k, "MISSING");
                        return
                    if torch.is_tensor(v):
                        if v.dim() == 0:
                            print(k, float(v.item()))
                        else:
                            print(k, "shape", tuple(v.shape),
                                  "finite", torch.isfinite(v).all().item(),
                                  "min", float(torch.nanmin(v).item()),
                                  "max", float(torch.nanmax(v).item()))
                    else:
                        print(k, v)

                for k in ["total", "ce_student", "hf_kd", "hf_kd_mag", "hf_kd_grad",
                          "warmup_alpha_hf", "hf_weight_eff",
                          "mask_mean", "mask_min", "mask_max",
                          "t_hf_mean", "s_hf_mean"]:
                    _p(k)

                def _log_map(name: str, tensor: torch.Tensor | None):
                    if tensor is None:
                        return
                    t = tensor.detach()
                    if t.dim() == 3:
                        t = t.unsqueeze(1)  # (B,1,H,W)
                    t = t[:4]  # 너무 많이 찍지 않기
                    grid = vutils.make_grid(t, normalize=True, scale_each=True)
                    writer.add_image(f"train/{name}", grid, global_step=epoch)
                    writer.add_histogram(f"train/{name}_hist", t, global_step=epoch)

                # --- new engine keys (2-source attention) ---
                b_map = out.get("logit_boundary")      # (B,1,H,W)
                e_raw = out.get("swt_energy")          # (B,1,h,w)
                e_imp = out.get("swt_importance")      # (B,1,h,w)
                a_map = out.get("final_attention")     # (B,1,H,W)

                _log_map("logit_boundary_probgrad", b_map)
                _log_map("swt_energy_raw", e_raw)
                _log_map("swt_importance", e_imp)
                _log_map("final_attention", a_map)

                # --- NEW: SWTFPNHFFeatureKD debug keys (masked HF feature KD) ---
                # gates
                gate_boundary = out.get("gate_boundary")  # (B,1,H,W)
                gate_conf = out.get("gate_conf")          # (B,1,H,W)
                hf_mask = out.get("hf_mask")              # (B,1,H,W)
                _log_map("gate_boundary", gate_boundary)
                _log_map("gate_conf", gate_conf)
                _log_map("hf_mask", hf_mask)

                # fused HF maps
                t_hf_fpn = out.get("t_hf_fpn")            # (B,1,H,W)
                s_hf_fpn = out.get("s_hf_fpn")            # (B,1,H,W)
                _log_map("t_hf_fpn", t_hf_fpn)
                _log_map("s_hf_fpn", s_hf_fpn)
                if (t_hf_fpn is not None) and (s_hf_fpn is not None):
                    # absolute difference heatmap (teacher vs student HF)
                    hf_diff = (s_hf_fpn - t_hf_fpn).abs()
                    _log_map("hf_fpn_abs_diff", hf_diff)
                    if hf_mask is not None:
                        # masked diff: where we actually distill
                        if hf_mask.shape[-2:] != hf_diff.shape[-2:]:
                            hf_mask_r = F.interpolate(hf_mask, size=hf_diff.shape[-2:], mode="bilinear", align_corners=False)
                        else:
                            hf_mask_r = hf_mask
                        _log_map("hf_fpn_abs_diff_masked", hf_diff * hf_mask_r)

                # per-stage HF maps (loop through returned keys)
                # teacher stage keys: t_swt_hf_raw_s{idx}, t_swt_hf_imp_s{idx}
                # student stage keys: s_swt_hf_raw_s{idx}, s_swt_hf_imp_s{idx}
                for k in sorted(out.keys()):
                    if k.startswith("t_swt_hf_imp_s") or k.startswith("s_swt_hf_imp_s"):
                        _log_map(k, out.get(k))
                    if k.startswith("t_swt_hf_raw_s") or k.startswith("s_swt_hf_raw_s"):
                        _log_map(k, out.get(k))

                # --- basic qualitative inspection: inputs / preds / GT / disagreement ---
                _log_segmentation_comparison(
                    writer,
                    out,
                    masks,
                    epoch=epoch,
                    global_step=global_step,
                    num_classes=config.DATA.NUM_CLASSES,
                )

                # --- GT boundary + attention-on-boundary (optional but very useful) ---
                gt_1ch = masks.unsqueeze(1)  # (B,1,H,W)
                gx = (gt_1ch[:, :, :, :-1] != gt_1ch[:, :, :, 1:])
                gy = (gt_1ch[:, :, :-1, :] != gt_1ch[:, :, 1:, :])

                boundary = torch.zeros_like(gt_1ch, dtype=torch.bool)
                boundary[:, :, :, :-1] |= gx
                boundary[:, :, :-1, :] |= gy
                boundary = boundary.float()  # (B,1,H,W)

                writer.add_image(
                    "train/gt_boundary",
                    vutils.make_grid(boundary[:4], normalize=True),
                    global_step=epoch,
                )

                if a_map is not None:
                    bnd = boundary
                    if bnd.shape[-2:] != a_map.shape[-2:]:
                        bnd = F.interpolate(bnd, size=a_map.shape[-2:], mode="nearest")

                    writer.add_image(
                        "train/final_attention_on_gt_boundary",
                        vutils.make_grid((bnd[:4] * a_map[:4]).detach(), normalize=True, scale_each=True),
                        global_step=epoch,
                    )

                # --- error heatmap + attention-weighted error (optional) ---
                s_logits = out.get("s_logits")
                if s_logits is not None:
                    s_pred = torch.argmax(s_logits, dim=1)   # (B,H,W)
                    gt = masks                               # (B,H,W)
                    err_full = (s_pred != gt).float().unsqueeze(1)  # (B,1,H,W)

                    writer.add_image(
                        "train/gt_error",
                        vutils.make_grid(err_full[:4], normalize=True),
                        global_step=epoch,
                    )

                    if a_map is not None:
                        attn_full = a_map
                        if attn_full.shape[-2:] != err_full.shape[-2:]:
                            attn_full = F.interpolate(attn_full, size=err_full.shape[-2:], mode="bilinear", align_corners=False)

                        err_attn = err_full * attn_full
                        writer.add_image(
                            "train/gt_error_weighted_by_final_attention",
                            vutils.make_grid(err_attn[:4].detach(), normalize=True, scale_each=True),
                            global_step=epoch,
                        )

                        # quantile error stats (low vs high attention)
                        err_f = err_full.view(err_full.size(0), -1)
                        attn_f = attn_full.view(attn_full.size(0), -1)
                        q_low = torch.quantile(attn_f, 0.2, dim=1, keepdim=True)
                        q_high = torch.quantile(attn_f, 0.8, dim=1, keepdim=True)
                        low_mask = attn_f <= q_low
                        high_mask = attn_f >= q_high
                        if low_mask.any():
                            writer.add_scalar("analysis/error_low_attention_final", err_f[low_mask].mean().item(), epoch)
                        if high_mask.any():
                            writer.add_scalar("analysis/error_high_attention_final", err_f[high_mask].mean().item(), epoch)

                # --- NEW: HF-mask driven error analysis (optional, safe if keys missing) ---
                # Compare GT error rates in low vs high HF-mask regions (semantic edge distillation target).
                if s_logits is not None and hf_mask is not None:
                    s_pred = torch.argmax(s_logits, dim=1)  # (B,H,W)
                    gt = masks
                    err_full = (s_pred != gt).float().unsqueeze(1)  # (B,1,H,W)
                    m = hf_mask
                    if m.shape[-2:] != err_full.shape[-2:]:
                        m = F.interpolate(m, size=err_full.shape[-2:], mode="bilinear", align_corners=False)
                    m_f = m.view(m.size(0), -1)
                    err_f = err_full.view(err_full.size(0), -1)
                    q_low = torch.quantile(m_f, 0.2, dim=1, keepdim=True)
                    q_high = torch.quantile(m_f, 0.8, dim=1, keepdim=True)
                    low_mask = m_f <= q_low
                    high_mask = m_f >= q_high
                    if low_mask.any():
                        writer.add_scalar("analysis/error_low_hfmask", err_f[low_mask].mean().item(), epoch)
                    if high_mask.any():
                        writer.add_scalar("analysis/error_high_hfmask", err_f[high_mask].mean().item(), epoch)

        # tqdm postfix
        postfix = {}
        for key in SCALAR_LOSS_KEYS:
            value = out.get(key)
            if value is None or not _is_scalar_loss_value(value):
                continue
            postfix_label = LOSS_KEY_TO_HEADER.get(key, key)
            postfix[postfix_label] = f"{_loss_value_to_float(value):.3f}"
        if postfix:
            pbar.set_postfix(postfix)

    num_batches = len(loader)
    avg_losses = {key: val / num_batches for key, val in epoch_losses.items()}
    return avg_losses


# ── (변경) 검증(학생 기준) ─────────────────────────────────
def validate_student(student_model, loader, criterion):
    student_model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, ascii=True, dynamic_ncols=True, leave=True, desc="Validation"):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = student_model(imgs)  # logits
            total_loss += criterion(preds, masks).item()
    return total_loss / len(loader)

def plot_progress(epochs, train_losses, val_losses):
    plt.figure(figsize=(8,6))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses,   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True)
    plt.savefig(str(config.GENERAL.SAVE_PLOT), bbox_inches="tight")
    plt.close()

def write_summary(init=False, best_epoch=None, best_miou=None):
    # 기존 동일 (단, 모델명은 학생/교사 둘 표시 권장)
    with open(config.GENERAL.SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("=== Training Configuration ===\n")
        f.write(f"Dataset path : {config.DATA.DATA_DIR}\n")
        og = optimizer.param_groups[0]
        f.write(f"Student Model: {student.__class__.__name__}  (source: {config.KD.STUDENT_NAME})\n")
        f.write(f"Teacher Model: {teacher.__class__.__name__}  (source: {config.KD.TEACHER_NAME})\n\n")
        f.write(f"Teacher Freeze: {config.KD.FREEZE_TEACHER}\n")
        f.write(f"Optimizer     : {optimizer.__class__.__name__}\n")
        f.write(f"  lr           : {og['lr']}\n")
        f.write(f"  weight_decay : {og.get('weight_decay')}\n")
        # Scheduler summary: Warmup(epoch) + PolyLR(iter-update)
        if warmup is not None:
            f.write(
                f"Scheduler     : Warmup({warmup.__class__.__name__}, epochs={warmup_epochs}) "
                f"+ PolyLR({poly_scheduler.__class__.__name__})\n"
            )
        else:
            f.write(f"Scheduler     : PolyLR({poly_scheduler.__class__.__name__})\n")
        f.write(f"  Poly max_decay_steps : {total_optimizer_updates}\n")
        f.write(f"  Poly power           : {POLY_POWER}\n")
        f.write(f"  Poly end_lr          : {POLY_END_LR}\n")
        f.write(f"AMP           : {USE_AMP} (device_type={AMP_DEVICE_TYPE})\n")
        f.write(f"Accum steps   : {ACCUM_STEPS}\n")
        f.write(f"Batch size    : {config.DATA.BATCH_SIZE}\n\n")
        f.write("=== Knowledge Distillation Configuration ===\n")
        f.write(f"Engine NAME        : {config.KD.ENGINE_NAME}\n")
        f.write(f"Teacher Source CKPT: {config.TEACHER_CKPT}\n\n")
        # 1. 현재 설정된 엔진의 파라미터 딕셔너리를 가져옵니다.
        engine_name = config.KD.ENGINE_NAME
        current_engine_params = config.KD.ALL_ENGINE_PARAMS.get(engine_name, {})
        f.write(f"--- Parameters for '{engine_name}' engine ---\n")
        # 2. 가져온 딕셔너리를 반복하면서 모든 파라미터를 자동으로 기록합니다.
        if not current_engine_params:
            f.write("No parameters found for this engine.\n")
        else:
            for key, value in current_engine_params.items():
                # 보기 좋게 정렬하기 위해 key 문자열의 길이를 25로 맞춥니다.
                f.write(f"{key:<25} : {value}\n")
        f.write("\n")

        if init:
            f.write("=== Best Model (to be updated) ===\n")
            f.write("epoch     : N/A\nbest_test_mIoU : N/A\n\n")
        else:
            f.write("=== Best Model ===\n")
            f.write(f"epoch     : {best_epoch}\n")
            f.write(f"best_test_mIoU : {best_miou:.4f}\n\n")

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

# 학습 진행 및 잘 되고있나 성능평가
def run_training(num_epochs):
    write_summary(init=True)
    start_dt = datetime.now()
    print(f"Started at : {start_dt:%Y-%m-%d %H:%M:%S}")

    best_miou = -float("inf")
    best_epoch = 0
    best_ckpt = config.GENERAL.BASE_DIR / "best_model.pth"

    # 마지막 5 epoch만 테스트 세트 평가에 사용 (총 epoch이 5보다 작으면 전체 평가)
    test_eval_start_epoch = max(1, num_epochs - 4)

    # CSV 로그 파일 경로 설정 및 헤더 생성
    log_csv_path = config.GENERAL.LOG_DIR / "training_log.csv"
    loss_headers = LOSS_HEADER_ORDER if LOSS_HEADER_ORDER else ["Total Loss"]
    csv_headers = ["Epoch", *loss_headers, "Val Loss", "Val mIoU", "Pixel Acc", "LR", "Test mIoU", "Test Pixel Acc"]
    # 클래스별 IoU 헤더 추가
    for i in range(config.DATA.NUM_CLASSES):
        csv_headers.append(f"IoU_{config.DATA.CLASS_NAMES[i]}")

    # 파일이 없으면 헤더를 포함하여 새로 생성
    if log_csv_path.exists():
        try:
            existing_cols = list(pd.read_csv(log_csv_path, nrows=0).columns)
            if len(existing_cols) > 0:
                csv_headers = existing_cols  # 기존 헤더 신뢰
        except Exception:
            pass
    else:
        # 새 파일 생성 시에만 헤더를 기록
        pd.DataFrame(columns=csv_headers).to_csv(log_csv_path, index=False)

    train_losses, val_losses = [], []
    writer = SummaryWriter(log_dir=config.GENERAL.LOG_DIR)
    loss_class = getattr(nn, config.TRAIN.LOSS_FN["NAME"])
    criterion = loss_class(**config.TRAIN.LOSS_FN["PARAMS"])

    global_step = 0
    for epoch in range(1, num_epochs + 1):
        # train_one_epoch_kd는 이제 손실 딕셔너리를 반환
        tr_losses_dict = train_one_epoch_kd(
            kd_engine,
            data_loader.train_loader,
            optimizer,
            device,
            writer=writer,
            epoch=epoch,
            global_step_start=global_step,
        )
        tr_loss = tr_losses_dict["total"]  # plot을 위한 total loss
        global_step += len(data_loader.train_loader)

        vl_loss = validate_student(student, data_loader.val_loader, criterion)
        metrics = evaluate.evaluate_all(model, data_loader.val_loader, device)
        miou = metrics["mIoU"]
        pa = metrics["PixelAcc"]

        test_miou = float("nan")
        test_pa = float("nan")
        if epoch >= test_eval_start_epoch:
            test_metrics = evaluate.evaluate_all(model, data_loader.test_loader, device)
            test_miou = test_metrics["mIoU"]
            test_pa = test_metrics["PixelAcc"]

        # Warmup: epoch 단위로만 step
        if warmup is not None and epoch <= warmup_epochs:
            warmup.step()

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        print(f"[{epoch}/{num_epochs}] "
              f"train_loss={tr_loss:.4f}, val_loss={vl_loss:.4f}, "
              f"val_mIoU={miou:.4f},  PA={pa:.4f}, "
              f"test_mIoU={test_miou:.4f}, test_PA={test_pa:.4f}")

        current_lr = optimizer.param_groups[0]['lr']

        if writer is not None:
            writer.add_scalar("val/loss", vl_loss, epoch)
            writer.add_scalar("val/mIoU", miou, epoch)
            writer.add_scalar("val/PixelAcc", pa, epoch)
            writer.add_scalar("lr", current_lr, epoch)
            if epoch >= test_eval_start_epoch:
                writer.add_scalar("test/mIoU", test_miou, epoch)
                writer.add_scalar("test/PixelAcc", test_pa, epoch)
        # CSV 파일에 성능 지표 기록
        log_data = {"Epoch": epoch}
        for key in SCALAR_LOSS_KEYS:
            header_name = LOSS_KEY_TO_HEADER.get(key, key)
            log_data[header_name] = tr_losses_dict.get(key, float("nan"))

        log_data.update({
            "Val Loss": vl_loss,
            "Val mIoU": miou,
            "Pixel Acc": pa,
            "LR": current_lr,
            "Test mIoU": test_miou,
            "Test Pixel Acc": test_pa,
        })
        # 클래스별 IoU를 log_data 딕셔너리에 추가
        per_cls_iou = metrics["per_class_iou"]
        for i in range(config.DATA.NUM_CLASSES):
            log_data[f"IoU_{config.DATA.CLASS_NAMES[i]}"] = float(per_cls_iou[i])


        # DataFrame으로 변환 후 CSV 파일에 append
        df_new_row = pd.DataFrame([log_data]).reindex(columns=csv_headers)
        df_new_row.to_csv(log_csv_path, mode='a', header=False, index=False)

        # best model 갱신 시 로그 기록
        # 마지막 5 epoch에 대해 테스트 mIoU 기반으로 best model 선정
        if epoch >= test_eval_start_epoch and test_miou > best_miou:
            best_miou = test_miou
            best_epoch = epoch

            # 모델 체크포인트 저장
            torch.save({
                "epoch": epoch,
                "model_state": student.state_dict(),
                "teacher_state": teacher.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "warmup_state": (warmup.state_dict() if warmup is not None else None),
                "poly_state": (poly_scheduler.state_dict() if poly_scheduler is not None else None),
                "use_amp": USE_AMP,
                "accum_steps": ACCUM_STEPS,
                "best_test_mIoU": best_miou
            }, best_ckpt)
            print(f"▶ New best test_mIoU at epoch {epoch}: {test_miou:.4f} → {best_ckpt}")
            write_summary(init=False, best_epoch=best_epoch, best_miou=best_miou)

        if epoch % 10 == 0:
            plot_progress(list(range(1, epoch + 1)), train_losses, val_losses)

    end_dt = datetime.now()
    writer.flush()
    writer.close()
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
    print(f"Best epoch (test mIoU): {best_epoch}, Best test_mIoU: {best_miou:.4f}")

    return best_ckpt

if __name__ == "__main__":
    run_training(config.TRAIN.EPOCHS)