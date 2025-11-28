# vis_swt_energy.py
#
# Teacher feature -> Haar SWT energy 를 이용해서
#   (1) 입력 이미지 (LR or HR)
#   (2) GT segmentation (color)
#   (3) SWT energy heatmap
#   (4) RGB + energy overlay
# 를 저장하는 qualitative 시각화 스크립트.

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import config
import data_loader
from models import create_model

# 필요하다면, 기존 SWTTunedKDEngine에서 HaarSWT만 import해도 됨
# from kd_engines.swt_tuned_kd import HaarSWT

# ---------------- HaarSWT + energy 함수 (SWTTunedKDEngine과 동일) ----------------

class HaarSWT(nn.Module):
    """Lightweight 2×2 Haar SWT applied per-channel."""
    def __init__(self, in_channels: int):
        super().__init__()
        k_ll = torch.tensor([[0.25, 0.25], [0.25, 0.25]])
        k_lh = torch.tensor([[0.25, -0.25], [0.25, -0.25]])
        k_hl = torch.tensor([[0.25, 0.25], [-0.25, -0.25]])
        k_hh = torch.tensor([[0.25, -0.25], [-0.25, 0.25]])

        filters = torch.stack([k_ll, k_lh, k_hl, k_hh], dim=0).unsqueeze(1)
        self.register_buffer("filters", filters.repeat(in_channels, 1, 1, 1))
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels but got {x.shape[1]}")
        x_pad = F.pad(x, (0, 1, 0, 1), mode="reflect")
        out = F.conv2d(x_pad, self.filters, stride=1, padding=0, groups=self.in_channels)
        return out.view(x.shape[0], self.in_channels, 4, x.shape[2], x.shape[3])


@torch.no_grad()
def compute_swt_energy(feat: torch.Tensor, energy_temperature: float = 1.5):
    """
    feat: (B, C, H, W) teacher feature
    return:
      energy: (B,1,H,W)
      attn  : (B,1,H,W) softmax attention (필요시)
    """
    swt = HaarSWT(feat.shape[1]).to(feat.device, feat.dtype)
    swt_out = swt(feat)
    lh = swt_out[:, :, 1]
    hl = swt_out[:, :, 2]
    hh = swt_out[:, :, 3]
    energy = (lh.abs() + hl.abs() + hh.abs()).mean(dim=1, keepdim=True)  # (B,1,H,W)
    norm_energy = energy / (energy.mean(dim=[1, 2, 3], keepdim=True) + 1e-6)
    attn = torch.softmax(norm_energy.flatten(2) * energy_temperature, dim=-1)
    attn = attn.view_as(norm_energy)
    return energy, attn


# ----------------- GT color decode (이미 가지고 있던 방식 재사용) -----------------

def decode_segmap(label_mask: np.ndarray) -> np.ndarray:
    """
    label_mask: 2D numpy array (H,W), [0..n_classes-1]
    return: (H,W,3) uint8
    """
    return config.DATA.CLASS_COLORS[label_mask]


# ----------------- 메인: 몇 장 뽑아서 figure 저장 -----------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Teacher 모델 로드 (train_kd와 동일한 방식)
    teacher = create_model(config.KD.TEACHER_NAME).to(device)
    teacher_ckpt = Path(config.TEACHER_CKPT)
    if teacher_ckpt.exists():
        ckpt = torch.load(teacher_ckpt, map_location=device, weights_only=False)
        if "model_state" in ckpt:
            teacher.load_state_dict(ckpt["model_state"])
            print(f"Loaded teacher ckpt from {teacher_ckpt} (model_state)")
        else:
            teacher.load_state_dict(ckpt)
            print(f"Loaded teacher ckpt from {teacher_ckpt} (direct dict)")
    else:
        print(f"[WARN] Teacher ckpt not found at {teacher_ckpt}. Using ImageNet weights.")

    teacher.eval()

    # val_loader 불러오기 (data_loader는 main_kd / train_kd와 동일한 모듈)
    val_loader = data_loader.val_loader

    # 저장 경로
    out_dir = Path(config.GENERAL.BASE_DIR) / "qualitative_swt"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Save dir: {out_dir}")

    num_samples_to_save = 8  # 원하는 만큼 조절
    saved = 0

    with torch.no_grad():
        for batch_idx, (imgs, masks) in enumerate(val_loader):
            imgs = imgs.to(device)            # (B,3,H,W) - 현재 pipeline 기준 (LR or HR)
            masks = masks.to(device)          # (B,H,W)

            # teacher forward: logits + features
            out = teacher(imgs, return_feats=True)
            if isinstance(out, tuple):
                if len(out) == 2:
                    t_logits, t_feats = out
                else:
                    t_logits, t_feats = out[0], out[1:]
            elif isinstance(out, dict):
                t_logits, t_feats = out.get("logits"), out.get("feats")
            else:
                t_logits, t_feats = out

            if isinstance(t_feats, torch.Tensor):
                t_feats = (t_feats,)
            else:
                t_feats = tuple(t_feats)

            # teacher_stage는 config나 SWTTunedKDEngine에서 쓰는 것과 맞춰라 (예: -2)
            teacher_stage = getattr(config.KD, "ENGINE_PARAMS", {}).get("teacher_stage", -2)
            if teacher_stage < 0:
                idx = len(t_feats) + teacher_stage
            else:
                idx = teacher_stage
            t_feat_sel = t_feats[idx]   # (B,C,H,W)

            # SWT energy 계산
            energy, attn = compute_swt_energy(t_feat_sel)

            # 해상도 맞추기 (feature 해상도 != 입력 해상도일 가능성)
            if energy.shape[-2:] != imgs.shape[-2:]:
                energy_up = F.interpolate(energy, size=imgs.shape[-2:], mode="bilinear", align_corners=False)
            else:
                energy_up = energy

            # 배치에서 몇 장만 뽑아서 저장
            B = imgs.shape[0]
            for i in range(B):
                if saved >= num_samples_to_save:
                    break

                img_i   = imgs[i].detach().cpu()         # (3,H,W)
                mask_i  = masks[i].detach().cpu().numpy()  # (H,W)
                en_i    = energy_up[i, 0].detach().cpu().numpy()  # (H,W)

                # --- 역정규화(unnormalize) ---
                # config에 맞게 사용 (CamVid면 보통 ImageNet 또는 직접 정의한 값일 것)
                # 예시) config.DATA.MEAN = [0.485, 0.456, 0.406]
                #      config.DATA.STD  = [0.229, 0.224, 0.225]
                try:
                    mean = torch.tensor(config.DATA.MEAN).view(3, 1, 1)
                    std = torch.tensor(config.DATA.STD).view(3, 1, 1)
                    img_denorm = img_i * std + mean
                except AttributeError:
                    # 혹시 MEAN/STD를 config에 안 넣었으면, 실제 transform에 맞게 직접 넣어라.
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img_denorm = img_i * std + mean

                rgb = torch.clamp(img_denorm, 0.0, 1.0).permute(1, 2, 0).numpy()  # (H,W,3)

                # GT color
                gt_color = decode_segmap(mask_i)  # (H,W,3), uint8

                # energy normalize (0~1)
                en_min, en_max = float(en_i.min()), float(en_i.max())
                en_vis = (en_i - en_min) / (en_max - en_min + 1e-6)

                # figure 생성
                fig, axes = plt.subplots(2, 2, figsize=(8, 8))

                axes[0,0].imshow(rgb)
                axes[0,0].set_title("Input (LR or HR)")
                axes[0,0].axis("off")

                axes[0,1].imshow(gt_color)
                axes[0,1].set_title("GT Segmentation")
                axes[0,1].axis("off")

                im2 = axes[1,0].imshow(en_vis, cmap="magma")
                axes[1,0].set_title("SWT Energy")
                axes[1,0].axis("off")
                fig.colorbar(im2, ax=axes[1,0], fraction=0.046, pad=0.04)

                axes[1,1].imshow(rgb)
                axes[1,1].imshow(en_vis, cmap="magma", alpha=0.5)
                axes[1,1].set_title("Energy Overlay")
                axes[1,1].axis("off")

                fig.tight_layout()

                save_path = out_dir / f"swt_qual_{saved:03d}.png"
                fig.savefig(save_path, dpi=200, bbox_inches="tight")
                plt.close(fig)

                print(f"Saved: {save_path}")
                saved += 1

            if saved >= num_samples_to_save:
                break

    print(f"Done. Total saved: {saved}")


if __name__ == "__main__":
    main()
