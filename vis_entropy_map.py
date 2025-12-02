from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import config
from data_loader import test_loader, val_loader
from models.segformer_wrapper import SegFormerWrapper


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CKPT_PATH = Path(r"D:\LAB\result_files\test_results\Aset_LR_d3pmv2_150epoch\best_model.pth")
SPLIT = "val"  # or "test"
NUM_SAMPLES = 8
SAVE_DIR = Path(r"D:\LAB\result_files\analysis_results\entropy_maps\d3pmv2_LR")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def denormalize(img: torch.Tensor) -> np.ndarray:
    """Convert a normalized tensor (3,H,W) back to uint8 HWC."""
    np_img = img.detach().cpu().numpy()
    np_img = (np_img * STD[:, None, None]) + MEAN[:, None, None]
    np_img = np.clip(np_img, 0.0, 1.0)
    return (np_img.transpose(1, 2, 0) * 255.0).astype(np.uint8)


def mask_to_color(mask: torch.Tensor) -> np.ndarray:
    """Map class indices to RGB colors using DATA.CLASS_COLORS."""
    palette = config.DATA.CLASS_COLORS
    mask_np = mask.detach().cpu().numpy()
    mask_np = np.clip(mask_np, 0, len(palette) - 1)
    return palette[mask_np]


def load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    model = SegFormerWrapper(name=config.MODEL.NAME, num_classes=config.DATA.NUM_CLASSES)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            # 너가 학습에 사용한 train/main 코드에서 저장한 포맷
            state = ckpt["model_state"]
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            # 이미 state_dict만 저장된 경우
            state = ckpt
    else:
        state = ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load_model] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    if missing:
        print("  some missing keys (check if they are really needed):")
        print("  ", list(missing)[:10], "...")
    if unexpected:
        print("  some unexpected keys:")
        print("  ", list(unexpected)[:10], "...")

    model.to(device)
    model.eval()
    return model



def compute_entropy_map(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1, keepdim=True)
    return entropy


def visualize_sample(
    img: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    entropy: torch.Tensor,
    idx: int,
    save_dir: Path,
) -> None:
    img_np = denormalize(img)
    gt_rgb = mask_to_color(gt_mask)
    pred_rgb = mask_to_color(pred_mask)

    ent_np = entropy.detach().cpu().numpy()
    ent_min, ent_max = ent_np.min(), ent_np.max()
    ent_norm = (
        (ent_np - ent_min) / (ent_max - ent_min + 1e-8)
        if ent_max > ent_min
        else np.zeros_like(ent_np)
    )

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # 1) Input image
    axes[0].imshow(img_np)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # 2) GT
    axes[1].imshow(gt_rgb)
    axes[1].set_title("GT Mask")
    axes[1].axis("off")

    # 3) Prediction
    axes[2].imshow(pred_rgb)
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")

    # 4) Entropy
    im = axes[3].imshow(ent_norm, cmap="magma")
    axes[3].set_title("Entropy Map")
    axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    save_path = save_dir / f"entropy_{idx:04d}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")



def get_loader(split: str):
    if split == "val":
        return val_loader
    if split == "test":
        return test_loader
    raise ValueError(f"Unsupported split: {split}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶ Using device: {device}")
    print(f"▶ Loading checkpoint: {CKPT_PATH}")
    print(f"▶ Split: {SPLIT} | Samples: {NUM_SAMPLES} | Save dir: {SAVE_DIR}")

    model = load_model(CKPT_PATH, device)
    loader = get_loader(SPLIT)

    with torch.no_grad():
        count = 0
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)  # (N, C, H, W)
            entropy = compute_entropy_map(logits)
            preds = torch.argmax(logits, dim=1)  # (N, H, W)

            for b in range(imgs.size(0)):
                visualize_sample(
                    img=imgs[b],
                    gt_mask=masks[b],
                    pred_mask=preds[b],
                    entropy=entropy[b, 0],
                    idx=count,
                    save_dir=SAVE_DIR,
                )
                count += 1
                if count >= NUM_SAMPLES:
                    return


if __name__ == "__main__":
    main()