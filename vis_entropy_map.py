# vis_entropy_map.py
#
# SegFormerWrapper + 저장된 .pth를 이용해서
#   - 입력 이미지
#   - GT mask
#   - logit의 entropy map
# 을 시각화하는 스크립트 예시.
#
# TODO라고 적힌 부분은 프로젝트 환경에 맞게 수정 필요.

import os
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# TODO: 프로젝트 구조에 맞게 import 수정
import config
import data_loader
from models.segformer_wrapper import SegFormerWrapper   # 실제 경로 확인


# ================= 사용자 설정 =================
CKPT_PATH = Path(
    r"D:\LAB\result_files\test_results\Aset_HR_segb3\best_model.pth"
)  # TODO: 실제 ckpt 경로로 변경
SAVE_DIR = Path(
    r"D:\LAB\result_files\analysis_results\entropy_maps"
)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

NUM_SAMPLES = 8  # 몇 장 시각화할지


# CamVid 기준 예시. 실제로는 config나 data_loader에서 사용하는 값으로 맞추는 것이 안전함.
IMG_MEAN = np.array([0.411, 0.432, 0.450], dtype=np.float32)
IMG_STD = np.array([0.275, 0.273, 0.278], dtype=np.float32)


def denormalize(t: torch.Tensor) -> np.ndarray:
    """
    t: (3,H,W), normalized
    return: (H,W,3), uint8
    """
    x = t.detach().cpu().numpy()
    x = (x * IMG_STD[:, None, None]) + IMG_MEAN[:, None, None]
    x = np.clip(x, 0.0, 1.0)
    x = (x.transpose(1, 2, 0) * 255.0).astype(np.uint8)  # CHW -> HWC
    return x


def build_dataloader():
    """
    프로젝트의 data_loader에 맞게 validation/test loader를 구성.
    예시는 'val' split 기준. 실제 함수명/인자에 맞게 수정 필요.
    """
    # 예시 코드: 실제 구현에 맞게 교체
    # return data_loader.get_loader(split="val", config=config)
    return data_loader.get_loader(
        split="val",  # 혹은 "test"
        batch_size=1,
        shuffle=False,
        num_workers=4,
        config=config,
    )


def build_model(device: torch.device) -> torch.nn.Module:
    """
    SegFormerWrapper 생성 후 .pth 로드.
    - SegFormerWrapper의 생성자 인자(backbone, num_classes 등)는
      프로젝트의 실제 정의에 맞게 수정 필요.
    """
    num_classes = config.DATA.NUM_CLASSES  # 예시
    backbone = getattr(config.MODEL, "BACKBONE", "mit_b3")  # 예시

    model = SegFormerWrapper(
        backbone=backbone,
        num_classes=num_classes,
    )

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    # 저장 형식에 따라 key 다를 수 있으므로 예외 처리
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def compute_entropy_map(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: (N, C, H, W)
    return: entropy_map: (N, 1, H, W), 값 범위는 [0, log(C)] 근처
    """
    # p: 소프트맥스 확률
    probs = F.softmax(logits, dim=1)
    # 엔트로피: -sum p log p
    entropy = -torch.sum(
        probs * torch.log(probs + 1e-8),
        dim=1,
        keepdim=True,
    )  # (N,1,H,W)
    return entropy


def visualize_sample(
    img_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    entropy_map: torch.Tensor,
    idx: int,
):
    """
    img_tensor: (3,H,W), normalized
    mask_tensor: (H,W), long
    entropy_map: (H,W), float
    """
    img_np = denormalize(img_tensor)
    mask_np = mask_tensor.detach().cpu().numpy()
    ent_np = entropy_map.detach().cpu().numpy()

    # entropy를 0~1로 정규화 (시각화용)
    ent_min, ent_max = ent_np.min(), ent_np.max()
    if ent_max > ent_min:
        ent_norm = (ent_np - ent_min) / (ent_max - ent_min + 1e-8)
    else:
        ent_norm = np.zeros_like(ent_np)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1) 입력 이미지
    axes[0].imshow(img_np)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # 2) GT mask (간단하게 cmap만)
    num_classes = int(mask_np.max()) + 1
    axes[1].imshow(mask_np, vmin=0, vmax=max(num_classes - 1, 1), cmap="tab20")
    axes[1].set_title("GT Mask")
    axes[1].axis("off")

    # 3) Entropy heatmap
    im = axes[2].imshow(ent_norm, cmap="magma")
    axes[2].set_title("Entropy Map (pixel-wise)")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    out_path = SAVE_DIR / f"entropy_{idx:04d}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶ Using device: {device}")

    loader = build_dataloader()
    model = build_model(device)

    model.eval()
    with torch.no_grad():
        count = 0
        for batch in loader:
            # TODO: batch 구조에 맞게 수정
            # 일반적으로: batch["image"] : (N,3,H,W), batch["mask"] : (N,H,W)
            if isinstance(batch, (list, tuple)):
                imgs, masks = batch[:2]
            elif isinstance(batch, dict):
                imgs, masks = batch["image"], batch["mask"]
            else:
                raise ValueError("지원하지 않는 batch 형식")

            imgs = imgs.to(device)               # (N,3,H,W)
            masks = masks.to(device)             # (N,H,W)

            # SegFormerWrapper 출력이 (logits) or (logits, aux...) 인지에 따라 분기
            out = model(imgs)
            if isinstance(out, (list, tuple)):
                logits = out[0]
            elif isinstance(out, dict) and "logits" in out:
                logits = out["logits"]
            else:
                logits = out  # (N,C,H,W) 라고 가정

            # entropy map 계산
            entropy = compute_entropy_map(logits)  # (N,1,H,W)

            # 배치 내 각각 시각화
            for b in range(imgs.size(0)):
                img_b = imgs[b]                  # (3,H,W)
                mask_b = masks[b]                # (H,W)
                ent_b = entropy[b, 0]            # (H,W)

                visualize_sample(img_b, mask_b, ent_b, idx=count)
                count += 1

                if count >= NUM_SAMPLES:
                    return


if __name__ == "__main__":
    main()
