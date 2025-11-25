import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

# [수정] descriptor_v2에서 임포트
from descriptor_v3 import FrequencyDegradationTarget, DegradationDescriptorNet

# =========================
# 1. 경로 및 하이퍼파라미터 설정
# =========================

# 상위 폴더 정의
HR_ROOT = Path(r"D:\LAB\datasets\project_use\CamVid_12_2Fold_v4\B_set")
LR_ROOT = Path(r"D:\LAB\datasets\project_use\CamVid_12_2Fold_LR_x4_Bilinear\B_set")

# Train 경로
TRAIN_HR_DIR = HR_ROOT / "train" / "images"
TRAIN_LR_DIR = LR_ROOT / "train" / "images"

# Val 경로
VAL_HR_DIR = HR_ROOT / "val" / "images"
VAL_LR_DIR = LR_ROOT / "val" / "images"

# HR / LR 리사이즈 크기 (H, W)
HR_SIZE: Tuple[int, int] = (360, 480)
LR_SIZE: Tuple[int, int] = (360, 480)

# 학습 설정
BATCH_SIZE: int = 8
NUM_EPOCHS: int = 200
LEARNING_RATE: float = 1e-4
NUM_WORKERS: int = 4

# [수정] 모델 저장 경로 (Wavelet 버전 구분)
SAVE_PATH: Path = Path(r"D:\LAB\result_files\des_SWT_module_Bset\descriptor_net.pth")


# =========================
# 2. 유틸리티 함수
# =========================

def plot_loss_curve(epochs, train_losses, val_losses, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(epochs, train_losses, label="train_loss")
    plt.plot(epochs, val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (L1)")
    plt.title("Train / Val Loss Curve (Log-Wavelet Target)")
    plt.legend()
    plt.grid(True)
    plt.savefig(str(out_dir / "loss_curve.png"))
    plt.close()


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(IMG_EXTENSIONS)


# =========================
# 3. Dataset 정의
# =========================

class HRLRImageDataset(Dataset):
    def __init__(
            self,
            hr_dir: Path,
            lr_dir: Path,
            hr_size: Tuple[int, int],
            lr_size: Tuple[int, int],
    ):
        super().__init__()
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_size = hr_size
        self.lr_size = lr_size

        if not hr_dir.exists():
            raise RuntimeError(f"[Error] HR directory does not exist: {hr_dir}")
        if not lr_dir.exists():
            raise RuntimeError(f"[Error] LR directory does not exist: {lr_dir}")

        hr_files = sorted([f for f in os.listdir(hr_dir) if is_image_file(f)])
        lr_files = sorted([f for f in os.listdir(lr_dir) if is_image_file(f)])

        hr_set = set(hr_files)
        lr_set = set(lr_files)
        common = sorted(list(hr_set.intersection(lr_set)))

        if len(common) == 0:
            raise RuntimeError(f"[Error] No common files found between:\n  {hr_dir}\n  {lr_dir}")

        if len(common) != len(hr_files) or len(common) != len(lr_files):
            print(f"[Warning] File count mismatch! Common: {len(common)}")

        self.filenames = common

        self.hr_transform = transforms.Compose([
            transforms.Resize(hr_size, interpolation=Image.BILINEAR),
            transforms.ToTensor()
        ])
        self.lr_transform = transforms.Compose([
            transforms.Resize(lr_size, interpolation=Image.BILINEAR),
            transforms.ToTensor()
        ])

        print(f"[Dataset] Loaded {len(self.filenames)} pairs from .../{hr_dir.parent.name}/{hr_dir.name}")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        fname = self.filenames[idx]
        hr_path = self.hr_dir / fname
        lr_path = self.lr_dir / fname

        hr_img = Image.open(hr_path).convert("RGB")
        lr_img = Image.open(lr_path).convert("RGB")

        hr_tensor = self.hr_transform(hr_img)
        lr_tensor = self.lr_transform(lr_img)

        return hr_tensor, lr_tensor, fname


# =========================
# 4. 학습 및 검증 함수
# =========================

def train_one_epoch(model, target_builder, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for hr, lr, _ in loader:
        hr, lr = hr.to(device), lr.to(device)

        with torch.no_grad():
            target = target_builder(hr, lr)  # Log-Wavelet Target

        pred = model(lr)
        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * hr.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, target_builder, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    sum_target = torch.zeros(3, device=device)
    sum_pred = torch.zeros(3, device=device)
    n_samples = 0

    for hr, lr, _ in loader:
        hr, lr = hr.to(device), lr.to(device)

        target = target_builder(hr, lr)
        pred = model(lr)
        loss = criterion(pred, target)

        total_loss += loss.item() * hr.size(0)
        sum_target += target.sum(dim=0)
        sum_pred += pred.sum(dim=0)
        n_samples += hr.size(0)

    avg_loss = total_loss / max(1, n_samples)
    avg_target = sum_target / max(1, n_samples)
    avg_pred = sum_pred / max(1, n_samples)
    return avg_loss, avg_target, avg_pred


# =========================
# 5. 메인 실행부
# =========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ---------------------------
    # 데이터셋 로드
    # ---------------------------
    print("\n[Data] Initializing Train Dataset...")
    train_dataset = HRLRImageDataset(TRAIN_HR_DIR, TRAIN_LR_DIR, HR_SIZE, LR_SIZE)

    print("[Data] Initializing Val Dataset...")
    val_dataset = HRLRImageDataset(VAL_HR_DIR, VAL_LR_DIR, HR_SIZE, LR_SIZE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # ---------------------------
    # 모델 및 학습 설정
    # ---------------------------
    descriptor_net = DegradationDescriptorNet(in_channels=3, num_bands=3).to(device)

    # [수정] Wavelet Target은 num_bands, cut 파라미터가 필요 없음 (구조적 고정)
    target_builder = FrequencyDegradationTarget(eps=1e-6)

    optimizer = torch.optim.Adam(descriptor_net.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss()  # Log scale이므로 L1 Loss가 안정적임

    print(f"\n[Train] Start training for {NUM_EPOCHS} epochs (Wavelet V2)...")

    best_val_loss = float("inf")
    epoch_indices = []
    train_losses = []
    val_losses = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(
            descriptor_net, target_builder, train_loader, optimizer, criterion, device
        )

        val_loss, avg_tgt, avg_pred = evaluate(
            descriptor_net, target_builder, val_loader, criterion, device
        )

        print(f"[Epoch {epoch:03d}] Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
        tgt_np = avg_tgt.cpu().numpy()
        pred_np = avg_pred.cpu().numpy()
        print(f"    Target(Avg): [{tgt_np[0]:.3f}, {tgt_np[1]:.3f}, {tgt_np[2]:.3f}]")
        print(f"    Pred(Avg)  : [{pred_np[0]:.3f}, {pred_np[1]:.3f}, {pred_np[2]:.3f}]")

        epoch_indices.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state": descriptor_net.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss
            }, str(SAVE_PATH))
            print(f"    [*] Best model saved: {SAVE_PATH.name}")

    plot_loss_curve(epoch_indices, train_losses, val_losses, SAVE_PATH.parent)
    print("\n[Done] Training finished.")


if __name__ == "__main__":
    main()