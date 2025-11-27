import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from PIL import Image
from tqdm import tqdm  # Progress bar

# ==========================================
# 1. Configuration (사용자 수정 필요)
# ==========================================
# 검증할 이미지들이 들어있는 폴더 경로를 입력하세요.
# 예: "./data/cityscapes/leftImg8bit/train" 또는 "./my_lr_dataset"
DATASET_PATH = r"D:\LAB\datasets\project_use\CamVid_12_2Fold_LR_x4_Bilinear\all\images"
IMAGE_EXTENSIONS = ['*.jpg', '*.png', '*.jpeg']  # 이미지 확장자


# ==========================================
# 2. HaarSWT Logic (동일 로직 사용)
# ==========================================
class HaarSWT(nn.Module):
    def __init__(self, in_channels=1):  # 분석용이므로 GrayScale(1) 기준
        super().__init__()
        k_ll = torch.tensor([[0.25, 0.25], [0.25, 0.25]])
        k_lh = torch.tensor([[0.25, -0.25], [0.25, -0.25]])
        k_hl = torch.tensor([[0.25, 0.25], [-0.25, -0.25]])
        k_hh = torch.tensor([[0.25, -0.25], [-0.25, 0.25]])

        filters = torch.stack([k_ll, k_lh, k_hl, k_hh], dim=0).unsqueeze(1)
        self.register_buffer('filters', filters)

    def forward(self, x):
        x_pad = F.pad(x, (0, 1, 0, 1), mode='reflect')
        out = F.conv2d(x_pad, self.filters, stride=1, groups=1)
        return out


# ==========================================
# 3. Analysis Function
# ==========================================
def analyze_dataset(folder_path, device='cuda'):
    print(f"[Info] Scanning dataset at: {folder_path}")
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(folder_path, "**", ext), recursive=True))

    if not image_files:
        print("[Error] No images found! Check the path.")
        return

    print(f"[Info] Found {len(image_files)} images. Starting analysis...")

    swt = HaarSWT().to(device)

    ratios = []
    structures = []
    noises = []

    # Analyze loop
    for img_path in tqdm(image_files):
        try:
            # Load & Preprocess
            with Image.open(img_path) as img:
                img = img.convert('L')  # 분석을 위해 Grayscale 변환
                img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]

            # SWT Analysis
            with torch.no_grad():
                out = swt(img_tensor)  # [1, 4, H, W]

                # Calculate Energy
                # out[:, 1] -> LH, out[:, 2] -> HL, out[:, 3] -> HH
                e_lh = torch.mean(torch.abs(out[:, 1]))
                e_hl = torch.mean(torch.abs(out[:, 2]))
                e_hh = torch.mean(torch.abs(out[:, 3]))

                structure = (e_lh + e_hl) / 2.0
                noise = e_hh

                # Ratio Calculation (S/N)
                ratio = structure / (noise + 1e-8)

                ratios.append(ratio.item())
                structures.append(structure.item())
                noises.append(noise.item())

        except Exception as e:
            print(f"Skipping {img_path}: {e}")
            continue

    return np.array(ratios), np.array(structures), np.array(noises), image_files


# ==========================================
# 4. Visualization & Reporting
# ==========================================
def visualize_results(ratios, structures, noises):
    plt.figure(figsize=(15, 5))

    # 1. Histogram of Ratios
    plt.subplot(1, 3, 1)
    plt.hist(ratios, bins=50, color='purple', alpha=0.7, edgecolor='black')
    plt.axvline(x=1.0, color='r', linestyle='--', label='Mixed/Noise Baseline (1.0)')
    plt.title(f'S/N Ratio Distribution\nMean: {np.mean(ratios):.4f}, Std: {np.std(ratios):.4f}')
    plt.xlabel('Structure-to-Noise Ratio')
    plt.ylabel('Count')
    plt.legend()

    # 2. Structure vs Noise Scatter
    plt.subplot(1, 3, 2)
    plt.scatter(noises, structures, alpha=0.5, s=10, c=ratios, cmap='viridis')
    plt.plot([0, max(noises)], [0, max(noises)], 'r--', label='Ratio=1.0')
    plt.title('Structure Energy vs Noise Energy')
    plt.xlabel('Noise Energy (HH)')
    plt.ylabel('Structure Energy (LH+HL)')
    plt.colorbar(label='S/N Ratio')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 3. Ratio Boxplot (For outlier detection)
    plt.subplot(1, 3, 3)
    plt.boxplot(ratios, vert=False)
    plt.title('Ratio Boxplot')
    plt.xlabel('S/N Ratio')

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 30)
    print("      ANALYSIS SUMMARY      ")
    print("=" * 30)
    print(f"Total Images: {len(ratios)}")
    print(f"Mean Ratio  : {np.mean(ratios):.4f}")
    print(f"Median Ratio: {np.median(ratios):.4f}")
    print(f"Min Ratio   : {np.min(ratios):.4f}")
    print(f"Max Ratio   : {np.max(ratios):.4f}")
    print("-" * 30)

    # Interpretation Guide
    mean_r = np.mean(ratios)
    if mean_r < 1.1:
        print(">> 해석: 데이터셋이 전반적으로 'Mixed/Noisy' 상태입니다. (구조가 많이 손상됨)")
        print(">> 전략: Denoising과 Structure Restoration이 동시에 강하게 필요합니다.")
    elif mean_r > 5.0:
        print(">> 해석: 데이터셋이 비교적 'Clean'하거나 'Simple Blur' 상태입니다.")
    else:
        print(">> 해석: 데이터셋에 다양한 열화 상태가 섞여 있거나, 중간 정도의 손상입니다.")


# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    # Check CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if os.path.exists(DATASET_PATH):
        r, s, n, _ = analyze_dataset(DATASET_PATH, device)
        if len(r) > 0:
            visualize_results(r, s, n)
    else:
        print(f"[Error] 경로를 찾을 수 없습니다: {DATASET_PATH}")
        print("스크립트 상단의 DATASET_PATH 변수를 수정해주세요.")