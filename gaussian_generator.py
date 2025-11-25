import os
import cv2
import numpy as np
import random
from pathlib import Path


def add_gaussian_noise(img, sigma, gray_noise=False):
    """
    img: uint8, BGR
    sigma: noise std ∈ [1,30]
    gray_noise: True → grayscale noise broadcast
    """
    h, w, c = img.shape

    if gray_noise:
        noise = np.random.normal(0, sigma, (h, w, 1)).astype(np.float32)
        noise = np.repeat(noise, 3, axis=2)
    else:
        noise = np.random.normal(0, sigma, (h, w, 3)).astype(np.float32)

    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def process_folder(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_exts = [".png", ".jpg", ".jpeg", ".bmp"]

    for fname in os.listdir(input_dir):
        if not any(fname.lower().endswith(ext) for ext in valid_exts):
            continue

        img_path = input_dir / fname
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Failed to load {img_path}")
            continue

        # Gaussian noise parameters
        sigma = random.uniform(1, 30)
        gray_noise = (random.random() < 0.40)   # Gray 40%, Color 60%

        noisy = add_gaussian_noise(img, sigma, gray_noise)

        # Save with identical filename
        out_path = output_dir / fname
        cv2.imwrite(str(out_path), noisy)

        print(f"[OK] Saved: {fname} | sigma={sigma:.2f}, gray={gray_noise}")


if __name__ == "__main__":
    input_folder  = r"D:\LAB\datasets\project_use\CamVid_12_2Fold_v4\B_set\test\images"
    output_folder = r"D:\LAB\datasets\project_use\CamVid_12_2Fold_Gaussian_Noise\B_set\test\images"

    process_folder(input_folder, output_folder)
