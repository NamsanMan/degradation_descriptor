# python analysis/swt_compare.py
# python analysis/swt_compare.py --hr D:\LAB\datasets\project_use\CamVid_12_2Fold_v4\all\images\0001TP_007290.png --lr D:\LAB\datasets\project_use\CamVid_12_2Fold_LR_x4_Bilinear\all\images\0001TP_007290.png --wavelet haar --save_fig swt_compare.png

import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import pywt

# SSIM 없이도 돌아가게 하기 위해 optional import
try:
    from skimage.metrics import structural_similarity as ssim
except Exception:
    ssim = None


def load_grayscale(path: str) -> np.ndarray:
    """Load image as float32 grayscale in [0,1]."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    # luminance (ITU-R BT.601)
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return gray.astype(np.float32)


def bilinear_resize(gray: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """Resize grayscale image (H,W) to target size (H,W) with bilinear."""
    pil = Image.fromarray(np.clip(gray * 255.0, 0, 255).astype(np.uint8), mode="L")
    pil = pil.resize((size_hw[1], size_hw[0]), resample=Image.BILINEAR)
    out = np.asarray(pil).astype(np.float32) / 255.0
    return out


def swt_level1_bands(gray: np.ndarray, wavelet: str = "haar"):
    """
    SWT2 level=1 -> returns LL, (LH, HL, HH)
    Note: pywt.swt2 returns list of levels. Each level: (cA, (cH, cV, cD))
          where (cH, cV, cD) correspond to (horizontal, vertical, diagonal) details.
    We'll map:
      LL = cA
      HL = cH (horizontal detail)
      LH = cV (vertical detail)
      HH = cD (diagonal detail)
    """
    coeffs = pywt.swt2(gray, wavelet=wavelet, level=1, start_level=0, axes=(0, 1))
    cA, (cH, cV, cD) = coeffs[0]
    LL = cA
    HL = cH
    LH = cV
    HH = cD
    return LL, HL, LH, HH


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.mean((a - b) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    m = mse(a, b)
    if m == 0:
        return float("inf")
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(m))


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def ssim_score(a: np.ndarray, b: np.ndarray) -> float | None:
    if ssim is None:
        return None
    # SSIM expects 2D, same size
    return float(ssim(a, b, data_range=1.0))


def norm_for_vis(x: np.ndarray, robust: bool = True) -> np.ndarray:
    """
    Normalize band for visualization.
    For detail bands (can be negative), use symmetric robust scaling.
    """
    x = x.astype(np.float32)
    if robust:
        lo = np.percentile(x, 1)
        hi = np.percentile(x, 99)
        if hi - lo < 1e-6:
            return np.zeros_like(x)
        x = np.clip(x, lo, hi)
    # center to 0..1 for imshow
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-6:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def compare_and_report(hr_path: str, lr_path: str, wavelet: str = "haar", save_fig: str | None = None):
    # 1) load
    hr = load_grayscale(hr_path)  # expected 360x480
    lr = load_grayscale(lr_path)  # expected 90x120

    H, W = hr.shape
    # 2) bilinear upsample LR -> HR size
    lr_up = bilinear_resize(lr, (H, W))

    # 3) SWT bands
    hr_LL, hr_HL, hr_LH, hr_HH = swt_level1_bands(hr, wavelet=wavelet)
    lr_LL, lr_HL, lr_LH, lr_HH = swt_level1_bands(lr_up, wavelet=wavelet)

    bands = {
        "LL": (hr_LL, lr_LL),
        "HL": (hr_HL, lr_HL),
        "LH": (hr_LH, lr_LH),
        "HH": (hr_HH, lr_HH),
    }

    # 4) metrics
    print(f"[Info] HR shape: {hr.shape}, LR shape: {lr.shape}, LR_up shape: {lr_up.shape}")
    print(f"[Info] Wavelet: {wavelet}, SWT level: 1")
    print("")
    print("Band |   MSE        PSNR(dB)    SSIM        Pearson r")
    print("-----+------------------------------------------------------")
    for k, (a, b) in bands.items():
        m = mse(a, b)
        p = psnr(a, b, data_range=1.0)  # note: SWT outputs are not strictly [0,1]; PSNR is still comparable across bands if consistent
        s = ssim_score(a, b)
        r = pearson_r(a, b)
        s_str = f"{s: .6f}" if s is not None else "  (skimage 없음)"
        p_str = f"{p: .3f}" if np.isfinite(p) else "   inf"
        print(f"{k:>3}  | {m: .6e}   {p_str:>9}   {s_str:>12}   {r: .6f}")

    # 5) visualization (HR vs LR_up) per band
    fig, axes = plt.subplots(4, 3, figsize=(12, 14))
    fig.suptitle("SWT Band Comparison (HR vs Bilinear Upsampled LR)", fontsize=14)

    for i, k in enumerate(["LL", "HL", "LH", "HH"]):
        a, b = bands[k]
        diff = a - b

        axes[i, 0].imshow(norm_for_vis(a), cmap="gray")
        axes[i, 0].set_title(f"{k} (HR)")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(norm_for_vis(b), cmap="gray")
        axes[i, 1].set_title(f"{k} (LR upsampled)")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(norm_for_vis(diff), cmap="gray")
        axes[i, 2].set_title(f"{k} (HR - LR_up)")
        axes[i, 2].axis("off")

    plt.tight_layout()
    if save_fig:
        plt.savefig(save_fig, dpi=200, bbox_inches="tight")
        print(f"\n[Saved] Figure -> {save_fig}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hr", type=str, required=True, help="Path to HR image (360x480)")
    parser.add_argument("--lr", type=str, required=True, help="Path to LR image (90x120, paired with HR)")
    parser.add_argument("--wavelet", type=str, default="haar", help="Wavelet name for SWT (e.g., 'haar', 'db2')")
    parser.add_argument("--save_fig", type=str, default=None, help="Optional path to save the figure (e.g., out.png)")
    args = parser.parse_args()

    compare_and_report(args.hr, args.lr, wavelet=args.wavelet, save_fig=args.save_fig)


if __name__ == "__main__":
    main()
