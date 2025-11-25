"""
Fourier Amplitude Swap Demo (Clean HR ↔ Noisy LR-upsampled)

- 입력:
    CLEAN_PATH: 오염 없는 HR 원본 이미지 경로
    DEG_PATH  : spatial size가 더 작고 Gaussian noise가 추가된 이미지 경로
- 처리:
    1) DEG 이미지를 CLEAN과 동일 크기로 bilinear upsampling
    2) 두 이미지 각각 채널별 FFT → amplitude, phase 추출
    3) amplitude swap:
        (a) A_deg + P_clean → IFFT
        (b) A_clean + P_deg → IFFT
    4) 시각화: 원본 2장, swap 결과 2장, amplitude 로그스펙트럼 2장
- 메모:
    - 채널별 독립 FFT/IFT
    - IFFT 결과를 [0,1]로 정규화하여 표시
"""

from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ---------------------- 설정 ----------------------
CLEAN_PATH = r"D:\LAB\datasets\project_use\CamVid_12_2Fold_v4\A_set\test\images\0006R0_f03360.png"   # 예: "D:/data/clean.png"
DEG_PATH   = r"D:\LAB\datasets\project_use\CamVid_12_2Fold_Gaussian_Noise\A_set\test\images\0006R0_f03360.png"   # 예: "D:/data/noisy_lr.png"
SAVE_DIR   = r"D:\LAB\result_files\analysis_results\fft_swap_results3"      # 결과 저장 폴더
# -------------------------------------------------

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

def load_rgb_float01(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr  # H x W x 3

def bilinear_resize(img_float01, size_hw):
    """img_float01: HxWxC float[0,1], size_hw: (H, W)"""
    H, W = size_hw
    pil = Image.fromarray(np.clip(img_float01 * 255.0, 0, 255).astype(np.uint8))
    pil = pil.resize((W, H), resample=Image.BILINEAR)
    return np.asarray(pil, dtype=np.float32) / 255.0

def fft2_amp_phase(img_float01):
    """
    채널별 FFT → (amplitude, phase)
    return:
      amp:  HxWxC (float)
      phs:  HxWxC (float, radians)
    """
    assert img_float01.ndim == 3 and img_float01.shape[2] in (1,3)
    H, W, C = img_float01.shape
    amp = np.zeros_like(img_float01, dtype=np.float64)
    phs = np.zeros_like(img_float01, dtype=np.float64)
    for c in range(C):
        F = np.fft.fft2(img_float01[..., c])
        amp[..., c] = np.abs(F)
        phs[..., c] = np.angle(F)
    return amp, phs

def ifft2_from_amp_phase(amp, phs):
    """
    amplitude와 phase로부터 채널별 IFFT 복원
    출력: HxWxC, 실수부를 취해 [0,1] 정규화
    """
    H, W, C = amp.shape
    out = np.zeros((H, W, C), dtype=np.float64)
    for c in range(C):
        F = amp[..., c] * np.exp(1j * phs[..., c])
        img_c = np.fft.ifft2(F)
        img_c = np.real(img_c)
        out[..., c] = img_c
    # 정규화(시각화 용이성 위해 채널 공통 min-max 정규화)
    mn, mx = out.min(), out.max()
    if mx > mn:
        out = (out - mn) / (mx - mn)
    else:
        out = np.clip(out, 0, 1)
    return out.astype(np.float32)

def log_magnitude_spectrum(amp, eps=1e-8):
    """
    시각화용 log|FFT|. 채널별 fftshift 후 평균.
    반환: HxW (단일채널)
    """
    H, W, C = amp.shape
    mag = np.zeros((H, W, C), dtype=np.float64)
    for c in range(C):
        # 원래 amp는 shift 전 기준. 시각화는 보통 중심 정렬
        # amp는 |F(u,v)|이므로 동일하게 shift 가능
        mag[..., c] = np.fft.fftshift(amp[..., c])
    mag_mean = mag.mean(axis=2)
    logmag = np.log(mag_mean + eps)
    # 보기 좋게 0~1 정규화
    mn, mx = logmag.min(), logmag.max()
    if mx > mn:
        logmag = (logmag - mn) / (mx - mn)
    else:
        logmag = np.zeros_like(logmag)
    return logmag.astype(np.float32)

def imsave(path, img_float01):
    arr = np.clip(img_float01 * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)

# ---------------------- 로드 & 전처리 ----------------------
clean = load_rgb_float01(CLEAN_PATH)          # HR clean
deg   = load_rgb_float01(DEG_PATH)            # LR+noise

# bilinear upsampling (deg → clean과 동일 크기)
H, W, _ = clean.shape
deg_up = bilinear_resize(deg, (H, W))

# ---------------------- FFT ----------------------
A_clean, P_clean = fft2_amp_phase(clean)
A_deg,   P_deg   = fft2_amp_phase(deg_up)

# amplitude swap (두 방향)
# (1) A_deg + P_clean
recon_Adeg_Pclean = ifft2_from_amp_phase(A_deg, P_clean)
# (2) A_clean + P_deg
recon_Aclean_Pdeg = ifft2_from_amp_phase(A_clean, P_deg)

# 스펙트럼 시각화 (log-magnitude)
logmag_clean = log_magnitude_spectrum(A_clean)
logmag_deg   = log_magnitude_spectrum(A_deg)

# ---------------------- 저장 ----------------------
imsave(f"{SAVE_DIR}/clean.png", clean)
imsave(f"{SAVE_DIR}/deg_upsampled.png", deg_up)
imsave(f"{SAVE_DIR}/swap_Adeg_Pclean.png", recon_Adeg_Pclean)
imsave(f"{SAVE_DIR}/swap_Aclean_Pdeg.png", recon_Aclean_Pdeg)

# 스펙트럼 저장(그레이)
Image.fromarray((logmag_clean * 255).astype(np.uint8)).save(f"{SAVE_DIR}/spectrum_clean_logmag.png")
Image.fromarray((logmag_deg   * 255).astype(np.uint8)).save(f"{SAVE_DIR}/spectrum_deg_logmag.png")

# ---------------------- 플롯 ----------------------
plt.figure(figsize=(12, 8))
plt.subplot(2,3,1); plt.imshow(clean);               plt.title("Clean (HR)");          plt.axis('off')
plt.subplot(2,3,2); plt.imshow(deg_up);             plt.title("Degraded (upsampled)");plt.axis('off')
plt.subplot(2,3,3); plt.imshow(logmag_clean, cmap='gray'); plt.title("Clean log|FFT|");plt.axis('off')
plt.subplot(2,3,4); plt.imshow(recon_Adeg_Pclean);  plt.title("Swap: A(deg)+P(clean)");plt.axis('off')
plt.subplot(2,3,5); plt.imshow(recon_Aclean_Pdeg);  plt.title("Swap: A(clean)+P(deg)");plt.axis('off')
plt.subplot(2,3,6); plt.imshow(logmag_deg, cmap='gray');   plt.title("Degraded log|FFT|");plt.axis('off')
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/panel.png", dpi=150)
plt.show()

print(f"Saved results to: {Path(SAVE_DIR).resolve()}")
