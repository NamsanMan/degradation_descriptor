import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

# --- 1. 이미지 불러오기 ---
# 'your_image.jpg' 부분에 실제 이미지 파일 경로를 입력하세요.
try:
    img = cv2.imread(r'E:\LAB\datasets\project_use\CamVid_12_2Fold_LR_x4_Bilinear\A_set\test\images\0016E5_08051.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
except Exception as e:
    print(f"이미지를 불러오는 데 실패했습니다: {e}")
    print("샘플용 검은색 이미지를 생성합니다.")
    img_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    gray_img = np.zeros((256, 256), dtype=np.uint8)


# --- 2. 웨이블릿 변환 수행 ---
wavelet_type = 'haar'
coeffs = pywt.dwt2(gray_img, wavelet_type)
cA, (cH, cV, cD) = coeffs

# --- 3. 결과 시각화 ---

# 3-1. 4개의 하위 대역을 하나의 이미지로 합치기 (이전과 동일)
top_row = np.hstack((cA, cH))
bottom_row = np.hstack((cV, cD))
wavelet_transformed_img = np.vstack((top_row, bottom_row))

# 3-2. LH, HL, HH 상세 계수(details) 합치기 (새로 추가된 부분)
# 각 방향의 경계선 정보를 모두 더해서 전체 윤곽을 확인합니다.
details_combined = cH + cV + cD


# 3-3. Matplotlib를 사용하여 3개의 이미지를 나란히 출력
plt.figure(figsize=(18, 6))

# 원본 이미지 출력
plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

# 전체 웨이블릿 변환 결과 출력 (4개 사분면)
plt.subplot(1, 3, 2)
plt.imshow(wavelet_transformed_img, cmap='gray')
plt.title(f'Wavelet Transform ({wavelet_type})')
plt.axis('off')

# LH+HL+HH 합친 결과 출력
plt.subplot(1, 3, 3)
plt.imshow(details_combined, cmap='gray')
plt.title('Combined Details (LH+HL+HH)')
plt.axis('off')

plt.tight_layout()
plt.show()