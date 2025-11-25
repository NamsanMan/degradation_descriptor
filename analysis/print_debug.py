import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from torchvision import transforms

# --- 1. 두 개의 서로 다른 사전 학습된 모델 로드 ---
# ResNet18 모델
model1_name = "ResNet18"
model1 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model1.eval()

# MobileNetV2 모델
model2_name = "MobileNetV2"
model2 = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model2.eval()

# --- 2. 동일한 입력 이미지 준비 ---
# Pycharm 환경에서 실행 시, 스크립트와 같은 위치에 'cat.jpg' 같은 이미지 파일을 준비해주세요.
# 예시 URL: https://placekitten.com/224/224
try:
    input_image = Image.open(r"E:\LAB\datasets\project_use\CamVid_12_2Fold_LR_x4_Bilinear\A_set\test\images\0001TP_006720.png")
except FileNotFoundError:
    print("Error: 'cat.jpg' 파일을 찾을 수 없습니다. 스크립트와 같은 폴더에 이미지 파일을 준비해주세요.")
    # 임시로 더미 텐서 생성
    input_image = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype('uint8'))


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # 배치 차원 추가: [C, H, W] -> [B, C, H, W]

# GPU 사용 가능 시
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model1.to('cuda')
    model2.to('cuda')
    print("CUDA를 사용하여 추론합니다.")

# --- 3. 각 모델로 추론하여 Logit 얻기 ---
with torch.no_grad():
    logits1 = model1(input_batch)
    logits2 = model2(input_batch)

# --- 4. Logit 값과 분포 분석 ---
# CPU로 이동하여 numpy 배열로 변환
logits1_np = logits1.squeeze().cpu().numpy()
logits2_np = logits2.squeeze().cpu().numpy()

print(f"\n--- {model1_name} Logit 분석 ---")
print(f"Logits Shape: {logits1_np.shape}")
print(f"Min: {logits1_np.min():.4f}, Max: {logits1_np.max():.4f}, Mean: {logits1_np.mean():.4f}")
pred1_idx = np.argmax(logits1_np)
print(f"가장 높은 Logit 값의 인덱스: {pred1_idx} (값: {logits1_np[pred1_idx]:.4f})")


print(f"\n--- {model2_name} Logit 분석 ---")
print(f"Logits Shape: {logits2_np.shape}")
print(f"Min: {logits2_np.min():.4f}, Max: {logits2_np.max():.4f}, Mean: {logits2_np.mean():.4f}")
pred2_idx = np.argmax(logits2_np)
print(f"가장 높은 Logit 값의 인덱스: {pred2_idx} (값: {logits2_np[pred2_idx]:.4f})")

# --- 5. 분포 시각화 ---
plt.figure(figsize=(14, 6))
plt.suptitle("Logit 값 분포 비교 (동일 이미지 입력)", fontsize=16)

plt.subplot(1, 2, 1)
sns.histplot(logits1_np, bins=50, kde=True)
plt.title(f"{model1_name} Logit Distribution")
plt.xlabel("Logit Value")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
sns.histplot(logits2_np, bins=50, kde=True)
plt.title(f"{model2_name} Logit Distribution")
plt.xlabel("Logit Value")
plt.ylabel("Frequency")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
