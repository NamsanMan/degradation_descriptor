import sys, os
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple

# --- 프로젝트 루트 추가 ---
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import config
from models import create_model


# ============================================================
# (1) SegFormer token형 출력을 2D로 변환
# ============================================================
def maybe_tokens_to_2d(layer_name: str, fmap: torch.Tensor, input_hw: Tuple[int, int]) -> torch.Tensor:
    if fmap.dim() == 4:
        return fmap  # [B,C,H,W]
    H, W = input_hw
    stride = None
    if "linear_c.0.proj" in layer_name:
        stride = 4
    elif "linear_c.1.proj" in layer_name:
        stride = 8
    elif "linear_c.2.proj" in layer_name:
        stride = 16
    elif "linear_c.3.proj" in layer_name:
        stride = 32

    if stride is None:
        return fmap

    h2, w2 = H // stride, W // stride
    if fmap.dim() == 3:  # [B, N, C]
        B, N, C = fmap.shape
        if N != h2 * w2:
            return fmap
        return fmap.transpose(1, 2).reshape(B, C, h2, w2)
    elif fmap.dim() == 2:  # [N, C]
        N, C = fmap.shape
        if N != h2 * w2:
            return fmap
        return fmap.transpose(0, 1).reshape(1, C, h2, w2)
    return fmap


# ============================================================
# (2) Grad-CAM 클래스
# ============================================================
class GradCAM:
    def __init__(self, model, target_layer, input_hw):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.input_hw = input_hw
        self._register_hooks()

    def _register_hooks(self):
        target_module = self.model.get_submodule(self.target_layer)

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_module.register_forward_hook(forward_hook)
        target_module.register_full_backward_hook(backward_hook)

    def __call__(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)  # segmentation logits
        if output.ndim == 4:
            output = F.interpolate(output, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        if class_idx is None:
            class_idx = torch.argmax(output)

        score = output[:, class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        grads = self.gradients
        acts = maybe_tokens_to_2d(self.target_layer, self.activations, self.input_hw)

        if grads is None or acts is None:
            raise RuntimeError(f"❌ Hook failed for {self.target_layer}")

        grads = maybe_tokens_to_2d(self.target_layer, grads, self.input_hw)

        weights = grads.mean(dim=(2, 3), keepdim=True)  # GAP over H,W
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam


# ============================================================
# (3) Heatmap overlay
# ============================================================
def overlay_heatmap_on_image(heatmap: np.ndarray, image: Image.Image, alpha: float = 0.5):
    heatmap_color = plt.get_cmap("jet")(heatmap)[:, :, :3]
    heatmap_color = (heatmap_color * 255).astype(np.uint8)

    img = np.array(image.resize((heatmap.shape[1], heatmap.shape[0]))).astype(np.float32)
    overlay = (1 - alpha) * img + alpha * heatmap_color
    overlay = np.clip(overlay / 255.0, 0, 1)
    return overlay


# ============================================================
# (4) 실행 함수
# ============================================================
def run_universal_gradcam(
    model_name="segformerb5",
    weight_path=r"E:\LAB\result_files\test_results\Aset_LR_segb5\best_model.pth",
    image_path=r"E:\LAB\datasets\project_use\CamVid_12_2Fold_v4\A_set\test\images\0016E5_08051.png",
    target_layer="model.decode_head.linear_c.2.proj",  # CNN은 decoder.aspp.0.project.0
    class_idx=10,  # 예: "car"
    save_dir=r"E:\LAB\result_files\analysis_results\gradcam_universal"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 모델 로드
    model = create_model(model_name).to(device)
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Warning] Missing keys: {missing}")
    print(f"✅ Loaded model: {model_name}")

    model.eval()

    # 입력 이미지
    input_image = Image.open(image_path).convert("RGB")
    resize_size = config.DATA.INPUT_RESOLUTION
    preprocess = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(input_image).unsqueeze(0).to(device)

    # Grad-CAM 실행
    gradcam = GradCAM(model, target_layer, input_hw=resize_size)
    heatmap = gradcam(input_tensor, class_idx)

    overlay = overlay_heatmap_on_image(heatmap, input_image)

    # 시각화
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(input_image)
    plt.title("Input Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title(f"Grad-CAM\n{model_name}\nLayer: {target_layer}\nClass: {class_idx}")
    plt.axis("off")

    out_path = Path(save_dir) / f"gradcam_{model_name}_class{class_idx}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved Grad-CAM visualization to: {out_path}")


if __name__ == "__main__":
    run_universal_gradcam()
