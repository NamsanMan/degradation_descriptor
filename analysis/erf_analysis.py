"""
Effective Receptive Field (ERF) analysis.

- 지정한 target layer의 feature map 중앙 (h//2, w//2) 위치의 1개 뉴런을 선택.
- 해당 뉴런 값에 대한 입력 이미지(전처리된 텐서) 픽셀의 gradient를 autograd로 계산.
- gradient의 절댓값(채널 축 L2 norm)을 시각화하여 '유효 수용 영역'을 확인.

사용 예:
python analysis/erf_analysis.py ^
  --model-name d3p ^
  --weight checkpoints\\d3p_camvid_best.pth ^
  --image E:\\LAB\\datasets\\project_use\\CamVid_12_2Fold_v4\\A_set\\test\\images\\0001TP_008280.png ^
  --target-layer encoder.features.13 ^
  --save-dir results\\erf_d3p

Transformer 예:
python analysis/erf_analysis.py ^
  --model-name segformerb0 ^
  --image samples\\sample_camvid.png ^
  --target-layer model.segformer.encoder.blocks.2  (모델에 맞는 레이어명을 --list-layers로 먼저 확인)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

# 프로젝트 루트를 import 우선순위 최상단에 추가
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from test import config  # noqa: E402
from models import create_model  # noqa: E402


def list_available_layers(model: torch.nn.Module):
    for name, _ in model.named_modules():
        if name:
            yield name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Effective Receptive Field (ERF) analysis")
    p.add_argument("--model-name", default=config.MODEL.NAME)
    p.add_argument("--weight", type=Path, default=None)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--target-layer", type=str, required=False,
                   help="named_modules() 중 hook을 걸 레이어 이름. (--list-layers로 확인)")
    p.add_argument("--channel-index", type=int, default=None,
                   help="선택 뉴런의 채널 인덱스. None이면 중앙 위치에서 절댓값이 가장 큰 채널을 자동 선택")
    p.add_argument("--save-dir", type=Path, default=None)
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--list-layers", action="store_true")
    p.add_argument("--gamma", type=float, default=0.5,
                   help="시각화 대비 향상용 감마(0.4~0.7 권장). 값이 작을수록 고응답 강조")
    p.add_argument("--overlay-alpha", type=float, default=0.45,
                   help="원본 위 heatmap overlay 투명도")
    return p.parse_args()


class FeatureTap:
    """Forward hook로 target layer 출력을 저장."""
    def __init__(self, model: torch.nn.Module, layer_name: str):
        self.model = model
        self.layer_name = layer_name
        self.tensor: Optional[torch.Tensor] = None
        self.handle = None

    def _hook(self, module, inputs, output):
        self.tensor = output

    def __enter__(self):
        layer = self.model.get_submodule(self.layer_name)
        self.handle = layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if state_dict and all(k.startswith("module.") for k in state_dict):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def load_model(model_name: str, weight_path: Optional[Path], device: torch.device) -> torch.nn.Module:
    model = create_model(model_name)
    model.to(device)
    if weight_path is not None:
        ckpt = torch.load(weight_path, map_location=device)
        if isinstance(ckpt, dict):
            state_dict = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
        else:
            raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt)}")
        state_dict = _strip_module_prefix(state_dict)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[Warn] missing keys: {missing}")
        if unexpected:
            print(f"[Warn] unexpected keys: {unexpected}")
        print(f"Loaded weights from {weight_path}")
    model.eval()
    return model


def make_preprocess(resize_hw):
    return transforms.Compose([
        transforms.Resize(resize_hw, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def visualize_erf(gradmap_hw: np.ndarray,
                  rgb_resized: np.ndarray,
                  gamma: float = 0.5,
                  overlay_alpha: float = 0.45,
                  save_dir: Optional[Path] = None,
                  tag: str = "erf"):

    # grad 정규화 및 감마 보정
    g = gradmap_hw.astype(np.float32)
    g -= g.min()
    if g.max() > 0:
        g /= g.max()
    # 감마 보정(대비 향상)
    g = np.power(g, gamma)

    # heatmap만
    plt.figure(figsize=(6, 5))
    plt.title(f"ERF heatmap ({tag})")
    plt.imshow(g, cmap="jet")
    plt.axis("off")
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"{tag}_heatmap.png", bbox_inches="tight", pad_inches=0.05)
    if not save_dir or not overlay_alpha:  # 저장만 하고 닫지 않으려면 조건 조정 가능
        pass
    if not save_dir is None:
        plt.close()

    # 원본 overlay
    plt.figure(figsize=(6, 5))
    plt.title(f"Overlay ({tag})")
    plt.imshow(rgb_resized)                # [H,W,3], 0~1
    plt.imshow(g, cmap="jet", alpha=overlay_alpha)
    plt.axis("off")
    if save_dir:
        plt.savefig(save_dir / f"{tag}_overlay.png", bbox_inches="tight", pad_inches=0.05)
    plt.show()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.model_name, args.weight, device)

    if args.list_layers:
        print("Available layers:")
        for name in list_available_layers(model):
            print("  -", name)
        return

    if args.target-layer is None:
        raise SystemExit("ERROR: --target-layer 를 지정하세요. (--list-layers 로 이름 확인)")

    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    # 입력/전처리
    resize_hw = config.DATA.INPUT_RESOLUTION  # e.g., (360, 480)
    preprocess = make_preprocess(resize_hw)
    pil = Image.open(args.image).convert("RGB")
    rgb_resized = np.asarray(pil.resize((resize_hw[1], resize_hw[0]), Image.BILINEAR), dtype=np.float32) / 255.0

    x = preprocess(pil).unsqueeze(0).to(device)
    x.requires_grad_(True)

    # forward + hook capture
    with FeatureTap(model, args.target_layer) as tap:
        with torch.no_grad():
            _ = model(x)  # 1차 forward로 feature shape 탐색
        fmap = tap.tensor
        if fmap is None:
            raise RuntimeError(f"Hooked layer produced no tensor: {args.target_layer}")

    # 중앙 좌표/채널 선택
    _, C, H, W = fmap.shape
    cy, cx = H // 2, W // 2
    fmap_detached = fmap.detach()
    if args.channel_index is None:
        # 중앙 위치에서 절댓값이 가장 큰 채널 선택
        center_vals = fmap_detached[0, :, cy, cx].abs()
        c_idx = int(torch.argmax(center_vals).item())
        print(f"[Auto] channel-index = {c_idx} (max |activation| at center)")
    else:
        c_idx = int(args.channel_index)
        if not (0 <= c_idx < C):
            raise SystemExit(f"Invalid --channel-index {c_idx}, valid range: [0, {C-1}]")
        print(f"[User] channel-index = {c_idx}")

    # 선택 스칼라에 대한 grad 계산을 위해 gradient graph 재생성
    model.zero_grad(set_to_none=True)
    x.grad = None
    with FeatureTap(model, args.target_layer) as tap2:
        y = model(x)  # forward with grad
        fmap2 = tap2.tensor
        if fmap2 is None:
            raise RuntimeError("Hook failed on second forward pass")

        scalar = fmap2[0, c_idx, cy, cx]
        # backprop: d(scalar)/d(input)
        scalar.backward(retain_graph=False)

    # 입력 텐서에 대한 gradient: [1, 3, H, W]
    grad_input = x.grad.detach()[0]        # [3,H,W], 전처리된 입력 공간
    # 채널 축 L2-norm으로 단일 채널 맵 생성
    grad_norm = torch.linalg.vector_norm(grad_input, ord=2, dim=0)  # [H,W]
    grad_np = grad_norm.cpu().numpy()

    tag = f"{args.model_name.replace('/', '_')}_{args.target_layer.replace('.', '_')}_c{c_idx}"
    visualize_erf(
        gradmap_hw=grad_np,
        rgb_resized=rgb_resized,
        gamma=args.gamma,
        overlay_alpha=args.overlay_alpha,
        save_dir=args.save_dir,
        tag=tag,
    )

    print("\nERF 분석 완료.")
    print(f"  - target layer : {args.target_layer}")
    print(f"  - channel index: {c_idx}")
    if args.save_dir:
        print(f"  - saved to     : {args.save_dir}")
    else:
        print("  - (no save-dir; only shown on screen)")


if __name__ == "__main__":
    main()
