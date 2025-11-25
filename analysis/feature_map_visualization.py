"""Visualise feature maps for models defined in this project.

Run this script to inspect intermediate representations from any model that
can be created via :func:`models.create_model`. Example usage::

가능한 layer 확인:
python analysis\feature_map_visualization.py --model-name d3p --list-layers --image E:\LAB\datasets\project_use\CamVid_12_2Fold_v4\A_set\test\images\0016E5_08051.png

실행 예시:
python analysis\feature_map_visualization.py --model-name segformerb5 --weight E:\LAB\result_files\test_results\Aset_LR_segb5\best_model.pth --image E:\LAB\datasets\project_use\CamVid_12_2Fold_v4\A_set\test\images\0016E5_08051.png --target-layers model.decode_head.linear_c.0.proj --num-channels 16 --save-dir E:\LAB\result_files\analysis_results\feature_map_visualization


The script supports listing available layers, saving visualisations to disk
and working in headless environments.
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.hooks import RemovableHandle

# Ensure the project root is on ``sys.path`` so the script can be executed
# from the repository root without additional PYTHONPATH configuration.
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import config
from models import create_model  # noqa: E402  # pylint: disable=wrong-import-position


class FeatureExtractor:
    """Use forward hooks to capture feature maps from selected layers."""

    def __init__(self, model: torch.nn.Module, target_layers: Iterable[str]):
        self.model = model
        self.target_layers = list(target_layers)
        self.features: Dict[str, torch.Tensor] = {}
        self._hooks: list[RemovableHandle] = []
        self._register_hooks()

    def _make_hook(self, name: str):
        def hook(module, inputs, output):  # pylint: disable=unused-argument
            self.features[name] = output

        return hook

    def _register_hooks(self) -> None:
        for name in self.target_layers:
            layer = self.model.get_submodule(name)
            handle = layer.register_forward_hook(self._make_hook(name))
            self._hooks.append(handle)

    def remove_hooks(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

def _maybe_tokens_to_2d(layer_name: str, fmap: torch.Tensor, input_hw: Tuple[int, int]) -> torch.Tensor:
    """
    SegFormer decode_head.linear_c.*.proj 같이 [B, N_tokens, C] 또는 [N_tokens, C] 텐서를
    [B, C, H', W']로 변환. H',W'는 입력 해상도/stride로 유추.
    일치하지 않으면 원본 그대로 반환.
    """
    if fmap.dim() == 4:
        return fmap  # [B,C,H,W] 이미 2D

    H, W = input_hw  # e.g., config.DATA.INPUT_RESOLUTION
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
        return fmap  # 모르는 레이어면 변환하지 않음

    h2, w2 = H // stride, W // stride

    if fmap.dim() == 3:  # [B, N, C]
        B, N, C = fmap.shape
        if N != h2 * w2:
            print(f"[Warn] token count mismatch for {layer_name}: N={N}, expected={h2*w2}. Skip reshape.")
            return fmap
        return fmap.transpose(1, 2).reshape(B, C, h2, w2)

    if fmap.dim() == 2:  # [N, C]
        N, C = fmap.shape
        if N != h2 * w2:
            print(f"[Warn] token count mismatch for {layer_name}: N={N}, expected={h2*w2}. Skip reshape.")
            return fmap
        return fmap.transpose(0, 1).reshape(1, C, h2, w2)

    return fmap


def visualize_feature_maps(
    feature_maps: Dict[str, torch.Tensor],
    num_channels_to_show: int = 16,
    save_dir: Optional[Path] = None,
    show_plots: bool = True,
) -> None:
    """Render feature maps as a grid, optionally saving them to disk."""

    for layer_name, feature_map in feature_maps.items():
        feature_map = feature_map.squeeze(0).detach().cpu()

        num_channels = feature_map.shape[0]
        channels_to_plot = min(num_channels, num_channels_to_show)
        cols = int(np.ceil(np.sqrt(channels_to_plot)))
        rows = int(np.ceil(channels_to_plot / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        fig.suptitle(
            f"Feature Maps from: {layer_name} (first {channels_to_plot} channels)",
            fontsize=14,
        )

        axes = np.atleast_1d(axes).flatten()

        for idx in range(channels_to_plot):
            ax = axes[idx]
            ax.imshow(feature_map[idx], cmap="viridis")
            ax.axis("off")
            ax.set_title(f"Channel {idx}", fontsize=8)

        for idx in range(channels_to_plot, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            sanitized = layer_name.replace(".", "_")
            out_path = save_dir / f"feature_maps_{sanitized}.png"
            fig.savefig(out_path, bbox_inches="tight")
            print(f"Saved feature maps to {out_path}")

        if show_plots:
            plt.show()

        plt.close(fig)


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    return state_dict


def load_model(model_name: str, weight_path: Optional[Path], device: torch.device) -> torch.nn.Module:
    model = create_model(model_name)
    model.to(device)

    if weight_path is not None:
        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict):
            if "model_state" in checkpoint:
                state_dict = checkpoint["model_state"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            raise RuntimeError(f"Unsupported checkpoint format: {type(checkpoint)}")

        state_dict = _strip_module_prefix(state_dict)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[Warning] Missing keys when loading weights: {missing}")
        if unexpected:
            print(f"[Warning] Unexpected keys in checkpoint: {unexpected}")
        print(f"Loaded weights from {weight_path}")

    model.eval()
    return model


def list_available_layers(model: torch.nn.Module) -> Iterable[str]:
    for name, _ in model.named_modules():
        if name:
            yield name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise intermediate feature maps.")
    parser.add_argument(
        "--model-name",
        default=config.MODEL.NAME,
        help="Model name registered in models.create_model (default: config.MODEL.NAME)",
    )
    parser.add_argument(
        "--weight",
        type=Path,
        default=None,
        help="Path to a checkpoint (.pth). If omitted, randomly initialised weights are used.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to an RGB image used as input for the forward pass.",
    )
    parser.add_argument(
        "--target-layers",
        nargs="+",
        default=["conv1", "layer1.0.conv1", "layer2.0.conv1", "layer4"],
        help="Layer names (from model.named_modules()) to hook for feature extraction.",
    )
    parser.add_argument(
        "--num-channels",
        type=int,
        default=16,
        help="Maximum number of channels per layer to display (default: 16).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="If provided, save the generated grids into this directory.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable interactive plot display (useful for headless environments).",
    )
    parser.add_argument(
        "--list-layers",
        action="store_true",
        help="List available layer names for the selected model and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.model_name, args.weight, device)

    if args.list_layers:
        print("Available layers:")
        for name in list_available_layers(model):
            print(f"  - {name}")
        return

    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    input_image = Image.open(args.image).convert("RGB")
    resize_size = config.DATA.INPUT_RESOLUTION
    preprocess = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image).unsqueeze(0).to(device)

    extractor = FeatureExtractor(model, args.target_layers)

    with torch.no_grad():
        _ = model(input_tensor)

    captured_features = extractor.features
    extractor.remove_hooks()

    resize_h, resize_w = int(resize_size[0]), int(resize_size[1])  # (H,W)
    converted = {}
    for lname, fmap in captured_features.items():
        converted[lname] = _maybe_tokens_to_2d(lname, fmap, (resize_h, resize_w))

    visualize_feature_maps(
        converted,
        num_channels_to_show=args.num_channels,
        save_dir=args.save_dir,
        show_plots=not args.no_show,
    )

    print("\n시각화 완료!")
    print("추출된 레이어 목록:", list(captured_features.keys()))


if __name__ == "__main__":
    main()
