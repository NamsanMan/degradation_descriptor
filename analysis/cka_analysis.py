"""
Centered Kernel Alignment (CKA) similarity analysis between two models.

- 동일한 이미지(들)를 두 모델에 통과시켜 각 레이어의 표현을 수집
- 표현을 (GAP 등으로) [N, C] 형태로 정규화
- 모든 (i, j) 레이어 쌍에 대해 Linear CKA 계산
- 히트맵으로 저장

예시:
python analysis/cka_analysis.py ^
  --model-a-name d3p --weight-a checkpoints\\d3p_camvid_best.pth ^
  --model-b-name segformerb0 --weight-b none ^
  --images-dir E:\\LAB\\datasets\\project_use\\CamVid_12_2Fold_v4\\A_set\\test\\images ^
  --layers-a encoder.features.3 encoder.features.6 encoder.features.13 encoder.features.17 decoder.aspp decoder.block1 decoder.block2 segmentation_head ^
  --layers-b model.segformer.encoder.block1 model.segformer.encoder.block2 model.segformer.encoder.block3 model.segformer.encoder.block4 decode_head ^
  --batch-size 4 ^
  --save-dir results\\cka_d3p_vs_b0

레이어 이름은 각각 --list-layers-a / --list-layers-b 로 먼저 확인하세요.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# 프로젝트 루트를 import 우선순위 최상단에 추가
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from test import config  # noqa: E402
from models import create_model  # noqa: E402


# -------------------------
# Utilities
# -------------------------

def list_named_modules(model: torch.nn.Module) -> List[str]:
    names = []
    for name, _ in model.named_modules():
        if name:
            names.append(name)
    return names


def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if sd and all(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def load_model(name: str, weight: Optional[Path], device: torch.device) -> torch.nn.Module:
    model = create_model(name)
    model.to(device)
    if weight is not None and str(weight).lower() != "none":
        ckpt = torch.load(weight, map_location=device)
        if isinstance(ckpt, dict):
            sd = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
        else:
            raise RuntimeError(f"Unsupported checkpoint type: {type(ckpt)}")
        sd = _strip_module_prefix(sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[Warn] missing keys in {name}: {missing}")
        if unexpected:
            print(f"[Warn] unexpected keys in {name}: {unexpected}")
        print(f"Loaded weights for {name} from: {weight}")
    model.eval()
    return model


class ImageFolderNoLabel(Dataset):
    def __init__(self, img_dir: Path, resize_hw: Tuple[int, int]):
        self.paths = sorted(
            [p for p in Path(img_dir).glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
        )
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No images found in: {img_dir}")
        self.resize_hw = resize_hw
        self.tf = T.Compose([
            T.Resize(resize_hw, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        pil = Image.open(self.paths[idx]).convert("RGB")
        x = self.tf(pil)
        return x


def make_single_image_loader(image_path: Path, resize_hw: Tuple[int, int]) -> DataLoader:
    pil = Image.open(image_path).convert("RGB")
    tf = T.Compose([
        T.Resize(resize_hw, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = tf(pil).unsqueeze(0)
    return DataLoader([x], batch_size=1, shuffle=False)


class LayerHook:
    """Register forward hooks on selected layer names and collect pooled features."""
    def __init__(self, model: torch.nn.Module, layer_names: List[str], pool: str = "gap"):
        self.model = model
        self.layer_names = layer_names
        self.pool = pool
        self.buffers: Dict[str, List[torch.Tensor]] = {n: [] for n in layer_names}
        self.handles = []

    def _hook_fn(self, name: str):
        def fn(module, inputs, output):
            # Expect output as Tensor [B, C, H, W] or [B, C]
            if isinstance(output, (list, tuple)):
                # try to pick the first tensor if module returns tuple
                out = None
                for o in output:
                    if isinstance(o, torch.Tensor):
                        out = o
                        break
                if out is None:
                    return
                output_t = out
            else:
                if not isinstance(output, torch.Tensor):
                    return
                output_t = output

            if output_t.dim() == 4:
                B, C, H, W = output_t.shape
                if self.pool == "gap":
                    feat = torch.mean(output_t, dim=(2, 3))  # [B, C]
                elif self.pool == "flatten_spatial":
                    feat = output_t.permute(0, 2, 3, 1).reshape(B * H * W, C)  # samples = B*H*W
                else:
                    # default to GAP
                    feat = torch.mean(output_t, dim=(2, 3))
            elif output_t.dim() == 2:
                feat = output_t  # [B, C]
            else:
                return
            # detach on CPU to save memory; we will cat later
            self.buffers[name].append(feat.detach().cpu())
        return fn

    def __enter__(self):
        for n in self.layer_names:
            layer = self.model.get_submodule(n)
            self.handles.append(layer.register_forward_hook(self._hook_fn(n)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles = []

    def stacked(self) -> Dict[str, torch.Tensor]:
        # Concatenate along first dim (samples)
        out = {}
        for n, chunks in self.buffers.items():
            if len(chunks) == 0:
                continue
            out[n] = torch.cat(chunks, dim=0)  # [N, C]
        return out


# -------------------------
# CKA (Linear) implementation
# -------------------------

def _center_rows(X: torch.Tensor) -> torch.Tensor:
    # X: [N, C]
    X = X - X.mean(dim=0, keepdim=True)
    return X


def linear_CKA(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Linear CKA between X and Y.
    X: [N, Cx], Y: [N, Cy]
    """
    Xc = _center_rows(X)
    Yc = _center_rows(Y)
    # Gram with linear kernel is simply Xc Xc^T, but we can avoid building [N,N] by using
    # ||Yc^T Xc||_F^2 / (||Xc^T Xc||_F * ||Yc^T Yc||_F)
    # Compute Frob of cross-covariance:
    XtY = Yc.T @ Xc  # [Cy, Cx]
    numerator = torch.norm(XtY, p="fro") ** 2
    denom = torch.norm(Xc.T @ Xc, p="fro") * torch.norm(Yc.T @ Yc, p="fro")
    if denom == 0:
        return float("nan")
    return (numerator / denom).item()


def compute_cka_matrix(reps_a: Dict[str, torch.Tensor],
                       reps_b: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, List[str], List[str]]:
    names_a = list(reps_a.keys())
    names_b = list(reps_b.keys())
    M = np.zeros((len(names_a), len(names_b)), dtype=np.float32)
    for i, na in enumerate(names_a):
        Xa = reps_a[na]  # [N, Ca]
        for j, nb in enumerate(names_b):
            Yb = reps_b[nb]  # [N, Cb]
            # Ensure same N
            N = min(Xa.shape[0], Yb.shape[0])
            val = linear_CKA(Xa[:N], Yb[:N])
            M[i, j] = np.nan_to_num(val, nan=0.0)
    return M, names_a, names_b


def plot_heatmap(M: np.ndarray, rows: List[str], cols: List[str], save_dir: Optional[Path]):
    plt.figure(figsize=(max(6, 0.6 * len(cols)), max(4, 0.6 * len(rows))))
    plt.imshow(M, aspect="auto", interpolation="nearest", cmap="viridis")
    plt.colorbar(label="Linear CKA")
    plt.xticks(range(len(cols)), [c.replace("model.", "").replace("encoder.", "enc.").replace("decoder.", "dec.") for c in cols], rotation=45, ha="right")
    plt.yticks(range(len(rows)), [r.replace("model.", "").replace("encoder.", "enc.").replace("decoder.", "dec.") for r in rows])
    plt.tight_layout()
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / "cka_heatmap.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved heatmap: {out}")
    plt.show()
    plt.close()


# -------------------------
# Main
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CKA similarity between two models")
    p.add_argument("--model-a-name", default=config.MODEL.NAME)
    p.add_argument("--weight-a", type=Path, default=None)
    p.add_argument("--model-b-name", required=True)
    p.add_argument("--weight-b", type=Path, default=None)

    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--images-dir", type=Path, help="이미지 폴더 (모든 *.png/jpg 등)")
    grp.add_argument("--image", type=Path, help="단일 이미지 경로")

    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--layers-a", nargs="+", required=False, default=None, help="model A의 hook 레이어 목록")
    p.add_argument("--layers-b", nargs="+", required=False, default=None, help="model B의 hook 레이어 목록")
    p.add_argument("--pool", type=str, default="gap", choices=["gap", "flatten_spatial"], help="피처 pooling 방식")
    p.add_argument("--save-dir", type=Path, default=None)
    p.add_argument("--list-layers-a", action="store_true", help="모델 A 레이어 나열 후 종료")
    p.add_argument("--list-layers-b", action="store_true", help="모델 B 레이어 나열 후 종료")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    model_a = load_model(args.model_a_name, args.weight_a, device)
    model_b = load_model(args.model_b_name, args.weight_b, device)

    if args.list_layers_a:
        print("[Model A] Available layers:")
        for n in list_named_modules(model_a):
            print("  -", n)
    if args.list_layers_b:
        print("[Model B] Available layers:")
        for n in list_named_modules(model_b):
            print("  -", n)
    if args.list_layers_a or args.list_layers_b:
        return

    if args.layers_a is None or args.layers_b is None:
        raise SystemExit("ERROR: --layers-a 와 --layers-b 를 지정하세요. (--list-layers-*)로 이름 확인)")

    # Data loader
    resize_hw = tuple(config.DATA.INPUT_RESOLUTION)  # (H, W)
    if args.images_dir:
        ds = ImageFolderNoLabel(args.images_dir, resize_hw)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    else:
        loader = make_single_image_loader(args.image, resize_hw)

    # Collect features
    with torch.no_grad():  # forward 중간 텐서만 후킹/저장
        reps_a: Dict[str, List[torch.Tensor]] = {}
        reps_b: Dict[str, List[torch.Tensor]] = {}

    with LayerHook(model_a, args.layers_a, pool=args.pool) as hook_a, \
         LayerHook(model_b, args.layers_b, pool=args.pool) as hook_b:

        for xb in loader:
            xb = xb.to(device, non_blocking=True)
            with torch.no_grad():
                _ = model_a(xb)
                _ = model_b(xb)

        reps_a = hook_a.stacked()  # {name: [N, Ca]}
        reps_b = hook_b.stacked()  # {name: [N, Cb]}

    # Sanity
    if len(reps_a) == 0 or len(reps_b) == 0:
        raise RuntimeError("No features captured. Check layer names and that layers output tensors.")

    # Compute CKA matrix
    M, rows, cols = compute_cka_matrix(reps_a, reps_b)
    print("CKA matrix shape:", M.shape)

    # Plot heatmap
    plot_heatmap(M, rows, cols, args.save_dir)

    # Print top-5 most similar pairs as text
    flat_idx = np.dstack(np.unravel_index(np.argsort(-M.ravel()), M.shape))[0]
    print("\nTop-5 similar layer pairs (Linear CKA):")
    for k in range(min(5, flat_idx.shape[0])):
        i, j = flat_idx[k]
        print(f"  {rows[i]}  <->  {cols[j]} : {M[i, j]:.4f}")

    if args.save_dir:
        np.save(args.save_dir / "cka_matrix.npy", M)
        with open(args.save_dir / "layers_a.txt", "w", encoding="utf-8") as fa:
            fa.write("\n".join(rows))
        with open(args.save_dir / "layers_b.txt", "w", encoding="utf-8") as fb:
            fb.write("\n".join(cols))
        print(f"Saved matrix and layer lists to: {args.save_dir}")


if __name__ == "__main__":
    main()
