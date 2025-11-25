# measure_modelonly_fps.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.profiler
from torchinfo import summary  # torchinfo 라이브러리

import config
from models import create_model

INPUT_SIZE = (1, 3, 360, 480)  # (batch size, C, H, W)
WARMUP = 50
RUNS = 200

AMP = False
CHANNELS_LAST = False


class PadToMultiple(nn.Module):
    """
    입력을 mult 배수로 패딩한 뒤 모델을 돌리고, 출력(logits 또는 tensor)을 원래 HxW로 크롭해서 반환.
    - 모델이 SMP(텐서 반환)든 HF(.logits 필드)든 상관없이 동작.
    - 속도 측정 목적이므로 업샘플은 하지 않고, 단순 크롭만 수행.
    """
    def __init__(self, model: nn.Module, mult: int = 16, keep_memory_format: bool = False):
        super().__init__()
        self.model = model
        self.mult = mult
        self.keep_memory_format = keep_memory_format

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        ph = (self.mult - (h % self.mult)) % self.mult
        pw = (self.mult - (w % self.mult)) % self.mult

        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))

        if self.keep_memory_format and x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)

        out = self.model(x)

        if hasattr(out, "logits"):
            out = out.logits

        out = out[..., :h, :w]
        return out


@torch.inference_mode()
def measure_model_only_latency(model,
                               input_size=(1, 3, 360, 480),
                               warmup=50,
                               runs=200,
                               amp=False,
                               channels_last=False):
    assert torch.cuda.is_available(), "CUDA(GPU)가 필요합니다."
    device = torch.device("cuda")
    model.eval().to(device)

    if channels_last:
        model.to(memory_format=torch.channels_last)

    x = torch.randn(input_size, device=device, dtype=torch.float32)
    if channels_last:
        x = x.to(memory_format=torch.channels_last)

    torch.backends.cudnn.benchmark = True

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    def fwd_once(inp):
        out = model(inp)
        if hasattr(out, "logits"):
            out = out.logits
        return out

    # Warm-up
    for _ in range(warmup):
        if amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = fwd_once(x)
        else:
            _ = fwd_once(x)
    torch.cuda.synchronize()

    # Measure
    times_ms = []
    for _ in range(runs):
        torch.cuda.synchronize()
        starter.record()
        if amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = fwd_once(x)
        else:
            _ = fwd_once(x)
        ender.record()
        torch.cuda.synchronize()
        times_ms.append(starter.elapsed_time(ender))

    t = np.array(times_ms, dtype=np.float64)
    stats = {
        "mean_ms": float(t.mean()),
        "median_ms": float(np.median(t)),
        "std_ms": float(t.std(ddof=1)) if runs > 1 else 0.0,
        "min_ms": float(t.min()),
        "max_ms": float(t.max()),
        "p95_ms": float(np.percentile(t, 95)),
        "p99_ms": float(np.percentile(t, 99)),
        "fps_from_median": float(1000.0 / np.median(t)),
        "runs": runs,
        "warmup": warmup,
        "input_size": list(input_size),
        "amp": amp,
        "channels_last": channels_last,
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    return stats


@torch.inference_mode()
def profile_model(model, input_size=(1, 3, 360, 480), log_dir="./log"):
    assert torch.cuda.is_available(), "CUDA(GPU)가 필요합니다."
    device = torch.device("cuda")
    model.eval().to(device)
    x = torch.randn(input_size, device=device, dtype=torch.float32)

    # warm-up
    for _ in range(10):
        _ = model(x)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
        record_shapes=True,
        with_stack=True
    ) as prof:
        for _ in range(10):
            model(x)
            prof.step()

    print(f"프로파일링 완료. 결과를 보려면 다음 명령어를 실행하세요:")
    print(f"tensorboard --logdir={log_dir}")


def main():
    print("Loading model...")
    base_model = create_model(config.MODEL.NAME)
    model = PadToMultiple(base_model, mult=16, keep_memory_format=CHANNELS_LAST)
    print("Done.")

    # ===== torchinfo로 파라미터 및 FLOPs 계산 =====
    try:
        model_stats = summary(model, input_size=INPUT_SIZE, verbose=0)
        total_params_M = model_stats.total_params / 1e6

        # MACs를 GFLOPs로 변환 (MACs * 2 / 1e9) - 1 MAC = 2 FLOPs 가정
        total_macs = model_stats.total_mult_adds
        gflops = (total_macs * 2) / 1e9

        print(f"Model analysis complete: {total_params_M:.2f} M params, {gflops:.2f} GFLOPs")
    except Exception as e:
        print(f"torchinfo analysis failed: {e}")
        total_params_M = -1.0
        gflops = -1.0

    # ===== latency 측정 =====
    stats = measure_model_only_latency(
        model,
        input_size=INPUT_SIZE,
        warmup=WARMUP,
        runs=RUNS,
        amp=AMP,
        channels_last=CHANNELS_LAST
    )

    # ===== CSV에 기록할 모델 분석 정보 추가 =====
    stats["model_name"] = getattr(config.MODEL, "NAME", "unknown")
    stats["parameters_M"] = round(float(total_params_M), 4)
    stats["gflops"] = round(float(gflops), 4)

    print("\n=== Model-only (forward only) ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    # ===== CSV 저장 =====
    try:
        import pandas as pd
        out_path = r"D:\LAB\result_files\analysis_results\fps_measure\model_only_latency.csv"

        # 한 번에 여러 모델 측정할 계획이라면, mode="a", header=not os.path.exists(out_path) 사용 가능
        df = pd.DataFrame([stats])
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"Saved: {out_path}")
    except Exception as e:
        print("CSV 저장 생략:", e)

    # ===== TensorBoard 프로파일링 =====
    profile_model(model, input_size=INPUT_SIZE, log_dir=r"D:\LAB\result_files\analysis_results\fps_measure\profile_log")


if __name__ == "__main__":
    main()
