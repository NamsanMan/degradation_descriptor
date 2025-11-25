import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from scipy.stats import pearsonr

import config
import data_loader
from models import create_model
from descriptor_v3 import FrequencyDegradationTarget  # GT 생성용 (옵션)

# ==========================================
# 설정
# ==========================================
CHECKPOINT_PATH = Path(r"D:\LAB\result_files\test_results\Aset_LR_d3presnet50_SWTdescriptor_FiLM\best_model.pth")
OUTPUT_DIR = CHECKPOINT_PATH.parent / "impact_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 분석할 샘플 수 (전체 Test set 사용 권장)
NUM_SAMPLES = 300

# ==========================================
# 1. 데이터 수집 및 Hook 설정
# ==========================================
feature_stats = {
    "before": {},
    "after": {}
}


def get_das_hook(stage_idx):
    def hook(module, input, output):
        # input[0]: x (Feature Map)
        # output: x_new (Modulated Feature Map)
        # 배치 단위 평균/표준편차 저장
        with torch.no_grad():
            x_in = input[0].detach().cpu()
            x_out = output.detach().cpu()

            if stage_idx not in feature_stats["before"]:
                feature_stats["before"][stage_idx] = []
                feature_stats["after"][stage_idx] = []

            feature_stats["before"][stage_idx].append(x_in.mean(dim=(1, 2, 3)).numpy())  # Global Mean
            feature_stats["after"][stage_idx].append(x_out.mean(dim=(1, 2, 3)).numpy())

    return hook


def analyze_impact():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"▶ Device: {device}")

    # 모델 로드
    model = create_model(config.MODEL.NAME)
    try:
        ckpt = torch.load(str(CHECKPOINT_PATH), map_location=device, weights_only=False)
    except:
        ckpt = torch.load(str(CHECKPOINT_PATH), map_location=device)

    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # Hook 등록 (DAS Block 전후 비교용)
    hooks = []
    for i, block in enumerate(model.das_blocks):
        if not isinstance(block, nn.Identity):
            h = block.register_forward_hook(get_das_hook(i))
            hooks.append(h)

    # 데이터 로더
    loader = data_loader.test_loader

    # 데이터 저장소
    records = []

    print("▶ Collecting Data & Running Inference...")

    with torch.no_grad():
        for i, (imgs, masks) in enumerate(tqdm(loader)):
            if i * config.DATA.BATCH_SIZE >= NUM_SAMPLES:
                break

            imgs = imgs.to(device)

            # 1. Descriptor Score
            scores = model.descriptor(imgs)  # [B, 3]

            # 2. Gamma/Beta Extraction from MLP
            gammas = {}
            for stage_idx, block in enumerate(model.das_blocks):
                if isinstance(block, nn.Identity):
                    continue

                # Re-run MLP logic manually to get values
                params = block.mlp(scores)
                g_raw, b_raw = torch.split(params, block.in_channels, dim=1)

                # Soft Gating Formula
                gamma = 1.0 + 0.2 * torch.tanh(g_raw)
                beta = 0.5 * torch.tanh(b_raw)

                # Channel Mean 저장 (배치 내 각 샘플별)
                gammas[f"s{stage_idx}_gamma_mean"] = gamma.mean(dim=1).cpu().numpy()
                gammas[f"s{stage_idx}_beta_mean"] = beta.mean(dim=1).cpu().numpy()

            # 3. Forward Pass (Hook will collect features)
            _ = model(imgs)

            # 4. Record Data
            B = imgs.size(0)
            for b in range(B):
                rec = {
                    "low_score": scores[b, 0].item(),
                    "mid_score": scores[b, 1].item(),
                    "high_score": scores[b, 2].item(),
                }
                # Add Gamma/Beta info
                for k, v in gammas.items():
                    rec[k] = v[b]
                records.append(rec)

    # Hook 제거
    for h in hooks: h.remove()

    df = pd.DataFrame(records)
    return df


# ==========================================
# 2. 시각화 및 분석 함수
# ==========================================

def plot_score_vs_gamma(df, stage_idx, out_dir):
    """
    Descriptor 점수(X축)와 Gamma 값(Y축)의 상관관계 분석
    가설: 노이즈 점수(High)가 높으면 -> Gamma(Scale)가 낮아져야 한다(Negative Correlation).
    """
    plt.figure(figsize=(18, 5))

    bands = ["low", "mid", "high"]
    gamma_col = f"s{stage_idx}_gamma_mean"

    for i, band in enumerate(bands):
        score_col = f"{band}_score"

        plt.subplot(1, 3, i + 1)
        sns.regplot(data=df, x=score_col, y=gamma_col, scatter_kws={'alpha': 0.5, 's': 10}, line_kws={'color': 'red'})

        corr, _ = pearsonr(df[score_col], df[gamma_col])
        plt.title(f"{band.upper()} Score vs Stage {stage_idx} Gamma\nCorr: {corr:.3f}")
        plt.xlabel(f"{band} degradation score")
        plt.ylabel("Mean Gamma (Scale)")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / f"corr_stage_{stage_idx}.png")
    plt.close()
    print(f"Saved correlation plot for Stage {stage_idx}")


def plot_gamma_distribution(df, out_dir):
    """
    Stage별 Gamma 값의 분포 확인.
    너무 1.0에 몰려있으면(Peaked) 작동 안 함, 넓게 퍼져있으면 Active함.
    """
    gamma_cols = [c for c in df.columns if "gamma_mean" in c]

    plt.figure(figsize=(10, 6))
    for col in gamma_cols:
        sns.kdeplot(df[col], label=col, fill=True, alpha=0.3)

    plt.title("Distribution of Mean Gamma per Stage")
    plt.xlabel("Gamma Value (Scale)")
    plt.axvline(1.0, color='k', linestyle='--', label="Identity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "gamma_distribution.png")
    plt.close()


def plot_feature_shift(stats, out_dir):
    """
    DAS 적용 전/후 Feature Map의 평균 에너지 변화 시각화
    """
    # Stats dictionary to lists
    for stage_idx in stats["before"].keys():
        before = np.concatenate(stats["before"][stage_idx])  # [N]
        after = np.concatenate(stats["after"][stage_idx])  # [N]

        plt.figure(figsize=(6, 6))
        plt.scatter(before, after, s=5, alpha=0.5, c='purple')

        # y=x line (No change)
        min_val = min(before.min(), after.min())
        max_val = max(before.max(), after.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="No Change")

        plt.title(f"Feature Mean Activation Shift (Stage {stage_idx})")
        plt.xlabel("Before DAS (Mean)")
        plt.ylabel("After DAS (Mean)")
        plt.legend()
        plt.grid(True)
        plt.savefig(out_dir / f"feature_shift_stage_{stage_idx}.png")
        plt.close()


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    print("=== Starting DAS Impact Analysis ===")
    df = analyze_impact()

    print(f"Analyzed {len(df)} samples.")

    # 1. Gamma 분포 확인
    plot_gamma_distribution(df, OUTPUT_DIR)

    # 2. 상관관계 분석 (핵심)
    # Active Stage (보통 2, 5)에 대해 분석 수행
    active_stages = [int(c.split('_')[0][1:]) for c in df.columns if "gamma_mean" in c]
    active_stages = sorted(list(set(active_stages)))

    for s_idx in active_stages:
        plot_score_vs_gamma(df, s_idx, OUTPUT_DIR)

    # 3. Feature 변화 확인
    plot_feature_shift(feature_stats, OUTPUT_DIR)

    print(f"=== Analysis Complete. Results saved to {OUTPUT_DIR} ===")