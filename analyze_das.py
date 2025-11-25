import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import math

import config
import data_loader
from models import create_model

# ==========================================
# 사용자 설정
# ==========================================
# 학습이 완료된 체크포인트 경로 (수정 필수)
CHECKPOINT_PATH = Path(r"D:\LAB\result_files\test_results\Bset_LR_d3presnet50_SWTdescriptor_FiLM\best_model.pth")

# 분석할 배치의 수
NUM_BATCHES = 20


def analyze_das_effective_stats():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"▶ Device: {device}")
    print(f"▶ Loading Checkpoint: {CHECKPOINT_PATH}")

    # 1. 모델 로드
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    # 모델 생성 (config.MODEL.NAME 등에 따라 d3p 생성)
    model = create_model(config.MODEL.NAME)

    # 가중치 로드
    try:
        ckpt = torch.load(str(CHECKPOINT_PATH), map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(str(CHECKPOINT_PATH), map_location=device)

    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    if not getattr(model, 'use_das', False):
        print("[Error] DAS 모드가 꺼져 있습니다.")
        return

    print("▶ Model loaded. Starting Effective Gamma/Beta Analysis...")

    # 2. 데이터 로더
    loader = data_loader.test_loader

    # 통계 저장소
    stats = {
        'gamma': {},  # Stage별 Gamma 모음
        'beta': {}  # Stage별 Beta 모음
    }

    # Descriptor Score 통계
    desc_scores_log = []

    # ==========================================
    # 3. 분석 루프
    # ==========================================
    with torch.no_grad():
        for i, (imgs, _) in enumerate(tqdm(loader, total=NUM_BATCHES, desc="Analyzing")):
            if i >= NUM_BATCHES:
                break

            imgs = imgs.to(device)

            # A) Descriptor Inference & Safety Logic
            scores = model.descriptor(imgs)  # [B, 3]

            # [Safety Logic 재현] High > 0.7 -> Mid >= 0.5
            high_score = scores[:, 2]
            mask = high_score > 0.7
            if mask.any():
                min_val = torch.tensor(0.5, device=scores.device)
                scores[mask, 1] = torch.max(scores[mask, 1], min_val)

            desc_scores_log.append(scores.cpu())

            # B) DAS Block Simulation
            for stage_idx, block in enumerate(model.das_blocks):
                if isinstance(block, nn.Identity):
                    continue

                # MLP Forward
                params = block.mlp(scores)

                # Split
                in_ch = block.in_channels
                gamma_raw, beta_raw = torch.split(params, in_ch, dim=1)

                # ==========================================
                # [CORE] Soft Gating 수식 적용 (Effective Value 계산)
                # ==========================================
                ALPHA = 0.2
                BETA_LIMIT = 0.5

                # 실제 Feature에 곱해지는 값
                gamma_effective = 1.0 + ALPHA * torch.tanh(gamma_raw)
                # 실제 Feature에 더해지는 값
                beta_effective = BETA_LIMIT * torch.tanh(beta_raw)

                # 수집
                if stage_idx not in stats['gamma']:
                    stats['gamma'][stage_idx] = []
                    stats['beta'][stage_idx] = []

                stats['gamma'][stage_idx].append(gamma_effective.cpu())
                stats['beta'][stage_idx].append(beta_effective.cpu())

    # ==========================================
    # 4. 리포트 출력
    # ==========================================

    # A) Descriptor Score 분포
    all_scores = torch.cat(desc_scores_log, dim=0)
    print("\n" + "=" * 50)
    print("[Descriptor Stats (Input to DAS)]")
    print(f"  Low Band : Mean={all_scores[:, 0].mean():.4f}, Std={all_scores[:, 0].std():.4f}")
    print(f"  Mid Band : Mean={all_scores[:, 1].mean():.4f}, Std={all_scores[:, 1].std():.4f}")
    print(f"  High Band: Mean={all_scores[:, 2].mean():.4f}, Std={all_scores[:, 2].std():.4f}")
    print("=" * 50 + "\n")

    # B) Gamma (Scale) Stats
    print("[Effective Gamma Stats] (Target: 0.8 ~ 1.2)")
    print(f"{'Stage':<6} | {'Ch':<4} | {'Mean':<8} | {'Std':<8} | {'Min':<8} | {'Max':<8}")
    print("-" * 60)
    for stage_idx in sorted(stats['gamma'].keys()):
        g_data = torch.cat(stats['gamma'][stage_idx], dim=0)
        ch = g_data.shape[1]
        print(
            f"{stage_idx:<6} | {ch:<4} | {g_data.mean():<8.4f} | {g_data.std():<8.4f} | {g_data.min():<8.4f} | {g_data.max():<8.4f}")
    print("-" * 60 + "\n")

    # C) Beta (Shift) Stats
    print("[Effective Beta Stats] (Target: -0.5 ~ 0.5)")
    print(f"{'Stage':<6} | {'Ch':<4} | {'Mean':<8} | {'Std':<8} | {'Min':<8} | {'Max':<8}")
    print("-" * 60)
    for stage_idx in sorted(stats['beta'].keys()):
        b_data = torch.cat(stats['beta'][stage_idx], dim=0)
        ch = b_data.shape[1]
        print(
            f"{stage_idx:<6} | {ch:<4} | {b_data.mean():<8.4f} | {b_data.std():<8.4f} | {b_data.min():<8.4f} | {b_data.max():<8.4f}")
    print("=" * 60)

    # D) 해석 가이드
    print("\n[평가 가이드]")
    print("1. Gamma Range: 모든 Stage의 Min/Max는 반드시 [0.8, 1.2] 사이여야 합니다.")
    print("2. Gamma Std: 0.001 ~ 0.05 사이면 '안정적인 미세 조정' 중입니다.")
    print("   -> 만약 Std가 0.0000이면 모듈이 죽은 것입니다 (LR 문제).")
    print("   -> 만약 Min이 0.8000, Max가 1.2000에 딱 붙어있으면 'Saturation(포화)' 상태입니다.")
    print("3. Beta Range: Min/Max는 [-0.5, 0.5] 사이여야 합니다.")


if __name__ == "__main__":
    analyze_das_effective_stats()