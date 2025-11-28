from pathlib import Path
import numpy as np
import os

# 'COLAB_GPU'는 Colab 환경에만 존재하는 환경 변수입니다.
IS_COLAB = 'COLAB_GPU' in os.environ

#  2. 플래그 값에 따라 경로를 다르게 설정
if IS_COLAB:
    # --- Colab 환경일 때의 경로 ---
    print("▶ Running in Google Colab environment.")

    # Colab의 구글 드라이브 경로를 기본 경로로 설정
    BASE_DRIVE_DIR = Path('/content/drive/MyDrive/LAB')

    DATA_DIR = BASE_DRIVE_DIR / "datasets/project_use/CamVid_12_2Fold_v4/B_set"
    BASE_DIR = BASE_DRIVE_DIR / "result_files/test_results"

    # KD용 weight load
    TEACHER_CKPT = BASE_DRIVE_DIR / 'result_files/test_results/Aset_LR_seg5_new/best_model.pth'

else:
    # --- 로컬 환경일 때의 경로 ---
    print("▶ Running in local environment.")

    # 기존에 사용하시던 로컬 경로 설정
    DATA_HR_DIR = Path(r"D:\LAB\datasets\project_use\CamVid_12_2Fold_v4\A_set")
    DATA_DIR = Path(r"D:\LAB\datasets\project_use\CamVid_12_2Fold_LR_x4_Bilinear\A_set")
    BASE_DIR = Path(r"D:\LAB\result_files\test_results")

    # KD용 weight load
    TEACHER_CKPT = r'D:\LAB\result_files\test_results\Aset_HR_segb3\best_model.pth'

# ──────────────────────────────────────────────────────────────────
# 1. GENERAL: 프로젝트 전반 및 실험 관리 설정
# ──────────────────────────────────────────────────────────────────
class GENERAL:
    # 실험 프로젝트 이름
    PROJECT_NAME = "Aset_segb3_300epoch_debug"

    # 결과 파일을 저장할 기본 경로
    BASE_DIR = BASE_DIR / PROJECT_NAME
    LOG_DIR = BASE_DIR / "log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    SUMMARY_TXT = LOG_DIR / "training_summary.txt"
    SAVE_PLOT = LOG_DIR / "training_progress.png"

    SEED = 42

# ──────────────────────────────────────────────────────────────────
# 2. DATA: 데이터셋 관련 설정
# ──────────────────────────────────────────────────────────────────
class DATA:
    # 데이터셋 경로
    DATA_HR_DIR   = DATA_HR_DIR  # data_loader에서 사용하던 경로
    TRAIN_HR_DIR = DATA_HR_DIR / "train"
    VAL_HR_DIR = DATA_HR_DIR / "val"
    TEST_HR_DIR = DATA_HR_DIR / "test"

    DATA_DIR   = DATA_DIR  # data_loader에서 사용하던 경로
    TRAIN_DIR = DATA_DIR / "train"
    VAL_DIR = DATA_DIR / "val"
    TEST_DIR = DATA_DIR / "test"

    TRAIN_IMG_DIR = TRAIN_DIR / "images"
    TRAIN_TEACHER_IMG_DIR = TRAIN_HR_DIR / "images"  # teacher(HR) 전용 오프라인 클린 이미지
    TRAIN_LABEL_DIR = TRAIN_DIR / "labels"
    VAL_IMG_DIR = VAL_DIR / "images"
    VAL_TEACHER_IMG_DIR = VAL_HR_DIR / "images"
    VAL_LABEL_DIR = VAL_DIR / "labels"
    TEST_IMG_DIR = TEST_DIR / "images"
    TEST_TEACHER_IMG_DIR = TEST_HR_DIR / "images"
    TEST_LABEL_DIR = TEST_DIR / "labels"

    FILE_LIST = None

    # 입력 이미지 해상도 >> 원본 이미지의 크기가 아닌 모델에 들어가게 되는 input size
    INPUT_RESOLUTION = (360, 480)  # H, W

    # 배치 사이즈 및 데이터 로딩 워커 수
    BATCH_SIZE = 4

    # 클래스 정보
    CLASS_NAMES = [
        "Sky", "Building", "Pole", "Road", "Sidewalk",
        "Tree", "SignSymbol", "Fence", "Car",
        "Pedestrian", "Bicyclist", "Void"
    ]
    NUM_CLASSES = len(CLASS_NAMES)  # =12
    IGNORE_INDEX = 11  # 'Void' 클래스의 인덱스

    # grayscale label(ground truth 포함)을 공식 컬러 매핑과 동일하게 시각화를 위해 컬러 매핑
    CLASS_COLORS = np.array([
        [128, 128, 128],  # Sky
        [128, 0, 0],  # Building
        [192, 192, 128],  # Pole
        [128, 64, 128],  # Road
        [0, 0, 192],  # Sidewalk
        [128, 128, 0],  # Tree
        [192, 128, 128],  # SignSymbol
        [64, 64, 128],  # Fence
        [64, 0, 128],  # Car
        [64, 64, 0],  # Pedestrian
        [0, 128, 192],  # Bicyclist
        [0, 0, 0],  # Void
    ], dtype=np.uint8)

# ──────────────────────────────────────────────────────────────────
# 2-1. PDM: Patchwise Degradation Mix 설정
# ──────────────────────────────────────────────────────────────────
class PDM:
    ENABLE = False          # 사용 여부
    APPLY_PROB = 1.0       # 적용 확률 (기본 100%)

    # 패치 관련
    PATCH_SIZE = 4         # "4x4 픽셀" 패치 (요구사항)
    REPLACE_RATIO = 0.20   # 전체 패치 중 교체 비율(0.0~1.0)

    # 다운스케일 비율(원본 대비) → (0.25, 0.5)이면 25%~50%로 축소 후 재업샘플
    DOWNSCALE_MIN = 0.25
    DOWNSCALE_MAX = 0.25

    # 가우시안 노이즈: 요구사항 고정
    GAUSS_MU = 0
    GAUSS_SIGMA_RANGE = (1, 30)  # 표준편차 U(1,30), 픽셀 스케일(0~255) 기준
    GRAY_NOISE_PROB = 0.40       # Gray 40%, Color 60%

    # 보간
    DOWNSCALE_INTERP = "bilinear"  # 'nearest' | 'bilinear' | 'bicubic'
    UPSCALE_INTERP   = "bilinear"

# ──────────────────────────────────────────────────────────────────
# 3. MODEL: 모델 설정
# ──────────────────────────────────────────────────────────────────

class MODEL:
    NAME = 'd3p'

    """
    available models:
    segformerb0
    segformerb1
    segformerb3
    segformerb4
    segformerb5
    d3p
    ddrnet23slim
    segformer_smp
    unet
    """

# ──────────────────────────────────────────────────────────────────
# 4. TRAIN: 훈련 과정 관련 설정
# ──────────────────────────────────────────────────────────────────
class TRAIN:
    EPOCHS = 300
    USE_WARMUP = True
    WARMUP_EPOCHS = 5

    # main문 실행시 checkpoint 로드할것인지 설정
    USE_CHECKPOINT = False
    CHECKPOINT_DIR = Path(r"D:\LAB\result_files\test_results\Aset_LR_segb3")

    # 딕셔너리 형태로 통일
    OPTIMIZER = {
        "NAME": "AdamW",
        "PARAMS": {
            "lr": 1e-4,
            "weight_decay": 1e-2
        }
    }

    SCHEDULER_RoP = {
        "NAME": "ReduceLROnPlateau",
        "PARAMS": {
            "mode": 'min',
            "factor": 0.5,
            "patience": 5,
            "min_lr": 1e-6
        }
    }

    SCHEDULER_CALR = {
        "NAME": "CosineAnnealingLR",
        "PARAMS": {
            "T_max": EPOCHS - WARMUP_EPOCHS,
            "eta_min": 1e-6
        }
    }

    LOSS_FN = {
        "NAME": "CrossEntropyLoss",
        "PARAMS": {
            "ignore_index": DATA.IGNORE_INDEX
        }
    }

    # Warmup도 동일한 구조로 추가
    WARMUP_SCHEDULER = {
        "NAME": "LinearLR",
        "PARAMS": {
            "start_factor": 0.1,
            "end_factor": 1.0,
        }
    }


# ──────────────────────────────────────────────────────────────────
# 5. KD: Knowledge Distillation 관련 설정
# ──────────────────────────────────────────────────────────────────
class KD:
    ENABLE = True

    ENGINE_NAME = "swt_attention"
    """
    available engines:
    segtoseg
    logit
    cross_arch_seg_kd   # cross_arch_seg_kd는 discriminator의 추가적인 parameter때문에 main_CAKD, train_CAKD를 별도로 이용해야됨
    hmkd
    gckd
    """

    # 모델 선택
    TEACHER_NAME = 'segformerb3'
    STUDENT_NAME = 'segformerb0'
    # 이미 학습된 teacher .pth 경로 (없으면 None), KD경로는 일단 colab경로로 해놓음

    # 교사 고정 여부
    FREEZE_TEACHER = True

    ALL_ENGINE_PARAMS = {
        "segtoseg": {
            "stage_weights": [0.25, 0.5, 0.75, 1.0],  # SegFormer 인코더 4단계 스테이지 가중치
            "t": 2.0,  # KD temperature
            "w_ce_student": 1.0,  # 학생 CE
            "w_ce_teacher": 0.0,  # 교사 CE (교사도 GT로 같이 fine tunning 하지 않으려면 0.0)
            "w_logit": 0.05,  # 로짓 KD
            "w_feat": 0.25,  # 피처 KD
            "ignore_index": DATA.IGNORE_INDEX,
            "use_logit_kd": True,  # logit KD 사용 여부 // teacher와 student의 label이 같지 않으면 false
            "feat_l2_normalize": True,  # 피처 KD 시 채널 방향 L2-정규화 사용 다음 실험때 이거 없애보기
            "freeze_teacher": FREEZE_TEACHER
        },
        "logit": {
            "w_ce_student": 1.0,
            "w_kd_logit": 0.1,
            "temperature": 2.0,
            "ignore_index": DATA.IGNORE_INDEX,
            "freeze_teacher": FREEZE_TEACHER
        },

        # cross_arch_seg_kd는 discriminator의 추가적인 parameter때문에 main_CAKD, train_CAKD를 별도로 이용해야됨
        "cross_arch_seg_kd": {
            "w_ce_student": 0.0,   # 논문 원문 충실 → 학생 CE 사용 안함
            "w_pca": 1.0,          # PCA loss (식 6,7)
            "w_gl": 1.0,           # GL loss (식 7)
            "w_mvg": 1.0e2,          # MVG loss (식 9,10), lambda 값
            "pca_qk_channels": 64,
            "pca_v_channels": 128,
            "gl_dropout_p": 0.1,
            "ignore_index": DATA.IGNORE_INDEX,
            "disc_hidden": (256, 64),
            "freeze_teacher": FREEZE_TEACHER
        },
        "hmkd": {
            # 손실 가중치
            "w_ce_student": 1.0,
            "w_gla": 1.0,  # PSAM(=GLA) 어텐션 MSE. 처음 0.05~0.2로 탐색 권장
            "w_hfa": 1.0,  # HSAM(HFA) 재구성 MSE. 0.5~2.0 범위 탐색

            "ignore_index": DATA.IGNORE_INDEX,

            # PSAM(=GLA)
            "gla_embed_dim": 64,  # 패치 임베딩 차원
            "gla_patch_size": 8,  # patch size (stride 기본값도 8)
            "gla_teacher_stage": 0,  # Hugginh Face 기준으로 SegFormer 인코더 stage 1 (0~3)
            "gla_student_stage": 0,  # 학생 인코더 대응 스테이지, smp deeplab이나 unet 기준으로 0은 입력이고, 1이 실질적인 첫번째 stage인데, 내 d3p.py 래퍼에서 0을 실직적인 1번 스테이지라고 재 정의함

            # HSAM(HFA)
            "hfa_aligned_channels": 160,  # proj 채널
            "hfa_offset_scale": 2.0,  # 오프셋 최대 픽셀 (±2)
            "hfa_align_corners": True,  # interpolate/grid_sample 일관 옵션
            "hfa_teacher_stage": -2,  # "-1"로 설정시 segformer MiT encoder의 stage 4
            "hfa_student_stage": -2,  # "-1"로 설정시 mobilenetV2의 stage 5

            "freeze_teacher": FREEZE_TEACHER
        },
        "gckd": {  # ← NEW
            # feature stage 선택(모델 래퍼/백본에 맞게 조정)
            "teacher_stage": -2,  # SegFormer: MiT stage-3(= -2) 등
            "student_stage": -2,  # MobileNetV2: stage-4(= -2) 등

            # LPF 설정
            "lpf_kernel": 3,  # AvgPool2d kernel (odd)
            "lpf_stride": 1,
            "lpf_pad": 1,

            # 도메인 정렬 블록
            "align_hidden": 256,  # 1x1 conv 중간 채널

            # 손실 가중치
            "w_gc": 1.0,  # L_GC 가중치

            # robustness weight
            "use_wrob": True,  # w_rob 사용 여부
            "beta": 0.5,  # exp( -beta * ||…||^2 )
            "freeze_teacher": True,  # True면 깨끗한 입력을 함께 넘긴다는 가정
        },
        "swt_attention": {
            "w_ce_student": 1.0,
            "w_kd_logit": 0.2,
            "w_kd_feat": 0.5,
            "temperature": 2.0,
            "ignore_index": DATA.IGNORE_INDEX,
            "teacher_stage": -2,
            "student_stage": -2,
            "energy_temperature": 1.5,
            "freeze_teacher": FREEZE_TEACHER,
        }
    }

    ENGINE_PARAMS = ALL_ENGINE_PARAMS[ENGINE_NAME]