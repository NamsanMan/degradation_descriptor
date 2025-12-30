import os
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import torchvision.transforms as T

import config

# Dateset 구현
class CamVidDataset(Dataset):
    #사용자 정의 Dataset 클래스는 반드시 3개 함수를 구현해야 합니다: __init__, __len__, and __getitem__.

    #__init__ 매서드(class안의 함수를 의미) Dataset 객체가 생성(instantiate)될 때 한 번만 실행됩니다. 여기서는 이미지와 주석 파일(annotation_file)이 포함된 디렉토리와 (다음 장에서 자세히 살펴볼) 두가지 변형(transform)을 초기화합니다.
    def __init__(self, images_dir, masks_dir, file_list=None, transform=None, teacher_images_dir=None):
        self.images_dir = images_dir
        self.teacher_images_dir = teacher_images_dir
        self.masks_dir = masks_dir
        if file_list:
            with open(file_list) as f:
                self.files = [line.strip() for line in f]
        else:
            self.files = sorted(f for f in os.listdir(images_dir) if f.endswith(".png"))
        self.transform = transform

    #__len__ 함수는 데이터셋의 샘플 개수를 반환합니다.
    def __len__(self):
        return len(self.files)

    #__getitem__ 함수는 주어진 인덱스 idx 에 해당하는 샘플을 데이터셋에서 불러오고 반환합니다.
    def __getitem__(self, idx):
        filename = self.files[idx]
        img_path = os.path.join(self.images_dir, filename)       #이미지 파일들이 저장된 디렉토리 경로
        mask_path = os.path.join(self.masks_dir, filename)       #레이블(마스크)가 저장된 디렉토리 경로
        image = Image.open(img_path).convert('RGB')                     #원본이 흑백이더라도 3채널로 맞춰 주며 3채널(RGB)형식으로 변환
        mask = Image.open(mask_path)  # 클래스 인덱스 그대로  >>>  주의해야될 점이, 마스크 파일이 png이고 0~11값이 class이어야 하고, void가 11이어야 함

        teacher_image = None
        if self.teacher_images_dir is not None:
            teacher_path = os.path.join(self.teacher_images_dir, filename)
            if not os.path.exists(teacher_path):
                raise FileNotFoundError(f"Teacher HR image not found: {teacher_path}")
            teacher_image = Image.open(teacher_path).convert('RGB')

        if self.transform:
            if teacher_image is not None:
                image, mask, teacher_image = self.transform(image, mask, teacher_image)
            else:
                image, mask = self.transform(image, mask)

        if teacher_image is not None:
            return (image, teacher_image), mask
        return image, mask

# train set에 대한 data augmentation: random crop, random flip, color jitter
class TrainAugmentation:
    def __init__(
        self,
        size,
        hflip_prob: float = 0.5,
        crop_prob: float = 0.7,
        crop_range: tuple[float, float] = (80.0, 100.0),
        brightness: tuple[float, float] = (0.6, 1.4),
        contrast: tuple[float, float]   = (0.7, 1.2),
        saturation: tuple[float, float] = (0.9, 1.3),
        hue: tuple[float, float]        = (-0.05, 0.05),
    ):
        self.size = size
        self.hflip_prob = hflip_prob

        self.crop_prob = crop_prob
        self.crop_min, self.crop_max = crop_range

        self.brightness = brightness
        self.contrast   = contrast
        self.saturation = saturation
        self.hue        = hue

    def __call__(self, img, mask):
        # 0) 초기 리사이즈
        img  = F.resize(img,  self.size)
        mask = F.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)

        # 1) 랜덤 크롭 (70% 확률)
        if random.random() < self.crop_prob:
            # 예: 원본의 80~100% 영역을 무작위 크롭 후 원래 크기로 리사이즈
            target_h, target_w = self.size
            scale_min = self.crop_min / 100.0
            scale_max = self.crop_max / 100.0
            crop_h = int(random.uniform(scale_min, scale_max) * target_h)
            crop_w = int(random.uniform(scale_min, scale_max) * target_w)
            i, j, h, w = T.RandomCrop.get_params(img, output_size=(crop_h, crop_w))
            img = F.crop(img, i, j, h, w)
            mask = F.crop(mask, i, j, h, w)
            img = F.resize(img, self.size, interpolation=InterpolationMode.BILINEAR)
            mask = F.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)

        # 2) 랜덤 수평 뒤집기
        if random.random() < self.hflip_prob:
            img  = F.hflip(img)
            mask = F.hflip(mask)

        # 4) 컬러 지터
        b = random.uniform(*self.brightness)
        c = random.uniform(*self.contrast)
        s = random.uniform(*self.saturation)
        h = random.uniform(*self.hue)
        img = F.adjust_brightness(img, b)
        img = F.adjust_contrast(img,   c)
        img = F.adjust_saturation(img, s)
        img = F.adjust_hue(img,        h)

        # 5) 텐서 변환 & 정규화
        img  = F.to_tensor(img)
        img  = F.normalize(img, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        # ★ 라벨 정규화: [0..10, 11] 이외는 11(Void)로 치환
        mask_np = np.array(mask, dtype=np.int64)
        mask_np[(mask_np < 0) | (mask_np > config.DATA.IGNORE_INDEX)] = config.DATA.IGNORE_INDEX
        mask = torch.from_numpy(mask_np).long()

        return img, mask


# 이미지와 마스크(레이블)을 동시에 전처리하기 위해 만든다
class SegmentationTransform:
    def __init__(self, size):
        # 크기(사이즈)
        self.size = size
    def __call__(self, img, mask):
        img = F.resize(img, self.size)
        mask = F.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)
        img = F.to_tensor(img)
        img = F.normalize(img, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        # ★ 라벨 정규화: [0..10, 11] 이외는 11(Void)로 치환
        mask_np = np.array(mask, dtype=np.int64)
        mask_np[(mask_np < 0) | (mask_np > config.DATA.IGNORE_INDEX)] = config.DATA.IGNORE_INDEX
        mask = torch.from_numpy(mask_np).long()
        return img, mask

class JointKDTrainAugmentation:
    """학생(LR)과 교사(HR) 이미지를 동일한 기하 변환으로 다루는 KD 전용 변환."""

    def __init__(
        self,
        size,
        hflip_prob: float = 0.5,
        crop_prob: float = 0.7,
        crop_range: tuple[float, float] = (80.0, 100.0),
        brightness: tuple[float, float] = (0.6, 1.4),
        contrast: tuple[float, float]   = (0.7, 1.2),
        saturation: tuple[float, float] = (0.9, 1.3),
        hue: tuple[float, float]        = (-0.05, 0.05),
    ):
        self.size = size
        self.hflip_prob = hflip_prob
        self.crop_prob = crop_prob
        self.crop_min, self.crop_max = crop_range
        self.brightness = brightness
        self.contrast   = contrast
        self.saturation = saturation
        self.hue        = hue

    def _apply_same_color_jitter(self, img_student, img_teacher):
        """
        Apply identical photometric jitter to student and teacher images.
        This enforces identical input distribution for LR->LR KD.
        """
        b = random.uniform(*self.brightness)
        c = random.uniform(*self.contrast)
        s = random.uniform(*self.saturation)
        h = random.uniform(*self.hue)

        img_student = F.adjust_brightness(img_student, b)
        img_teacher = F.adjust_brightness(img_teacher, b)

        img_student = F.adjust_contrast(img_student, c)
        img_teacher = F.adjust_contrast(img_teacher, c)

        img_student = F.adjust_saturation(img_student, s)
        img_teacher = F.adjust_saturation(img_teacher, s)

        img_student = F.adjust_hue(img_student, h)
        img_teacher = F.adjust_hue(img_teacher, h)

        return img_student, img_teacher

    def __call__(self, img_student, mask, img_teacher=None):
        img_teacher = img_teacher if img_teacher is not None else img_student

        # 초기 리사이즈
        img_student = F.resize(img_student, self.size)
        img_teacher = F.resize(img_teacher, self.size)
        mask = F.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)

        # 동일 파라미터의 랜덤 크롭
        if random.random() < self.crop_prob:
            target_h, target_w = self.size
            scale_min = self.crop_min / 100.0
            scale_max = self.crop_max / 100.0
            crop_h = int(random.uniform(scale_min, scale_max) * target_h)
            crop_w = int(random.uniform(scale_min, scale_max) * target_w)
            i, j, h, w = T.RandomCrop.get_params(img_student, output_size=(crop_h, crop_w))
            img_student = F.crop(img_student, i, j, h, w)
            img_teacher = F.crop(img_teacher, i, j, h, w)
            mask = F.crop(mask, i, j, h, w)
            img_student = F.resize(img_student, self.size, interpolation=InterpolationMode.BILINEAR)
            img_teacher = F.resize(img_teacher, self.size, interpolation=InterpolationMode.BILINEAR)
            mask = F.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)

        # 랜덤 수평 뒤집기
        if random.random() < self.hflip_prob:
            img_student = F.hflip(img_student)
            img_teacher = F.hflip(img_teacher)
            mask = F.hflip(mask)

        # 공통 photometric jitter: student/teacher 동일 분포 강제
        img_student, img_teacher = self._apply_same_color_jitter(img_student, img_teacher)

        # Tensor 및 정규화
        img_student = F.to_tensor(img_student)
        img_student = F.normalize(img_student, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

        img_teacher = F.to_tensor(img_teacher)
        img_teacher = F.normalize(img_teacher, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

        mask_np = np.array(mask, dtype=np.int64)
        mask_np[(mask_np < 0) | (mask_np > config.DATA.IGNORE_INDEX)] = config.DATA.IGNORE_INDEX
        mask = torch.from_numpy(mask_np).long()

        return img_student, mask, img_teacher


_train_teacher_dir = config.DATA.TRAIN_TEACHER_IMG_DIR
if _train_teacher_dir is not None and os.path.exists(_train_teacher_dir):
    _train_transform = JointKDTrainAugmentation(
        size=config.DATA.INPUT_RESOLUTION,
    )
else:
    _train_teacher_dir = None
    _train_transform = TrainAugmentation(
        size=config.DATA.INPUT_RESOLUTION,
    )

# A_set 만 B_set으로 바꿔서 2fold 진행
train_dataset = CamVidDataset(
    images_dir = config.DATA.TRAIN_IMG_DIR,
    masks_dir  = config.DATA.TRAIN_LABEL_DIR,
    file_list = config.DATA.FILE_LIST,
    transform = _train_transform,
    teacher_images_dir=_train_teacher_dir,
)

val_dataset = CamVidDataset(
    images_dir = config.DATA.VAL_IMG_DIR,
    masks_dir  = config.DATA.VAL_LABEL_DIR,
    file_list = config.DATA.FILE_LIST,
    transform =SegmentationTransform(config.DATA.INPUT_RESOLUTION)
)

test_dataset = CamVidDataset(
    images_dir = config.DATA.TEST_IMG_DIR,
    masks_dir  = config.DATA.TEST_LABEL_DIR,
    file_list = config.DATA.FILE_LIST,
    transform =SegmentationTransform(config.DATA.INPUT_RESOLUTION)
)

# 데이터셋을 train.py로 넘겨줌
# train_loader에만 shuffle true
# num_workers는 전부 같은 값으로 통일(0 아니면 1)
# val_loader와 test_lodaer의 batch_size는 1로 하는게 맞고, train_loader의 batchsize는 4를 추천
train_loader = DataLoader(train_dataset, batch_size=config.DATA.BATCH_SIZE,  shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=1,  shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=1,  shuffle=False, num_workers=0)