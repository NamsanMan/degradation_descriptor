import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F   # ★ 추가

from config import DATA

@torch.inference_mode()
def evaluate_all(model, loader, device):
    model.eval()

    all_preds = []
    all_masks = []

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)  # ★ GPU로 옮겨서 shape 맞춤에 사용 (필수는 아니지만 안전)

        logits = model(imgs)  # (B,C,h,w) possibly low-res

        # ★ 핵심: GT mask 해상도에 맞춰 logits 업샘플 후 argmax
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        preds = torch.argmax(logits, dim=1)  # (B,H,W)

        all_preds.append(preds.cpu().numpy())
        all_masks.append(masks.cpu().numpy())

    # 1) 합치기
    preds_np = np.concatenate([p.flatten() for p in all_preds]).astype(np.int64)
    masks_np = np.concatenate([m.flatten() for m in all_masks]).astype(np.int64)

    # 2) 라벨 방어적 정규화
    oob_true = (masks_np != DATA.IGNORE_INDEX) & (
        (masks_np < 0) | (masks_np >= DATA.NUM_CLASSES)
    )
    masks_np[oob_true] = DATA.IGNORE_INDEX

    # 3) Void 제외
    valid = masks_np != DATA.IGNORE_INDEX
    masks_np = masks_np[valid]
    preds_np = preds_np[valid]

    # 4) 예측 클립
    if preds_np.size > 0:
        np.clip(preds_np, 0, DATA.NUM_CLASSES - 1, out=preds_np)

    # 5) Pixel Acc
    den = len(masks_np)
    pa = (np.sum(preds_np == masks_np) / den) if den > 0 else 0.0

    # 6) Confusion Matrix
    cm = confusion_matrix(masks_np, preds_np, labels=list(range(DATA.NUM_CLASSES)))

    # 7) IoU
    intersection = np.diag(cm)
    union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm)
    iou = np.zeros(DATA.NUM_CLASSES, dtype=np.float64)
    np.divide(intersection, union, out=iou, where=(union > 0))

    valid_classes_iou = [iou[c] for c in range(DATA.NUM_CLASSES) if c != DATA.IGNORE_INDEX and union[c] > 0]
    miou = np.nanmean(valid_classes_iou)

    per_class_iou = np.full(DATA.NUM_CLASSES, np.nan)
    for c in range(DATA.NUM_CLASSES):
        if c != DATA.IGNORE_INDEX and union[c] > 0:
            per_class_iou[c] = iou[c]

    return {
        "mIoU": miou,
        "PixelAcc": pa,
        "per_class_iou": per_class_iou,
        "confusion_matrix": cm
    }
