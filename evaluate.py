import torch
import numpy as np
from sklearn.metrics import confusion_matrix

from config import DATA

@torch.inference_mode()
def evaluate_all(model, loader, device):
    model.eval()

    all_preds = []
    all_masks = []

    for imgs, masks in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = torch.argmax(logits, dim=1)

        all_preds.append(preds.cpu().numpy())
        all_masks.append(masks.cpu().numpy())

    # 1) 합치기
    preds_np = np.concatenate([p.flatten() for p in all_preds]).astype(np.int64)
    masks_np = np.concatenate([m.flatten() for m in all_masks]).astype(np.int64)

    # 2) ★ 라벨 방어적 정규화 (테스트 라벨이 16-bit 등일 때를 대비)
    #    [0..10] + 11(Void) 외 값은 Void로 보정
    oob_true = (masks_np != DATA.IGNORE_INDEX) & (
        (masks_np < 0) | (masks_np >= DATA.NUM_CLASSES)
    )
    masks_np[oob_true] = DATA.IGNORE_INDEX

    # 3) Void 제외
    valid = masks_np != DATA.IGNORE_INDEX
    masks_np = masks_np[valid]
    preds_np = preds_np[valid]

    # 4) ★ 예측도 안전하게 [0..10]으로 클립 (이론상 필요 없지만 방어적으로)
    if preds_np.size > 0:
        np.clip(preds_np, 0, DATA.NUM_CLASSES - 1, out=preds_np)

    # 5) Pixel Acc
    den = len(masks_np)
    pa = (np.sum(preds_np == masks_np) / den) if den > 0 else 0.0

    # 6) Confusion Matrix (이제 안전)
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