from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_engine import BaseKDEngine


class VanillaLogitKD(BaseKDEngine):
    """
    Baseline logit-KD:
      L = CE(student_logits, GT) + w_kd * KL(teacher || student)

    - no spatial weighting
    - no feature KD
    - no attention
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        w_ce: float = 1.0,
        w_kd: float = 0.2,
        temperature: float = 2.0,
        ignore_index: int = 255,
        freeze_teacher: bool = True,
    ):
        super().__init__(teacher, student)

        self.w_ce = float(w_ce)
        self.w_kd = float(w_kd)
        self.temperature = float(temperature)
        self.ignore_index = int(ignore_index)

        self.ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        # 수정: reduction="none" 으로 두고, (N,C,H,W)에 대해 직접 평균
        self.kl = nn.KLDivLoss(reduction="none")

        if freeze_teacher:
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()

    def _forward_logits(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        out = model(x, return_feats=False)
        if isinstance(out, (tuple, list)):
            return out[0]
        if isinstance(out, dict):
            return out["logits"]
        return out

    def compute_losses(self, imgs, masks, device):
        """
        imgs:
          - Tensor: same input to teacher & student
          - (student_img, teacher_img): explicit LR/HR split
        """

        if isinstance(imgs, (tuple, list)):
            s_img, t_img = imgs
        else:
            s_img = t_img = imgs

        # --- forward ---
        s_logits = self._forward_logits(self.student, s_img)
        with torch.no_grad():
            t_logits = self._forward_logits(self.teacher, t_img)

        # --- CE (standard) ---
        loss_ce = self.ce(s_logits, masks)

        # --- logit KD: KL(teacher || student), pixel-wise 평균 ---
        T = self.temperature

        # (N,C,H,W)
        log_p_s = F.log_softmax(s_logits / T, dim=1)
        log_p_t = F.log_softmax(t_logits / T, dim=1)
        p_t = log_p_t.exp()

        # kl_map: (N,C,H,W), 각 위치에서 p_t * (log_p_t - log_p_s)
        kl_map = self.kl(log_p_s, p_t)          # (N,C,H,W)
        kl_map = kl_map.sum(dim=1)              # 채널 합산 → (N,H,W)
        loss_kd = kl_map.mean() * (T ** 2)      # 전체 평균 + T^2 스케일링

        total = self.w_ce * loss_ce + self.w_kd * loss_kd

        return {
            "total": total,
            "ce": loss_ce.detach(),
            "kd_logit": loss_kd.detach(),
            "student_input": s_img.detach(),
            "teacher_input": t_img.detach(),
            "s_logits": s_logits.detach(),
            "t_logits": t_logits.detach(),
        }
