from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_engine import BaseKDEngine


class VanillaLogitKD(BaseKDEngine):
    """
    Baseline logit-KD:
      L = CE(student_logits, GT) + w_kd * KL(student || teacher)

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
        self.kl = nn.KLDivLoss(reduction="batchmean")

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

        # --- logit KD (plain KL) ---
        log_p_s = F.log_softmax(s_logits / self.temperature, dim=1)
        p_t = F.softmax(t_logits / self.temperature, dim=1)

        loss_kd = self.kl(log_p_s, p_t) * (self.temperature ** 2)

        total = self.w_ce * loss_ce + self.w_kd * loss_kd

        return {
            "total": total,
            "ce": loss_ce.detach(),
            "kd_logit": loss_kd.detach(),
            "s_logits": s_logits.detach(),
            "t_logits": t_logits.detach(),
        }
