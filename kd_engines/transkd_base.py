# kd_engines/transkd_base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import config

from .base_engine import BaseKDEngine
from .transkd_csf import build_kd_trans
from .transkd_ops import hcl, CriterionCWD


def _split_student_teacher_imgs(imgs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Your dataloader can return:
      - imgs: Tensor (student only)
      - imgs: (student_img, teacher_img)
    """
    if isinstance(imgs, (tuple, list)) and len(imgs) == 2:
        if torch.is_tensor(imgs[0]) and torch.is_tensor(imgs[1]):
            return imgs[0], imgs[1]
    if torch.is_tensor(imgs):
        return imgs, imgs
    raise TypeError(f"Unsupported imgs type for TransKDBaseEngine: {type(imgs)}")


@dataclass
class TransKDBaseParams:
    # TransKD options
    knowledge_distillation_loss: str = "hcl"  # {"hcl","CWloss","KL"}
    embed: int = 5  # {0,1,2,3,4,5}

    # weights
    w_ce_student: float = 1.0
    w_ce_teacher: float = 0.0
    review_kd_loss_weight: float = 1.0  # HCL weight
    lambda_cwd: float = 1.0            # CWD weight

    # CWD options
    norm_type: str = "channel"         # {"channel","spatial","channel_mean","none"}
    divergence: str = "kl"             # {"kl","mse"}
    temperature: float = 1.0

    # embed loss
    embed_weights: Tuple[float, float, float, float] = (0.1, 0.1, 0.5, 1.0)

    ignore_index: int = config.DATA.IGNORE_INDEX


class TransKDBaseEngine(BaseKDEngine):
    """
    TransKD-Base KD engine adapted to your repo.

    Ports TransKD repo's:
      - train/CSF.py  (SK/SKF/build_kd_trans + HCL/CWD)
      - train/train_TransKDBase.py (loss composition logic)

    Required by your train_kd.py:
      - compute_losses(imgs, masks, device) -> dict including "total"
      - get_extra_parameters() -> iterable of params not already in student.parameters()
    """

    def __init__(self, teacher: nn.Module, student: nn.Module, **kwargs):
        super().__init__(teacher, student)

        # ---- params from config.KD.ENGINE_PARAMS ----
        p = {}
        try:
            p = config.KD.ENGINE_PARAMS or {}
        except Exception:
            p = {}

        self.params = TransKDBaseParams(
            knowledge_distillation_loss=str(p.get("knowledge_distillation_loss", p.get("kd_loss", "hcl"))),
            embed=int(p.get("embed", 5)),
            w_ce_student=float(p.get("w_ce_student", 1.0)),
            w_ce_teacher=float(p.get("w_ce_teacher", 0.0)),
            review_kd_loss_weight=float(p.get("review_kd_loss_weight", 1.0)),
            lambda_cwd=float(p.get("lambda_cwd", 1.0)),
            norm_type=str(p.get("norm_type", "channel")),
            divergence=str(p.get("divergence", "kl")),
            temperature=float(p.get("temperature", 1.0)),
            embed_weights=tuple(p.get("embed_weights", (0.1, 0.1, 0.5, 1.0))),
            ignore_index=int(getattr(config.DATA, "IGNORE_INDEX", 255)),
        )

        if self.params.knowledge_distillation_loss.lower() == "kl":
            self.params.norm_type = "spatial"

        # ---- CSF/SKF wrap student backbone ----
        csf_in = p.get("csf_in_channels", [32, 64, 160, 256])
        csf_out = p.get("csf_out_channels", [64, 128, 320, 512])
        self.student_kd = build_kd_trans(self.student, self.params.embed, in_channels=csf_in, out_channels=csf_out)

        # ---- losses ----
        self.ce = nn.CrossEntropyLoss(ignore_index=self.params.ignore_index)

        # For TransKD: "CWloss" uses MSE divergence, "KL" uses KL divergence.
        kd_mode = self.params.knowledge_distillation_loss.lower()
        divergence = "mse" if kd_mode == "cwloss" else self.params.divergence

        self.criterion_cwd = CriterionCWD(
            norm_type=self.params.norm_type,
            divergence=divergence,
            temperature=self.params.temperature,
        )
        self.mse = nn.MSELoss()

        # cache extra params (SKF params not in original student backbone)
        self._extra_params = self._collect_extra_params()

        # ✅ 논문 비교실험용: patch embedding hook 강제
        if hasattr(self.teacher, "set_force_patch_embeds"):
            self.teacher.set_force_patch_embeds(True)
        if hasattr(self.student, "set_force_patch_embeds"):
            self.student.set_force_patch_embeds(True)

    def _collect_extra_params(self) -> List[nn.Parameter]:
        base_ids = {id(p) for p in self.student.parameters()}
        extra: List[nn.Parameter] = []
        for p in self.student_kd.parameters():
            if id(p) not in base_ids:
                extra.append(p)
        return extra

    def get_extra_parameters(self) -> Iterable[nn.Parameter]:
        return self._extra_params

    @torch.no_grad()
    def _forward_teacher(self, x_t: torch.Tensor):
        # ensure teacher is in eval mode (important even if globally frozen)
        self.teacher.eval()
        return self.teacher(x_t, is_feat=True)

    def _forward_student(self, x_s: torch.Tensor):
        # SKF wrapper returns (features, logits, embeds?) under is_feat=True
        return self.student_kd(x_s, is_feat=True)

    def compute_losses(self, imgs: Any, masks: torch.Tensor, device) -> Dict[str, Any]:
        x_s, x_t = _split_student_teacher_imgs(imgs)
        if masks.dim() == 4 and masks.size(1) == 1:
            masks = masks[:, 0]

        # --- student forward ---
        s_out = self._forward_student(x_s)
        if self.params.embed == 0:
            fstudent, s_logits = s_out
            estudent = None
        else:
            fstudent, s_logits, estudent = s_out

        # --- teacher forward ---
        with torch.no_grad():
            fteacher, t_logits, eteacher = self._forward_teacher(x_t)

        # --- CE (student) ---
        ce_student = self.ce(s_logits, masks) * self.params.w_ce_student

        # --- optional teacher CE (usually 0 in your setting) ---
        ce_teacher = torch.tensor(0.0, device=s_logits.device)
        if self.params.w_ce_teacher > 0.0:
            ce_teacher = self.ce(t_logits, masks) * self.params.w_ce_teacher

        # --- feature KD ---
        kd_feat = torch.tensor(0.0, device=s_logits.device)
        kd_mode = self.params.knowledge_distillation_loss.lower()

        if kd_mode == "hcl":
            kd_feat = hcl(fstudent, fteacher) * self.params.review_kd_loss_weight
        elif kd_mode in ("cwloss", "kl"):
            loss_cwd = 0.0
            for fs, ft in zip(fstudent, fteacher):
                loss_cwd = loss_cwd + self.criterion_cwd(fs, ft)
            kd_feat = self.params.lambda_cwd * loss_cwd
        else:
            raise ValueError(f"Unsupported knowledge_distillation_loss: {self.params.knowledge_distillation_loss}")

        # --- embedding KD (TransKD train_TransKDBase.py uses MSE) ---
        kd_embed = torch.tensor(0.0, device=s_logits.device)
        if self.params.embed != 0:
            if estudent is None or eteacher is None:
                raise RuntimeError("embed KD requested but embeddings are missing from student/teacher outputs.")

            if self.params.embed == 5:
                w = self.params.embed_weights
                if len(w) != 4:
                    raise ValueError("embed_weights must have length 4 when embed==5")
                for i in range(4):
                    kd_embed = kd_embed + float(w[i]) * self.mse(estudent[i], eteacher[i])
            else:
                kd_embed = self.mse(estudent[0], eteacher[self.params.embed - 1])

        total = ce_student + ce_teacher + kd_feat + kd_embed

        # --- return dict for your logger ---
        return {
            "total": total,
            "ce_student": ce_student.detach(),
            "ce_teacher": ce_teacher.detach(),
            "kd_feat": kd_feat.detach(),
            "kd_embed": kd_embed.detach(),

            # for your train_kd.py visualization utilities
            "s_logits": s_logits.detach(),
            "t_logits": t_logits.detach(),
            "student_input": x_s.detach(),
            "teacher_input": x_t.detach(),
        }

    # optional hook for your training loop (safe no-op)
    def set_epoch(self, epoch: int):
        return
