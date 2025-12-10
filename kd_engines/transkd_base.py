# kd_engines/transkd_base.py

from __future__ import annotations
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_engine import BaseKDEngine
from .transkd_ops import hcl, CriterionCWD, CriterionEmbedCWD


class TransKDBaseEngine(BaseKDEngine):
    """
    TransKD-Base 스타일 KD 엔진 (INHO 프로젝트 버전).

    - teacher, student: SegFormer 계열 (ex: B3 -> B0)
    - forward(x, return_feats=True, return_embeds=True) 지원 필요
      * logits: (B, num_classes, H, W)
      * feats:  List[Tensor], 각 (B,C_i,H_i,W_i), stage 1~4
      * embeds: List[Tensor], 각 (B,N_i,D_i), patch embedding

    Loss 구성:
      L = w_ce * CE(student_logits, GT)
        + w_hcl * HCL(student_feats, teacher_feats)
        + w_cwd * Σ_i CWD(student_feats[i], teacher_feats[i])
        + w_embed * EmbedKD(student_embeds, teacher_embeds)   # MSE or EmbedCWD
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        w_ce: float = 1.0,
        w_hcl: float = 1.0,
        w_cwd: float = 0.0,
        w_embed: float = 0.0,
        embed_mode: str = "mse",  # "mse" or "cwd"
        ignore_index: int = 255,
        freeze_teacher: bool = True,
        cwd_norm_type: str = "channel",
        cwd_divergence: str = "mse",
        cwd_temperature: float = 1.0,
        embed_cwd_norm_type: str = "channel",
        embed_cwd_divergence: str = "mse",
        embed_cwd_temperature: float = 1.0,
    ) -> None:
        super().__init__(teacher, student)

        self.w_ce = float(w_ce)
        self.w_hcl = float(w_hcl)
        self.w_cwd = float(w_cwd)
        self.w_embed = float(w_embed)
        self.embed_mode = embed_mode
        self.ignore_index = ignore_index

        if freeze_teacher:
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()

        # feature CWD
        self.criterion_cwd = CriterionCWD(
            norm_type=cwd_norm_type,
            divergence=cwd_divergence,
            temperature=cwd_temperature,
        )

        # embedding CWD
        self.criterion_embed_cwd = CriterionEmbedCWD(
            norm_type=embed_cwd_norm_type,
            divergence=embed_cwd_divergence,
            temperature=embed_cwd_temperature,
        )

        # embed MSE (TransKDBase에서 embedding alignment시 사용)
        self.mse_embed = nn.MSELoss(reduction="mean")

    @torch.no_grad()
    def _forward_teacher(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        teacher forward (gradient off).
        """
        out = self.teacher(x, return_feats=True, return_embeds=True)
        # teacher 인터페이스를 (logits, feats, embeds)로 맞춘다는 전제
        t_logits, t_feats, t_embeds = out
        return t_logits, t_feats, t_embeds

    def _forward_student(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        student forward (gradient on).
        """
        out = self.student(x, return_feats=True, return_embeds=True)
        s_logits, s_feats, s_embeds = out
        return s_logits, s_feats, s_embeds

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        INHO 프로젝트의 train loop에서 호출되는 entry.
        batch: {"image": (B,3,H,W), "label": (B,H,W) 또는 (B,1,H,W)}
        """
        imgs: torch.Tensor = batch["image"]
        gts: torch.Tensor = batch["label"]

        # label shape 정리 (B,H,W)
        if gts.dim() == 4:
            # (B,1,H,W) -> (B,H,W)
            gts = gts.squeeze(1)

        # 1) teacher / student forward
        with torch.no_grad():
            _, t_feats, t_embeds = self._forward_teacher(imgs)
        s_logits, s_feats, s_embeds = self._forward_student(imgs)

        # 2) CE loss (student vs GT)
        loss_ce = F.cross_entropy(
            s_logits,
            gts,
            ignore_index=self.ignore_index,
        )

        # 3) HCL (multi-stage feature distillation)
        loss_hcl = torch.tensor(0.0, device=s_logits.device)
        if self.w_hcl > 0.0:
            loss_hcl = hcl(s_feats, t_feats)

        # 4) CWD (channel-wise distillation per stage)
        loss_cwd = torch.tensor(0.0, device=s_logits.device)
        if self.w_cwd > 0.0:
            for fs, ft in zip(s_feats, t_feats):
                loss_cwd = loss_cwd + self.criterion_cwd(fs, ft)

        # 5) Embedding KD (patch embedding alignment)
        loss_embed = torch.tensor(0.0, device=s_logits.device)
        if self.w_embed > 0.0 and len(s_embeds) > 0 and len(t_embeds) > 0:
            # stage 수가 다르면 min 길이까지만
            num_levels = min(len(s_embeds), len(t_embeds))

            if self.embed_mode == "mse":
                for i in range(num_levels):
                    # shape align: (B,N,C) 통일 가정
                    se = s_embeds[i]
                    te = t_embeds[i].detach()
                    # 필요 시 proj/reshape 추가 가능
                    loss_embed = loss_embed + self.mse_embed(se, te)

            elif self.embed_mode == "cwd":
                for i in range(num_levels):
                    se = s_embeds[i]
                    te = t_embeds[i].detach()
                    loss_embed = loss_embed + self.criterion_embed_cwd(se, te)

            else:
                raise ValueError(f"Unsupported embed_mode: {self.embed_mode}")

        # 6) total loss
        total_loss = (
            self.w_ce * loss_ce
            + self.w_hcl * loss_hcl
            + self.w_cwd * loss_cwd
            + self.w_embed * loss_embed
        )

        # 로그용 dict 반환
        return {
            "loss": total_loss,
            "loss_ce": loss_ce.detach(),
            "loss_hcl": loss_hcl.detach(),
            "loss_cwd": loss_cwd.detach(),
            "loss_embed": loss_embed.detach(),
        }
