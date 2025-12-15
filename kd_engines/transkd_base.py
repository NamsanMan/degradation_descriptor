# kd_engines/transkd_base.py

from __future__ import annotations
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_engine import BaseKDEngine
from .transkd_ops import hcl, CriterionCWD, CriterionEmbedCWD


TensorOrList = Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]


class TransKDBaseEngine(BaseKDEngine):
    """
    TransKD-Base 스타일 KD 엔진 (LR/LR 입력 버전).

    - teacher, student: SegFormerWrapper (예: B3 -> B0)
    - teacher(student) forward 인터페이스:
        logits, feats, embeds = model(x, return_feats=True, return_embeds=True)

      * logits: (B, num_classes, H, W)
      * feats:  List[Tensor], 각 (B,C_i,H_i,W_i)
      * embeds: List[Tensor], 각 (B,N_i,C_i)   # patch embedding

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

        # embed MSE
        self.mse_embed = nn.MSELoss(reduction="mean")
        self.feat_adapters: nn.ModuleList | None = None
        self.embed_adapters: nn.ModuleList | None = None

    def _build_adapters(
        self,
        s_feats: List[torch.Tensor],
        t_feats: List[torch.Tensor],
        s_embeds: List[torch.Tensor],
        t_embeds: List[torch.Tensor],
    ) -> None:
        """
        최초 1회, student → teacher 차원으로 projection 하는 모듈 생성.
        - feature: 1x1 Conv2d
        - embed: Linear
        """
        device = s_feats[0].device

        # feature adapters
        feat_adapters: List[nn.Module] = []
        for fs, ft in zip(s_feats, t_feats):
            c_s = fs.shape[1]
            c_t = ft.shape[1]
            if c_s == c_t:
                feat_adapters.append(nn.Identity())
            else:
                conv = nn.Conv2d(c_s, c_t, kernel_size=1, bias=False)
                conv.to(device)
                feat_adapters.append(conv)
        self.feat_adapters = nn.ModuleList(feat_adapters)

        # embedding adapters
        embed_adapters: List[nn.Module] = []
        if len(s_embeds) > 0 and len(t_embeds) > 0:
            for se, te in zip(s_embeds, t_embeds):
                d_s = se.shape[-1]
                d_t = te.shape[-1]
                if d_s == d_t:
                    embed_adapters.append(nn.Identity())
                else:
                    lin = nn.Linear(d_s, d_t, bias=False)
                    lin.to(device)
                    embed_adapters.append(lin)
        self.embed_adapters = nn.ModuleList(embed_adapters)

    # ----------------- public entry -----------------
    def compute_losses(
            self,
            imgs: TensorOrList,
            masks: torch.Tensor,
            device: torch.device | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        - imgs: (B,3,H,W) tensor or [tensor, ...]
        - masks: (B,H,W) or (B,1,H,W)
        - device: optional. 주어지면 imgs/masks를 해당 device로 이동.
        """

        # LR/LR: list/tuple이면 첫 번째 텐서만 사용
        if isinstance(imgs, (list, tuple)):
            imgs = imgs[0]

        if device is not None:
            imgs = imgs.to(device)
            masks = masks.to(device)

        if imgs.dim() != 4:
            raise RuntimeError(f"imgs must be 4D tensor, got {imgs.shape}")

        if masks.dim() == 4:
            masks = masks.squeeze(1)

        return self._compute_all_losses(imgs, masks)

    # ----------------- internal helpers -----------------
    @torch.no_grad()
    def _forward_teacher(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        teacher forward (gradient off)
        """
        out = self.teacher(x, return_feats=True, return_embeds=True)

        # SegFormerWrapper.forward 반환 형태: (logits, feats) 또는 (logits, feats, embeds)
        if isinstance(out, tuple) and len(out) == 3:
            logits, feats, embeds = out
        elif isinstance(out, tuple) and len(out) == 2:
            logits, feats = out
            embeds = []
        else:
            raise RuntimeError(
                f"Unexpected teacher output format: type={type(out)}, len={len(out) if isinstance(out, tuple) else 'NA'}"
            )
        return logits, list(feats), list(embeds)

    def _forward_student(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        student forward (gradient on)
        """
        out = self.student(x, return_feats=True, return_embeds=True)

        if isinstance(out, tuple) and len(out) == 3:
            logits, feats, embeds = out
        elif isinstance(out, tuple) and len(out) == 2:
            logits, feats = out
            embeds = []
        else:
            raise RuntimeError(
                f"Unexpected student output format: type={type(out)}, len={len(out) if isinstance(out, tuple) else 'NA'}"
            )
        return logits, list(feats), list(embeds)

    # ----------------- core loss computation -----------------
    def _compute_all_losses(
        self,
        imgs: torch.Tensor,
        masks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        device = imgs.device

        # 1) teacher / student forward
        with torch.no_grad():
            _, t_feats, t_embeds = self._forward_teacher(imgs)

        s_logits, s_feats, s_embeds = self._forward_student(imgs)

        # 1-1) 어댑터가 아직 없으면 현재 feature/embedding shape로 생성
        if self.feat_adapters is None or self.embed_adapters is None:
            self._build_adapters(s_feats, t_feats, s_embeds, t_embeds)

        # 1-2) student feature/embedding을 teacher 차원으로 projection
        proj_s_feats: List[torch.Tensor] = []
        for adapter, fs in zip(self.feat_adapters, s_feats):
            proj_s_feats.append(adapter(fs))

        proj_s_embeds: List[torch.Tensor] = []
        if len(s_embeds) > 0 and len(t_embeds) > 0 and len(self.embed_adapters) > 0:
            num_levels = min(len(s_embeds), len(t_embeds), len(self.embed_adapters))
            for i in range(num_levels):
                se = s_embeds[i]          # (B,N,D_s)
                te = t_embeds[i]          # (B,N,D_t)
                adapter = self.embed_adapters[i]
                if isinstance(adapter, nn.Identity):
                    proj_s_embeds.append(se)
                else:
                    b, n, d_s = se.shape
                    se_flat = se.reshape(b * n, d_s)
                    se_proj_flat = adapter(se_flat)      # (B*N, D_t)
                    se_proj = se_proj_flat.reshape(b, n, -1)
                    proj_s_embeds.append(se_proj)
        else:
            proj_s_embeds = s_embeds  # 어댑터 없으면 그대로 (차원 같을 때)

        # 2) CE loss
        loss_ce = F.cross_entropy(
            s_logits,
            masks,
            ignore_index=self.ignore_index,
        )

        # 3) HCL (student proj vs teacher)
        loss_hcl = torch.tensor(0.0, device=device)
        if self.w_hcl > 0.0:
            loss_hcl = hcl(proj_s_feats, t_feats)

        # 4) CWD (student proj vs teacher)
        loss_cwd = torch.tensor(0.0, device=device)
        if self.w_cwd > 0.0:
            for fs, ft in zip(proj_s_feats, t_feats):
                loss_cwd = loss_cwd + self.criterion_cwd(fs, ft)

        # 5) Embedding KD (proj_s_embeds vs t_embeds)
        loss_embed = torch.tensor(0.0, device=device)
        if self.w_embed > 0.0 and len(proj_s_embeds) > 0 and len(t_embeds) > 0:
            num_levels = min(len(proj_s_embeds), len(t_embeds))
            if self.embed_mode == "mse":
                for i in range(num_levels):
                    se = proj_s_embeds[i]          # (B,N,D_t)
                    te = t_embeds[i].detach()      # (B,N,D_t)
                    loss_embed = loss_embed + self.mse_embed(se, te)
            elif self.embed_mode == "cwd":
                for i in range(num_levels):
                    se = proj_s_embeds[i]          # (B,N,D_t)
                    te = t_embeds[i].detach()      # (B,N,D_t)
                    loss_embed = loss_embed + self.criterion_embed_cwd(se, te)
            else:
                raise ValueError(f"Unsupported embed_mode: {self.embed_mode}")

        total_loss = (
            self.w_ce * loss_ce
            + self.w_hcl * loss_hcl
            + self.w_cwd * loss_cwd
            + self.w_embed * loss_embed
        )

        return {
            "total": total_loss,
            "loss_ce": loss_ce.detach(),
            "loss_hcl": loss_hcl.detach(),
            "loss_cwd": loss_cwd.detach(),
            "loss_embed": loss_embed.detach(),
        }

