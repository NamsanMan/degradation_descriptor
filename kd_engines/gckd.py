# kd_engines/gckd.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .base_engine import BaseKDEngine


# 간단한 LayerNorm2d (채널축 정규화; CNN용 대체)
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        # 채널 전체를 하나의 그룹으로 두는 GroupNorm → LayerNorm 대용
        self.ln = nn.GroupNorm(1, num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)


class _AlignBlock(nn.Module):
    """1x1 → LN → GELU → 1x1: Student 도메인을 Teacher 채널로 정렬"""
    def __init__(self, c_in: int, c_mid: int, c_out: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(c_in, c_mid, kernel_size=1, bias=False),
            LayerNorm2d(c_mid),
            nn.GELU(),
            nn.Conv2d(c_mid, c_out, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class GlobalContextKD(BaseKDEngine):
    """
    Global-Context KD:
      v_T = GAP( LPF( F_T^{(stage)} ) )
      v_S = GAP( LPF( Align( F_S^{(stage)} ) ) )
      L_gc = w_rob * || (v_S - mean(v_S)) - (v_T - mean(v_T)) ||_2^2
      w_rob = exp( -beta * || (v_T_clean - mean) - (v_T_deg - mean) ||_2^2 ) / sqrt(C)

    - 훅으로 지정 스테이지 feature를 자동 수집
    - teacher_clean 입력이 없으면 w_rob = 1.0
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        ignore_index: int,
        teacher_stage: int = -1,
        student_stage: int = -1,
        lpf_kernel: int = 3,
        lpf_stride: int = 1,
        lpf_pad: int = 1,
        align_hidden: int = 256,
        w_gc: float = 1.0,
        use_wrob: bool = True,
        beta: float = 0.5,
        freeze_teacher: bool = True,
    ) -> None:
        super().__init__(teacher, student)

        self.ignore_index = int(ignore_index)
        self.teacher_stage = int(teacher_stage)
        self.student_stage = int(student_stage)
        self.lpf_kernel = int(lpf_kernel)
        self.lpf_stride = int(lpf_stride)
        self.lpf_pad = int(lpf_pad)
        self.align_hidden = int(align_hidden)
        self.w_gc = float(w_gc)
        self.use_wrob = bool(use_wrob)
        self.beta = float(beta)
        self.freeze_teacher = bool(freeze_teacher)

        # 훅/피처 버퍼
        self._t_feats: List[torch.Tensor] = []
        self._s_feats: List[torch.Tensor] = []
        self._t_hooks: List = []
        self._s_hooks: List = []

        if self.freeze_teacher:
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()

        # Lazy 초기화 정렬 블록
        self._align: Optional[nn.Module] = None

    # --------- 공용 유틸 ----------
    @staticmethod
    def _lpf2d(x: torch.Tensor, k: int, s: int, p: int) -> torch.Tensor:
        """평균풀링 기반 저역통과(anti-alias)."""
        return F.avg_pool2d(x, kernel_size=k, stride=s, padding=p)

    @staticmethod
    def _gap(x: torch.Tensor) -> torch.Tensor:
        """(N,C,H,W) → (N,C)"""
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    @staticmethod
    def _mean_only(v: torch.Tensor) -> torch.Tensor:
        """채널 평균만 제거(분산/스케일 의존성 제거)."""
        return v - v.mean(dim=1, keepdim=True)

    @staticmethod
    def _pick_stage(feats: List[torch.Tensor], idx: int) -> torch.Tensor:
        """음수 인덱스 포함 python 스타일 stage 선택."""
        if len(feats) == 0:
            raise RuntimeError("No 4D features captured. Check hook placement.")
        return feats[idx]

    def _register_hooks(self) -> None:
        """범용 훅: 4D feature만 수집."""
        def _mk_cb(store):
            def cb(_m, _i, o):
                if isinstance(o, torch.Tensor) and o.dim() == 4:
                    store.append(o)
            return cb

        def _find_candidates(root: nn.Module):
            # encoder/backbone/features 우선, 없으면 루트
            names = ["encoder", "backbone", "features"]
            for n in names:
                if hasattr(root, n) and isinstance(getattr(root, n), nn.Module):
                    return [getattr(root, n)]
            return [root]

        for m in _find_candidates(self.teacher):
            self._t_hooks.append(m.register_forward_hook(_mk_cb(self._t_feats)))
        for m in _find_candidates(self.student):
            self._s_hooks.append(m.register_forward_hook(_mk_cb(self._s_feats)))

    def _clear_hooks_and_feats(self) -> None:
        for h in self._t_hooks + self._s_hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._t_hooks.clear()
        self._s_hooks.clear()
        self._t_feats.clear()
        self._s_feats.clear()

    def _lazy_build_align(self, s_feat: torch.Tensor, t_feat: torch.Tensor) -> None:
        if self._align is None:
            c_in = s_feat.shape[1]
            c_mid = self.align_hidden
            c_out = t_feat.shape[1]
            self._align = _AlignBlock(c_in, c_mid, c_out).to(s_feat.device)

    def get_extra_parameters(self):
        return list(self._align.parameters()) if self._align is not None else []

    # ---------------- 학습 루프에서 호출 ----------------
    def compute_losses(self, imgs, masks, device):
        """
        imgs:
          - torch.Tensor: 열화(deg) 이미지만 포함
          - dict: {'deg': Tensor, 't_clean': Tensor(optional)}
        """
        if isinstance(imgs, dict):
            x_deg = imgs["deg"]
            x_clean = imgs.get("t_clean", None)
        else:
            x_deg = imgs
            x_clean = None

        x_deg = x_deg.to(device, non_blocking=True)
        if x_clean is not None:
            x_clean = x_clean.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        self._clear_hooks_and_feats()
        self._register_hooks()

        # ----- forward -----
        with torch.no_grad() if self.freeze_teacher else torch.enable_grad():
            _ = self.teacher(x_deg)  # deg용 feature 수집
            if x_clean is not None and self.use_wrob:
                _ = self.teacher(x_clean)  # clean feature 수집
                t_feats_clean = list(self._t_feats)  # 복사
                self._t_feats.clear()
                _ = self.teacher(x_deg)  # deg feature 다시 수집

        s_logits = self.student(x_deg)

        # ----- feature 선택 -----
        t_feat = self._pick_stage(self._t_feats, self.teacher_stage)
        s_feat = self._pick_stage(self._s_feats, self.student_stage)
        if x_clean is not None and self.use_wrob:
            t_feat_clean = self._pick_stage(t_feats_clean, self.teacher_stage)
        self._clear_hooks_and_feats()

        # ----- 도메인 정렬 + LPF/GAP → v_T, v_S -----
        self._lazy_build_align(s_feat, t_feat)
        s_aligned = self._align(s_feat)

        t_lpf = self._lpf2d(t_feat, self.lpf_kernel, self.lpf_stride, self.lpf_pad)
        s_lpf = self._lpf2d(s_aligned, self.lpf_kernel, self.lpf_stride, self.lpf_pad)

        v_T = self._gap(t_lpf)  # (N,Ct)
        v_S = self._gap(s_lpf)  # (N,Ct)  <-- 정렬로 채널 일치

        vT_hat = self._mean_only(v_T)
        vS_hat = self._mean_only(v_S)

        # ----- w_rob 계산 -----
        if (x_clean is not None) and self.use_wrob:
            v_Tc = self._gap(self._lpf2d(t_feat_clean, self.lpf_kernel, self.lpf_stride, self.lpf_pad))
            vTc_hat = self._mean_only(v_Tc)
            diff_tc = (vTc_hat - vT_hat)                 # (N,C)
            # 배치별 스칼라 가중치
            w_rob = torch.exp(-self.beta * diff_tc.pow(2).sum(dim=1)) / (v_T.shape[1] ** 0.5)
        else:
            w_rob = torch.ones(v_T.shape[0], device=v_T.device)

        # ----- L_GC -----
        diff_ts = (vS_hat - vT_hat)                      # (N,C)
        l_gc_per = diff_ts.pow(2).sum(dim=1) * w_rob     # (N,)
        l_gc = l_gc_per.mean()

        # ----- CE(student) -----
        ce_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        ce_student = ce_fn(s_logits, masks)

        total = self.w_gc * l_gc + ce_student
        return {
            "total": total,
            "ce_student": ce_student.detach(),
            "kd_gc": (self.w_gc * l_gc).detach(),
        }
