# kd_engines/segtoseg_kd.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_engine import BaseKDEngine


def _conv1x1(in_ch, out_ch):
    if in_ch == out_ch:
        return nn.Identity()
    m = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    return m


class SegToSegKD(BaseKDEngine):
    """
    - teacher, student: SegFormer 래퍼 (forward(..., return_feats=True) 지원)
    - compute_losses(imgs, masks, device) -> dict(loss 항목들)
    - 특징: stage별 student->teacher 채널 정합(1x1 conv) + 공간 크기 불일치시 bilinear 보간
    """
    def __init__(self, teacher, student,
                 stage_weights, t: float,
                 w_ce_student: float, w_ce_teacher: float,
                 w_logit: float, w_feat: float,
                 ignore_index: int,
                 use_logit_kd: bool = True,
                 feat_l2_normalize: bool = True,
                 freeze_teacher: bool = False):
        super().__init__(teacher, student)

        assert len(stage_weights) == 4, "SegFormer 인코더는 4개 stage를 가정합니다."
        self.stage_weights = stage_weights
        self.t = float(t)
        self.w_ce_student = float(w_ce_student)
        self.w_ce_teacher = float(w_ce_teacher)
        self.w_logit = float(w_logit)
        self.w_feat = float(w_feat)
        self.ignore_index = int(ignore_index)
        self.use_logit_kd = bool(use_logit_kd)
        self.feat_l2_normalize = bool(feat_l2_normalize)
        self._freeze_teacher = bool(freeze_teacher)

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        # 교사 고정
        if self._freeze_teacher:
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()

        # stage별 student->teacher 채널 정합 projection (lazy build)
        self._proj_built = False
        self.proj_s2t = nn.ModuleList([nn.Identity() for _ in range(4)])

    def train(self, mode: bool = True):
        super().train(mode)  # 학생은 mode에 따름
        if self._freeze_teacher:
            self.teacher.eval()  # 교사는 항상 eval
        return self

    # 첫 forward 때 실제 채널 크기에 맞춰 1x1 conv 생성
    def _build_projs_if_needed(self, s_feats, t_feats):
        if self._proj_built:
            return
        device = s_feats[0].device  # ← 현재 feature의 디바이스(GPU)를 기준으로
        mods = []
        for sf, tf in zip(s_feats, t_feats):
            in_c, out_c = sf.shape[1], tf.shape[1]
            if in_c == out_c:
                m = nn.Identity()
            else:
                m = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            mods.append(m.to(device))  # ← 여기에서 디바이스 이동
        self.proj_s2t = nn.ModuleList(mods)
        self._proj_built = True

    def _forward_with_feats(self, model, imgs):
        """
        (logits, feats) 튜플을 반환.
        1) 모델이 return_feats=True를 지원하면 그대로 사용
        2) 안되면 HuggingFace SegFormer의 hidden_states/encoder_hidden_states를 켜서 마지막 4개 stage를 feats로 사용
        """
        # 1) return_feats 지원 경로 (래퍼가 이미 구현된 경우)
        try:
            out = model(imgs, return_feats=True)
            if isinstance(out, tuple) and len(out) == 2:
                return out[0], out[1]
            if isinstance(out, dict) and "logits" in out and "feats" in out:
                return out["logits"], out["feats"]
        except TypeError:
            pass  # 지원 안 하면 2)로 폴백

        # 2) HF SegFormer 경로
        if hasattr(model, "config"):
            if hasattr(model.config, "output_hidden_states") and not model.config.output_hidden_states:
                model.config.output_hidden_states = True  # 한 번만 켜두면 됨

            outputs = model(imgs)  # return_dict=True 기본
            logits = getattr(outputs, "logits", outputs[0] if isinstance(outputs, tuple) else None)

            feats = getattr(outputs, "encoder_hidden_states", None)
            if feats is None:
                feats = getattr(outputs, "hidden_states", None)
            if feats is None or len(feats) < 4:
                raise RuntimeError("hidden_states/encoder_hidden_states를 얻지 못함. "
                                   "모델 래퍼가 (logits, feats)를 직접 반환하도록 하거나 hidden_states를 켜세요.")
            feats = tuple(feats[-4:])  # 마지막 4개 stage
            return logits, feats

        # 둘 다 안 되면 명시적으로 실패
        raise RuntimeError("모델이 return_feats도, hidden_states도 지원하지 않습니다.")

    def _logit_kd_loss(self, s_logits, t_logits, masks):
        """
        KLDiv( log_softmax(S/T), softmax(T/T) ) * T^2, void 무시
        s_logits, t_logits: (B, C, H, W)
        masks: (B, H, W) with ignore_index
        """
        if (not self.use_logit_kd) or (self.w_logit <= 0.0):
            return s_logits.new_tensor(0.0)

        # KD는 FP32에서 수행 (AMP라도 언더/오버플로 방지)
        s_logits = s_logits.float()
        t_logits = t_logits.float()

        T = self.t
        B, C, H, W = s_logits.shape

        s = F.log_softmax(s_logits / T, dim=1)      # (B,C,H,W)
        t = F.softmax(t_logits.detach() / T, dim=1) # (B,C,H,W)

        s = s.permute(0, 2, 3, 1).reshape(-1, C)  # (BHW, C)
        t = t.permute(0, 2, 3, 1).reshape(-1, C)
        valid = (masks.view(-1) != self.ignore_index)

        if valid.any():
            s = s[valid]
            t = t[valid]
            kd = F.kl_div(s, t, reduction='batchmean') * (T * T)
        else:
            kd = s_logits.new_tensor(0.0)
        return kd

    def _feat_kd_loss(self, s_feats, t_feats, masks):
        """
        스테이지별 feature KD (student -> teacher 채널 정합)
        s_feats/t_feats: tuple of 4 tensors, each (B, C_i, H_i, W_i)
        """
        if self.w_feat <= 0.0:
            return s_feats[0].new_tensor(0.0)

        # FP32 고정
        s_feats = tuple(f.float() for f in s_feats)
        t_feats = tuple(f.float() for f in t_feats)

        # 첫 호출 시 projection 모듈 생성
        self._build_projs_if_needed(s_feats, t_feats)

        total = s_feats[0].new_tensor(0.0)
        for i, (ws, sf, tf, proj) in enumerate(zip(self.stage_weights, s_feats, t_feats, self.proj_s2t)):
            if ws <= 0.0:
                continue

            # 채널 정합: student -> teacher
            proj = proj.to(sf.device)
            sf = proj(sf)

            # 공간 크기 정합
            if sf.shape[-2:] != tf.shape[-2:]:
                sf = F.interpolate(sf, size=tf.shape[-2:], mode='bilinear', align_corners=False)

            # void 마스크 다운샘플 → 유효 위치만 평균
            with torch.no_grad():
                valid = (F.interpolate(
                    (masks.unsqueeze(1).float() != self.ignore_index).float(),
                    size=tf.shape[-2:], mode='nearest'
                ).squeeze(1).bool())

            s_use = sf
            t_use = tf.detach()

            if self.feat_l2_normalize:
                s_use = F.normalize(s_use, p=2, dim=1)  # 채널 정규화
                t_use = F.normalize(t_use, p=2, dim=1)

            diff2 = (s_use - t_use) ** 2  # (B,C,H,W)

            denom = valid.sum() * diff2.shape[1] + 1e-6
            if denom.item() == 0:
                # 전 픽셀이 ignore인 극단적 케이스 보호
                stage_loss = diff2.new_tensor(0.0)
            else:
                stage_loss = (diff2 * valid.unsqueeze(1)).sum() / denom

            total = total + ws * stage_loss
        return total

    def compute_losses(self, imgs, masks, device):
        # forward with features
        s_logits, s_feats = self._forward_with_feats(self.student, imgs)

        if self._freeze_teacher:
            with torch.no_grad():
                t_logits, t_feats = self._forward_with_feats(self.teacher, imgs)
        else:
            t_logits, t_feats = self._forward_with_feats(self.teacher, imgs)

        # CE
        ce_student = self.ce_loss(s_logits, masks)
        ce_teacher = (
            self.ce_loss(t_logits, masks)
            if (self.w_ce_teacher > 0.0) and (not self._freeze_teacher)
            else s_logits.new_tensor(0.0)
        )

        # KD
        kd_logit = self._logit_kd_loss(s_logits, t_logits, masks) if self.use_logit_kd else s_logits.new_tensor(0.0)
        kd_feat  = self._feat_kd_loss(s_feats, t_feats, masks)

        total = (self.w_ce_student * ce_student +
                 self.w_ce_teacher * ce_teacher +
                 self.w_logit * kd_logit +
                 self.w_feat  * kd_feat)

        return {
            "total": total,
            "ce_student": ce_student.detach(),
            "ce_teacher": ce_teacher.detach(),
            "kd_logit": kd_logit.detach(),
            "kd_feat": kd_feat.detach(),
            "s_logits": s_logits  # 평가용
        }

    # 옵티마이저에 포함되도록 노출 (projection 학습 포함)
    def get_extra_parameters(self):
        return list(self.proj_s2t.parameters()) if self._proj_built else []
