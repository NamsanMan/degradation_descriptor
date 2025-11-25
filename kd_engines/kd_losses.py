import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_engine import BaseKDEngine


class BasicLogitKD(BaseKDEngine):
    """
    가장 기본적인 Logit Knowledge Distillation을 수행하는 간단한 KD 엔진입니다.
    [수정] ignore_index를 올바르게 처리하도록 개선되었습니다.
    """

    def __init__(self, teacher, student,
                 w_ce_student: float,
                 w_kd_logit: float,
                 temperature: float,
                 ignore_index: int,
                 freeze_teacher: bool = True):
        super().__init__(teacher, student)

        self.w_ce_student = float(w_ce_student)
        self.w_kd_logit = float(w_kd_logit)
        self.temperature = float(temperature)
        self.ignore_index = int(ignore_index)
        self._freeze_teacher = bool(freeze_teacher)

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        # ✅ 변경점: 픽셀별 손실을 계산하기 위해 reduction='none'으로 설정
        self.kd_loss = nn.KLDivLoss(reduction='none', log_target=False)

        if self._freeze_teacher:
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._freeze_teacher:
            self.teacher.eval()
        return self

    def compute_losses(self, imgs, masks, device):
        s_output = self.student(imgs, return_feats=False)
        s_logits = s_output[0] if isinstance(s_output, tuple) else s_output

        with torch.no_grad():
            t_output = self.teacher(imgs, return_feats=False)
            t_logits = t_output[0] if isinstance(t_output, tuple) else t_output

        # 학생 CE Loss는 ignore_index를 자동으로 처리합니다.
        loss_ce = self.ce_loss(s_logits, masks)

        # ✅ 변경점: KD Loss 계산 시 ignore_index를 마스킹하여 제외

        # 1. 유효한 픽셀 마스크 생성 (ignore_index가 아닌 픽셀만 True)
        # masks의 shape: [B, H, W]
        valid_mask = (masks != self.ignore_index)

        # 2. KLDivLoss 계산 (픽셀 단위)
        # log_softmax 결과 shape: [B, C, H, W]
        log_p_s = F.log_softmax(s_logits / self.temperature, dim=1)
        p_t = F.softmax(t_logits / self.temperature, dim=1)

        # kd_loss 결과 shape: [B, C, H, W]
        pixel_wise_kd_loss = self.kd_loss(log_p_s, p_t)
        # 채널(C) 차원에 대해 합산하여 픽셀별 최종 손실 계산
        # 결과 shape: [B, H, W]
        pixel_wise_kd_loss = pixel_wise_kd_loss.sum(dim=1)

        # 3. 유효한 픽셀에 대해서만 손실 평균 계산
        masked_kd_loss = pixel_wise_kd_loss[valid_mask]

        # 유효한 픽셀이 하나도 없는 경우를 대비 (loss가 NaN이 되는 것 방지)
        if masked_kd_loss.numel() > 0:
            loss_kd = masked_kd_loss.mean()
        else:
            loss_kd = torch.tensor(0.0, device=device)

        # 4. 온도 스케일링 적용
        loss_kd = loss_kd * (self.temperature ** 2)

        total = (self.w_ce_student * loss_ce +
                 self.w_kd_logit * loss_kd)

        return {
            "total": total,
            "ce_student": loss_ce.detach(),
            "kd_logit": loss_kd.detach(),
            "s_logits": s_logits
        }

    def get_extra_parameters(self):
        return []