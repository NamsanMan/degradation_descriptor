# kd_engines/transkd_ops.py
# Ported from TransKD train/CSF.py (ops part)
# - hcl
# - CriterionCWD / CriterionEmbedCWD
# - ChannelNorm / EmbedChannelNorm
#
# Inputs:
#   feature: (N,C,H,W)
#   embed:   (N,Tokens,C)  (HF patch embedding hook output)
#
# NOTE: This is intended to match the provided CSF.py behavior as closely as possible.

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def hcl(fstudent, fteacher):
    """
    Hierarchical Context Loss (exactly as original CSF.py).
    - fstudent, fteacher: iterables of feature maps (N,C,H,W), same length.
    - IMPORTANT: channels/spatial must be aligned before calling (SKF should do that).
    """
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n, c, h, w = fs.shape
        loss = F.mse_loss(fs, ft, reduction='mean')
        cnt = 1.0
        tot = 1.0
        for l in [4, 2, 1]:
            if l >= h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l, l))
            tmpft = F.adaptive_avg_pool2d(ft, (l, l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all


class ChannelNorm(nn.Module):
    """
    Original CSF.py ChannelNorm:
      featmap (N,C,H,W) -> reshape (N,C,H*W) -> softmax over last dim (spatial).
    This makes each channel a distribution over spatial positions.
    """
    def __init__(self):
        super(ChannelNorm, self).__init__()

    def forward(self, featmap):
        n, c, h, w = featmap.shape
        featmap = featmap.reshape((n, c, -1))
        featmap = featmap.softmax(dim=-1)
        return featmap


class CriterionCWD(nn.Module):
    """
    Original CSF.py CriterionCWD:
      - norm_type: 'channel', 'spatial', 'channel_mean', or 'none'
      - divergence: 'mse' or 'kl'
      - temperature used only for 'kl' case (and still multiplied by T^2 at end)

    Behavior matches your pasted code:
      if norm_type=='channel': normalize = ChannelNorm()  (softmax over spatial)
      if norm_type=='spatial': normalize = Softmax(dim=1)  (softmax over channel)
      if norm_type=='channel_mean': normalize = mean over spatial -> (N,C)
    """
    def __init__(self, norm_type='none', divergence='mse', temperature=1.0):
        super(CriterionCWD, self).__init__()

        if norm_type == 'channel':
            self.normalize = ChannelNorm()
        elif norm_type == 'spatial':
            self.normalize = nn.Softmax(dim=1)
        elif norm_type == 'channel_mean':
            self.normalize = lambda x: x.view(x.size(0), x.size(1), -1).mean(-1)
        else:
            self.normalize = None
        self.norm_type = norm_type

        self.temperature = 1.0

        if divergence == 'mse':
            self.criterion = nn.MSELoss(reduction='sum')
        elif divergence == 'kl':
            self.criterion = nn.KLDivLoss(reduction='sum')
            self.temperature = float(temperature)
        else:
            raise ValueError(f"Unsupported divergence: {divergence}")
        self.divergence = divergence

    def forward(self, preds_S, preds_T):
        n, c, h, w = preds_S.shape

        if self.normalize is not None:
            norm_s = self.normalize(preds_S / self.temperature)
            norm_t = self.normalize(preds_T.detach() / self.temperature)
        else:
            # original code's else branch is strange (preds_S[0]) but keep behavior
            norm_s = preds_S[0]
            norm_t = preds_T[0].detach()

        if self.divergence == 'kl':
            norm_s = norm_s.log()

        loss = self.criterion(norm_s, norm_t)

        if self.norm_type == 'channel' or self.norm_type == 'channel_mean':
            loss /= n * c
        else:
            loss /= n * h * w

        return loss * (self.temperature ** 2)


# ---------- Embed KD CWD (from original CSF.py) ----------

class EmbedChannelNorm(nn.Module):
    """
    Original CSF.py EmbedChannelNorm:
      embed (N,C,T) or (N,?,?) -> softmax over last dim.
    In original they do: embed_S = embed_S.transpose(1,2) => (N,C,T)
    then softmax(dim=-1).
    """
    def __init__(self):
        super(EmbedChannelNorm, self).__init__()

    def forward(self, embed):
        embed = embed.softmax(dim=-1)
        return embed


class CriterionEmbedCWD(nn.Module):
    """
    Original CSF.py CriterionEmbedCWD.
    - Expects embed_S, embed_T: (N,T,C) then transposes to (N,C,T).
    """
    def __init__(self, norm_type='none', divergence='mse', temperature=1.0):
        super(CriterionEmbedCWD, self).__init__()

        if norm_type == 'channel':
            self.normalize = EmbedChannelNorm()
        elif norm_type == 'spatial':
            self.normalize = nn.Softmax(dim=1)
        elif norm_type == 'channel_mean':
            self.normalize = lambda x: x.view(x.size(0), x.size(1), -1).mean(-1)
        else:
            self.normalize = None
        self.norm_type = norm_type

        self.temperature = 1.0

        if divergence == 'mse':
            self.criterion = nn.MSELoss(reduction='sum')
        elif divergence == 'kl':
            self.criterion = nn.KLDivLoss(reduction='sum')
            self.temperature = float(temperature)
        else:
            raise ValueError(f"Unsupported divergence: {divergence}")
        self.divergence = divergence

    def forward(self, embed_S, embed_T):
        embed_S = embed_S.transpose(1, 2).contiguous()
        embed_T = embed_T.transpose(1, 2).contiguous()
        n, c, _ = embed_S.shape

        if self.normalize is not None:
            norm_s = self.normalize(embed_S / self.temperature)
            norm_t = self.normalize(embed_T.detach() / self.temperature)
        else:
            norm_s = embed_S[0]
            norm_t = embed_T[0].detach()

        if self.divergence == 'kl':
            norm_s = norm_s.log()

        loss = self.criterion(norm_s, norm_t)

        if self.norm_type == 'channel' or self.norm_type == 'channel_mean':
            loss /= n * c

        return loss * (self.temperature ** 2)


def hcl_feaw(fstudent, fteacher):
    """
    Original CSF.py's stage-weighted HCL variant.
    Not used by train_TransKDBase.py pasted, but included for completeness.
    """
    loss_all = 0.0
    fea_weights = [0.1, 0.1, 0.5, 1]
    for fs, ft, fea_w in zip(fstudent, fteacher, fea_weights):
        n, c, h, w = fs.shape
        loss = F.mse_loss(fs, ft, reduction='mean')
        cnt = 1.0
        tot = 1.0
        for l in [4, 2, 1]:
            if l >= h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l, l))
            tmpft = F.adaptive_avg_pool2d(ft, (l, l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + fea_w * loss
    return loss_all
