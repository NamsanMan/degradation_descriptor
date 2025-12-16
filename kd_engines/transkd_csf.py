# kd_engines/transkd_csf.py
# Ported from TransKD train/CSF.py (core CSF / SKF modules)
# - SK, SKF, build_kd_trans
#
# Requires:
#   student(x, is_feat=True) -> (feats_tuple_len4, logits, embeds_tuple_len4)

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F


class SK(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse, len=32, reduce=16):
        super(SK, self).__init__()
        len = max(mid_channel // reduce, len)
        self.fuse = fuse
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            # SKNet-style selection fusion
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Sequential(
                nn.Conv2d(mid_channel, len, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(len),
                nn.ReLU(inplace=True)
            )
            self.fc1 = nn.Sequential(
                nn.Conv2d(mid_channel, len, kernel_size=1, stride=1, bias=False),
                nn.ReLU(inplace=True)
            )
            self.fcs = nn.ModuleList([])
            for _ in range(2):
                self.fcs.append(nn.Conv2d(len, mid_channel, kernel_size=1, stride=1))
            self.softmax = nn.Softmax(dim=1)

        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)

    def forward(self, x, y=None, shape=None):
        x = self.conv1(x)
        if self.fuse:
            shape = x.shape[-2:]
            b = x.shape[0]
            y = F.interpolate(y, shape, mode="nearest")

            feas_U = [x, y]
            feas_U = torch.stack(feas_U, dim=1)
            attention = torch.sum(feas_U, dim=1)
            attention = self.gap(attention)

            # original conditional branch for b==1
            if b == 1:
                attention = self.fc1(attention)
            else:
                attention = self.fc(attention)

            attention = [fc(attention) for fc in self.fcs]
            attention = torch.stack(attention, dim=1)
            attention = self.softmax(attention)

            x = torch.sum(feas_U * attention, dim=1)

        y = self.conv2(x)
        return y, x


class SKF(nn.Module):
    def __init__(self, student, in_channels, out_channels, mid_channel, embed):
        super(SKF, self).__init__()
        self.student = student

        skfs = nn.ModuleList()
        for idx, in_channel in enumerate(in_channels):
            skfs.append(SK(in_channel, mid_channel, out_channels[idx], idx < len(in_channels) - 1))
        # reverse order (deep->shallow)
        self.skfs = skfs[::-1]

        self.embed = int(embed)

        # Embed projection (Linear) exactly as original
        if self.embed == 5:
            self.embed1_linearproject = nn.Linear(in_channels[0], out_channels[0])
            self.embed2_linearproject = nn.Linear(in_channels[1], out_channels[1])
            self.embed3_linearproject = nn.Linear(in_channels[2], out_channels[2])
            self.embed4_linearproject = nn.Linear(in_channels[3], out_channels[3])
        elif self.embed == 1:
            self.embed1_linearproject = nn.Linear(in_channels[0], out_channels[0])
        elif self.embed == 2:
            self.embed1_linearproject = nn.Linear(in_channels[1], out_channels[1])
        elif self.embed == 3:
            self.embed1_linearproject = nn.Linear(in_channels[2], out_channels[2])
        elif self.embed == 4:
            self.embed1_linearproject = nn.Linear(in_channels[3], out_channels[3])
        elif self.embed == 0:
            pass
        else:
            raise ValueError("the number of embeddings not supported")

    def forward(self, x, is_feat: bool = False):
        """
        To match degradation_descriptor + evaluate.py:
          - if is_feat=False: return logits only (so evaluate_all works)
          - if is_feat=True : return (results, logit, embedproj?) like original

        student(x, is_feat=True) is expected to return:
          (feats_tuple_len4, logits, embeds_tuple_len4)
        """
        if not is_feat:
            return self.student(x)  # logits only

        feats_s, logit, embed = self.student(x, is_feat=True)

        # Original code assumes feats_s is list/tuple and reverses it
        x_rev = list(feats_s)[::-1]

        results: List[torch.Tensor] = []
        embedproj: List[torch.Tensor] = []

        out_features, res_features = self.skfs[0](x_rev[0])
        results.append(out_features)

        for features, skf in zip(x_rev[1:], self.skfs[1:]):
            out_features, res_features = skf(features, res_features)
            results.insert(0, out_features)

        if self.embed == 5:
            embedproj = [*embedproj, self.embed1_linearproject(embed[0])]
            embedproj = [*embedproj, self.embed2_linearproject(embed[1])]
            embedproj = [*embedproj, self.embed3_linearproject(embed[2])]
            embedproj = [*embedproj, self.embed4_linearproject(embed[3])]
            return results, logit, embedproj
        elif self.embed == 0:
            return results, logit
        elif self.embed == 1:
            embedproj = [*embedproj, self.embed1_linearproject(embed[0])]
            return results, logit, embedproj
        elif self.embed == 2:
            embedproj = [*embedproj, self.embed1_linearproject(embed[1])]
            return results, logit, embedproj
        elif self.embed == 3:
            embedproj = [*embedproj, self.embed1_linearproject(embed[2])]
            return results, logit, embedproj
        elif self.embed == 4:
            embedproj = [*embedproj, self.embed1_linearproject(embed[3])]
            return results, logit, embedproj
        else:
            raise ValueError("the number of embeddings not supported")


def build_kd_trans(model, embed, in_channels=[32, 64, 160, 256], out_channels=[64, 128, 320, 512]):
    """
    Exact signature of original CSF.py.

    NOTE (important):
    - This hardcodes in_channels/out_channels (MiT-B0 -> MiT-B2/B3).
    - If your teacher/student are not this pair, this must be changed or inferred dynamically.
    """
    mid_channel = 64
    student = model
    model = SKF(student, in_channels, out_channels, mid_channel, embed)
    return model
