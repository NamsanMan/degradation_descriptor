###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from .base import BaseNet
from .fcn import FCNHead
from ...nn import PyramidPooling

class PSP(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(PSP, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = PSPHead(2048, nclass, norm_layer, self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    # def forward(self, x):
    #     _, _, h, w = x.size()
    #     _, _, c3, c4 = self.base_forward(x)
    #
    #     outputs = []
    #     x = self.head(c4)
    #     x = interpolate(x, (h,w), **self._up_kwargs)
    #     outputs.append(x)
    #     if self.aux:
    #         auxout = self.auxlayer(c3)
    #         auxout = interpolate(auxout, (h,w), **self._up_kwargs)
    #         outputs.append(auxout)
    #     return tuple(outputs)
    def forward(self, x):
        _, _, h, w = x.size()
        _, _, c3, c4 = self.base_forward(x)

        x_main = self.head(c4)
        x_main = interpolate(x_main, (h, w), **self._up_kwargs)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = interpolate(auxout, (h, w), **self._up_kwargs)
            # 보조 출력은 계산하지만 최종 반환에서는 사용하지 않음

        return x_main

class PSPHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(PSPHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = nn.Sequential(PyramidPooling(in_channels, norm_layer, up_kwargs),
                                   nn.Conv2d(in_channels * 2, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(True),
                                   nn.Dropout(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)

def get_psp(dataset='pascal_voc', backbone='resnet50s', pretrained=False,
            root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ...datasets import datasets, acronyms
    model = PSP(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('psp_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model

def get_psp_resnet50_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_psp_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_psp('ade20k', 'resnet50s', pretrained, root=root, **kwargs)

if __name__ == '__main__':
    from torchinfo import summary
    from thop import profile as profile_thop  # https://github.com/Lyken17/pytorch-OpCounter
    from DLCs.FLOPs import profile

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _B, _C, _H, _W = 1, 3, 360, 480

    model = PSP(nclass=12, backbone='resnet50')

    print("\n --- Info ---\n")
    summary(model, input_size=(_B, _C, _H, _W))

    print("\n --- THOP ---\n")
    input = torch.randn(_B, _C, _H, _W).to(device)
    macs, params = profile_thop(model, inputs=(input,))
    _giga = 1000000000
    print("THOP", macs, macs / _giga, "G", params)
    print("Multi-Adds (G):", round(macs / _giga, 4))

    print("\n --- FLOPs ---\n")
    flops, params = profile(model, input_size=(_B, _C, _H, _W))
    print('Input: {} x {}, flops: {:.10f} GFLOPs, params: {}'.format(_H, _W, flops / (1e9), params))
    print("FLOPS (G):", round(flops / (1e9), 4))