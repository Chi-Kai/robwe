import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

class PassportPrivateBlock(nn.Module):
    def __init__(self, i, o, ks=3, s=1, pd=1):
        super().__init__()

        self.conv = nn.Conv2d(i, o, ks, s, pd, bias=False)
        self.weight = self.conv.weight

        self.init_scale(True)
        self.init_bias(True)
        self.bn = nn.BatchNorm2d(o, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.reset_parameters()
    # 这里的bias 和 scale 会参与梯度更新
    def init_bias(self, force_init=False):
        if force_init:
            self.bias = nn.Parameter(torch.Tensor(self.conv.out_channels).to(self.weight.device))
            init.zeros_(self.bias)
        else:
            self.bias = None
    
    def init_scale(self, force_init=False):
        if force_init:
            self.scale = nn.Parameter(torch.Tensor(self.conv.out_channels).to(self.weight.device))
            init.ones_(self.scale)
        else:
            self.scale = None

    def reset_parameters(self):
        init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        #对x进行缩放和偏置
        x = x * self.scale [None, :, None, None] + self.bias [None, :, None, None]
        x = self.relu(x)
        return x