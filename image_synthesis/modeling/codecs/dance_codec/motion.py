import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from librosa.filters import mel as librosa_mel_fn
from torch.nn.utils import weight_norm
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

def WNConv2d(*arg, **kwargs):
	return weight_norm(nn.Conv2d(*arg, **kwargs))

def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))

def WNConvTranspose2d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose2d(*args, **kwargs))


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            WNConv1d(dim, dim, kernel_size=3, dilation=dilation),
            nn.LeakyReLU(0.2),
            WNConv1d(dim, dim, kernel_size=1),
        )
        self.shortcut = WNConv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class MotionEncoder(nn.Module):
    def __init__(self, context_length):
        super().__init__()
        self.lin1 = nn.Linear(219, 800)
        # self.lin2 = nn.Linear(789, 1024)
        self.lin2 = nn.Linear(786, 1024)
        model = [
            nn.Conv1d(60*context_length, 512, kernel_size=6),
            nn.LeakyReLU(0.2),
                ]
        model += [ResnetBlock(512, dilation=3**0)]
        model += [ResnetBlock(512, dilation=3**1)]
        model += [ResnetBlock(512, dilation=3**2)]
        model += [ResnetBlock(512, dilation=3**3)]
        model += [
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 512, kernel_size=4),
                ]
        model += [ResnetBlock(512, dilation=3**0)]
        model += [ResnetBlock(512, dilation=3**1)]
        model += [ResnetBlock(512, dilation=3**2)]
        model += [ResnetBlock(512, dilation=3**3)]
        model += [
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 1024, kernel_size=4),
                ]
        model += [ResnetBlock(1024, dilation=3**0)]
        model += [ResnetBlock(1024, dilation=3**1)]
        model += [ResnetBlock(1024, dilation=3**2)]
        model += [ResnetBlock(1024, dilation=3**3)]
        model += [
            nn.LeakyReLU(0.2),
            nn.Conv1d(1024, 1, kernel_size=4),
                ]
        self.model = nn.Sequential(*model)
        self.apply(weights_init)

    def forward(self, x):
        x = self.lin1(x)
        out = self.model(x)
        # print("check out size:", out.size())
        out = self.lin2(out)
        return out