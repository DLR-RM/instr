"""
Base layers.
"""

import torch.nn as nn


class Conv2dNormActiv(nn.Module):
    """
    Module for one Conv2d + an Activation (e.g. ReLU, leakyReLU)
    Assumption: odd kernel_size
    """

    def __init__(self, in_ch, out_ch, k_size=3, stride=1, norm=nn.GroupNorm, activation=nn.ReLU, dilation=1, padding=None):
        super(Conv2dNormActiv, self).__init__()

        if k_size % 2 == 0:
            raise ValueError('k_size has to be odd')

        if not padding:
            padding=k_size//2
            if k_size < 2:
                padding = 0 

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k_size, stride=stride, padding=padding, dilation=dilation)

        if norm is None:
            self.norm = norm
        elif norm is nn.BatchNorm2d:
            self.norm = norm(out_ch)
        elif norm is nn.GroupNorm:
            # note: hard coded number of channels per group as 1/4th of total channels
            # doesn't make a huge difference according to https://arxiv.org/pdf/1803.08494.pdf
            self.norm = norm(num_groups=int(out_ch / 4), num_channels=out_ch)
        else:
            raise NotImplementedError

        self.activ = activation(inplace=True) if activation is not None else None

        # for __str__
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x):
        if type(x) is tuple and len(x) == 1:
            x = x[0]
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activ is not None:
            x = self.activ(x)
        return x

    def __str__(self):
        norm_str = "None" if self.norm is None else str(self.norm)
        if type(self.norm) == nn.BatchNorm2d:
            norm_str = "BatchNorm2d"
        elif type(self.norm) == nn.GroupNorm:
            norm_str = "GroupNorm"
        act_str = "None" if self.activ is None else str(self.activ)
        if type(self.activ) == nn.ReLU:
            act_str = "ReLU"
        elif type(self.activ) == nn.LeakyReLU:
            act_str = "LeakyReLU"
        return f"Conv: ({self.in_ch}, {self.out_ch}), Norm: {norm_str}, Act: {act_str}"
