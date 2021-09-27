"""
SubpixelCorrelation layer and a wrapper with a downsampling conv.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubpixelCorrelationWrapper(nn.Module):
    """
    Wrapper for SubpixelCorrelation layer.
    Adds a downsampling conv.
    """
    def __init__(self, layer='1', mode='bilinear'):
        """
        Args:
            layer (str): after which layer the correlation happens
            mode (str): see torch.nn.functional.grid_sample() for more details
        """
        super().__init__()

        if layer == '1':
            planes = 256
            height, width = 120, 160
            d_max = 64
            c_c = 64
        elif layer == '2':
            planes = 512
            height, width = 60, 80
            d_max = 32
            c_c = 32
        else:
            raise NotImplementedError

        self.down = nn.Conv2d(planes, planes // 8, 1, padding=0, bias=False)
        self.corr = SubpixelCorrelation(d_max=d_max, c_c=c_c, height=height, width=width, mode=mode)

    def forward(self, left, right):
        l = self.down(left)
        r = self.down(right)
        corr = self.corr(l, r)
        return corr


class SubpixelCorrelation(nn.Module):
    """
    Subpixel Correlation layer.
    A right image of a stereo pair is shifted over its left pair to the right until a certain maximum displacement in
    certain intervals. This layer supports float values as displacement and performs respective interpolation. This
    allows to generalize to novel intrinsic parameters during inference.
    For a visual explanation see our paper, Fig. 2 (right).
    """
    def __init__(self, d_max=30, c_c=64, height=120, width=160, device=torch.device('cuda'), mode='bilinear'):
        """
        Args:
            d_max (int): maximum displacement in pixels
            c_c (int): number of displacement steps to perform
            height (int): image height
            width (int): image width
            device (torch.device): device where to perform computation
            mode (string): see torch.nn.functional.grid_sample() for more details
        """
        super().__init__()

        self.height = height
        self.width = width
        self.mode = mode
        self.device = device
        self.c_c = c_c

        self.calculate_grid(d_max=d_max)

    def calculate_grid(self, d_max):
        steps = torch.linspace(0, d_max, self.c_c, dtype=torch.float32, device=self.device) / ((self.width - 1.) / 2)

        mh, mw = torch.meshgrid([torch.arange(0, self.height, dtype=torch.float32, device=self.device),
                                 torch.arange(0, self.width, dtype=torch.float32, device=self.device)])

        x_coords = mw / ((self.width - 1.) / 2.) - 1.
        y_coords = mh / ((self.height - 1.) / 2.) - 1.

        self.grid = torch.stack([x_coords, y_coords], dim=2).unsqueeze(0).repeat(self.c_c, 1, 1, 1)
        self.grid[:, :, :, 0] -= steps.view(-1, 1, 1)
        self.d_max = d_max

    def forward(self, left, right):
        b, _, h, w = left.shape
        right = right.repeat_interleave(repeats=self.c_c, dim=0)
        left = left.repeat_interleave(repeats=self.c_c, dim=0)

        sampled = F.grid_sample(right, self.grid.repeat(b, 1, 1, 1), mode=self.mode, padding_mode='zeros', align_corners=True)
        corr = torch.sum(torch.mul(left, sampled), dim=1)
        corr = corr.view(b, self.c_c, h, w)
        return corr


# simple forward example
if __name__ == '__main__':
    corr = SubpixelCorrelation(d_max=30, c_c=32, height=120, width=160, mode='nearest').cuda()
    tens = torch.randn(2, 64, 120, 160).cuda()
    out = corr(tens, tens)
    print(out.shape)  # should be 2, 32, 120, 160
