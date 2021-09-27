"""
Simple test case for testing the grid sampling in the subpixel correlation layer compared to conventional shifting.
"""

from unittest import TestCase

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.subpixel_corr import SubpixelCorrelation


class TestSubpixelCorrelation(TestCase):
    def test_compare_with_standard_corr_nearest(self):
        b, c, h, w = 2, 32, 60, 80
        d_max = 10
        c_c = 10
        corr_layer = CorrelationLayer(max_displacement=d_max)
        scorr_layer = SubpixelCorrelation(d_max=d_max, c_c=c_c+1, height=h, width=w, mode='nearest')
        t1, t2 = torch.randn(b, c, h, w).cuda(), torch.randn(b, c, h, w).cuda()

        corr = corr_layer((t1, t2))
        scorr = scorr_layer(t1, t2)

        # corr computes from -displacement until 0, scorr the other way round
        corr = corr.flip(dims=[1])

        self.assertTrue(torch.all(torch.eq(corr, scorr)))

    def test_compare_with_standard_corr_bilinear(self):
        b, c, h, w = 2, 32, 60, 80
        max_displacement = 10
        c_c = 10
        corr_layer = CorrelationLayer(max_displacement=max_displacement)
        scorr_layer = SubpixelCorrelation(d_max=max_displacement, c_c=c_c+1, height=h, width=w, mode='bilinear')
        t1, t2 = torch.randn(b, c, h, w).cuda(), torch.randn(b, c, h, w).cuda()

        corr = corr_layer((t1, t2))
        scorr = scorr_layer(t1, t2)

        # corr computes from -displacement until 0, scorr the other way round
        corr = corr.flip(dims=[1])

        # mean
        mean = torch.abs(corr - scorr).mean()

        self.assertTrue(mean < 1e-4)


class CorrelationLayer(nn.Module):
    """
    This is an earlier implementation of the correlation layer without subpixel sampling.
    It is used here as testing ground truth.
    """
    def __init__(self, max_displacement=0):
        """
        Args:
            max_displacement (integer): maximum displacement to compute the correlation for
        """
        super(CorrelationLayer, self).__init__()
        self.max_displacement = max_displacement

    def forward(self, x):
        """
        Computes local horziontal correlation between two images.

        Args:
            x (tuple): tuple of TWO [4D tensor]; x = (a, b); each of them is either an image or a batch of images

        Returns:
            torch.tensor: correlation map between a and b.
        """

        a, b = x

        zero_tens = torch.empty(a.shape[0], self.max_displacement + 1, a.shape[-2], a.shape[-1]).to(a.device)

        ctr = 0
        step_size = 1
        for i in range(-self.max_displacement * step_size, 1, step_size):
            # horizontal slice from -max_disp up to .shape[-1]
            horizontal_slice = b[..., :b.shape[-1] + i]
            shifted = F.pad(horizontal_slice, (-i, 0), 'constant', 0)
            zero_tens[:, ctr:ctr + 1, :, :] = torch.sum(torch.mul(shifted, a), dim=1).unsqueeze(1)
            ctr += 1

        return zero_tens
