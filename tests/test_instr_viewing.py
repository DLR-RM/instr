"""
Simple test case for testing permutations in the INSTR network.
"""

from unittest import TestCase

import torch


class TestINSTRViewing(TestCase):
    def test_viewing(self):
        d = 6  # num_decoder_layers
        b = 2  # bsize
        q = 20  # num_queries
        h = 15  # height
        w = 20  # width
        f = 256  # feature channel
        hs = torch.arange(d*b*q*h*w).view(b, d, q, h*w)  # num_dec_layers, bsize, num_queries, hw
        enc = torch.arange(b*f*h*w).view(b, f, h, w)  # bsize, feats, h, w

        # view for multiplication
        hs_v = hs.unsqueeze(-2).flatten(0, 2).view(-1, 1, h, w)  # b*d*q, 1, h, w
        enc_v = enc.repeat_interleave(repeats=d*q, dim=0)  # b*d*q, f, h, w

        # multiply
        res = enc_v * hs_v

        # then: view for interpolation
        res = res.view(b, d, q, f, h, w)

        # alternative version: don't flatten / repeat
        hs = torch.arange(b * d * q * h * w).view(b, d, q, h, w)  # num_dec_layers, bsize, num_queries, h, w
        enc = torch.arange(b * f * h * w).view(b, 1, 1, f, h, w)  # bsize, feats, h, w

        # view for multiplication
        hs_v = hs.unsqueeze(3)
        enc_v = enc.repeat(1, d, q, 1, 1, 1)  # d*b*q, f, h, w

        # multiply
        res_alt = enc_v * hs_v

        # check
        self.assertTrue(torch.all(torch.eq(res, res_alt)))
