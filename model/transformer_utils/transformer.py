"""
INSTR Transformer class, adapted from https://github.com/facebookresearch/detr.
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""

import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from model.custom_mhatt import CustomMultiheadAttention


class Transformer(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, query_proc="expanded", h=15, w=20):
        """
        Args:
            d_model (int): input feature dimensionality
            nhead (int): number of heads
            num_encoder_layers (int): number of encoder layers
            num_decoder_layers (int): number of decoder layers
            dim_feedforward (int): feedforward dimensionality
            dropout (float): dropout ratio
            activation (string): which activation function to use
            normalize_before (bool): whether to normalize after the layer pass or not
            return_intermediate_dec (bool): whether to return intermediate decoder outputs for auxiliary loss calculation
            query_proc (string): type of query processing (`Query Proc.` in Table 3 of the paper). One of 'expanded', 'att', 'attcat_tfenc', 'attcat_bb'
            h (int): input feature height
            w (int): input feature width
        """
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.h = h
        self.w = w

        if query_proc == "expanded":
            self._decoder_forward = self._expanded_forward
        elif query_proc == "att":
            self._decoder_forward = self._att_forward
        elif query_proc == "attcat_tfenc":
            self._decoder_forward = self._attcat_tfenc_forward
        elif query_proc == "attcat_bb":
            self._decoder_forward = self._attcat_bb_forward
        else:
            raise NotImplementedError

    def _expanded_forward(self, tgt, memory, memory_key_padding_mask, pos, query, **kwargs):
        """
        Performs expanded forward; see Sec. 3B in the paper; `c-att-exp` in Tab. 3 of the paper
        """
        hs, hs_full, hs_att = self.decoder(tgt, memory, memory_key_padding_mask=memory_key_padding_mask,
                                           pos=pos, query_pos=query)
        hs_viz = hs_full[-1].clone().detach().transpose(0, 1).mean(dim=2)
        hs_full = hs_full.permute(2, 0, 1, 3, 4).flatten(0, 2).view(-1, 256, self.h, self.w)
        return hs_full, hs_viz, memory

    def _att_forward(self, tgt, memory, memory_key_padding_mask, pos, query, **kwargs):
        """
        Only returns the attention weight maps; `c-att` in Tab. 3 of the paper
        """
        hs, hs_full, hs_att = self.decoder(tgt, memory, memory_key_padding_mask=memory_key_padding_mask,
                                           pos=pos, query_pos=query)
        hs_viz = hs_att[-1].clone().detach().mean(dim=1)
        hs_att = hs_att.permute(1, 0, 3, 2, 4).flatten(0, 2).view(-1, 8, self.h, self.w)
        return hs_att, hs_viz, memory

    def _attcat_tfenc_forward(self, tgt, memory, memory_key_padding_mask, pos, query, **kwargs):
        """
        Returns attention weight maps concatenated with the decoder inputs (memory); `c-att-cat-tfenc` in Tab. 3 of the paper
        """
        hs, hs_full, hs_att = self.decoder(tgt, memory, memory_key_padding_mask=memory_key_padding_mask,
                                           pos=pos, query_pos=query)
        hs_viz = hs_att[-1].clone().detach().mean(dim=1)
        hs_att = hs_att.permute(1, 0, 3, 2, 4).flatten(0, 2).view(-1, 8, self.h, self.w)
        encoded_transformer = memory.permute(1, 2, 0).view(-1, 256, self.h, self.w)
        encoded_transformer = encoded_transformer.repeat_interleave(repeats=int(hs_att.shape[0]/encoded_transformer.shape[0]), dim=0)  # b * num_decoder_outs * q, 256, h, w
        return torch.cat((hs_att, encoded_transformer), dim=1), hs_viz, memory

    def _attcat_bb_forward(self, tgt, memory, memory_key_padding_mask, pos, query, src):
        """
        Returns attention weight maps concatenated with the encoder inputs (src); `c-att-cat-bb` in Tab. 3 of the paper
        """
        hs, hs_full, hs_att = self.decoder(tgt, memory, memory_key_padding_mask=memory_key_padding_mask,
                                           pos=pos, query_pos=query)
        hs_viz = hs_att[-1].clone().detach().mean(dim=1)
        hs_att = hs_att.permute(1, 0, 3, 2, 4).flatten(0, 2).view(-1, 8, self.h, self.w)
        encoded_transformer = src.permute(1, 2, 0).view(-1, 256, self.h, self.w)
        encoded_transformer = encoded_transformer.repeat_interleave(repeats=int(hs_att.shape[0]/encoded_transformer.shape[0]), dim=0)  # b * num_decoder_outs * q, 256, h, w
        return torch.cat((hs_att, encoded_transformer), dim=1), hs_viz, memory

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        return self._decoder_forward(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query=query_embed, src=src)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []
        output_fulls = []
        output_atts = []

        for layer in self.layers:
            output, output_full, output_att = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                output_fulls.append(output_full)
                output_atts.append(output_att)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(output_fulls), torch.stack(output_atts)

        return output.unsqueeze(0), output_full.unsqueeze(0), output_att.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # only difference to the DETR implementation is that we replace the multihead attention layer with a layer
        # that returns the attention weight maps as well as the `expanded` outputs
        # For details refer to Sec. 3B of the paper
        self.multihead_attn = CustomMultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, tgt2_full, tgt2_att = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, tgt2_full, tgt2_att

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """
    Return an activation function given a string
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


# simple forward example for all query processing options
if __name__ == '__main__':
    for query_proc in ['expanded', 'att', 'attcat_bb', 'attcat_tfenc']:
        tf = Transformer(d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, return_intermediate_dec=True,
                         query_proc=query_proc, h=15, w=20).cuda()

        bb = torch.randn(2, 256, 15, 20).cuda()  # backbone output
        mask = torch.zeros(2, 15, 20).cuda().bool()
        q = torch.randn(15, 256).cuda()
        pos_enc = torch.randn_like(bb)

        hs_full, hs_viz, memory = tf(bb, mask, q, pos_enc)
        print(f"query proc: {query_proc}; output shape: {hs_full.shape}")
