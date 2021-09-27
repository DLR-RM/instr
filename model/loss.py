"""
Dice loss and Bipartite matching loss.
"""

import torch
import torch.nn.functional as F

from model.matcher import hungarian_matcher


def get_preds_permutation_idx(indices, n_queries):
    # permute PREDICTIONS following indices
    batch_idx = torch.cat([torch.full_like(preds, i) for i, (preds, _) in enumerate(indices)])
    preds_idx = torch.cat([preds for (preds, _) in indices])
    return batch_idx * n_queries + preds_idx


def get_tgt_permutation_idx(indices, n_queries):
    # permute TARGETS following indices
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx * n_queries + tgt_idx


def _dice_loss(preds, targets, power=0.2, pos_weight=1., neg_weight=1.):
    """
    Compute the DICE loss, similar to generalized IOU for masks. Follows Eq. 4 in https://arxiv.org/pdf/1912.04488.pdf
    Args:
        preds (torch.tensor): float tensor of arbitrary shape - predictions for each sample.
        targets (torch.tensor): tensor of same shape as preds - stores binary classification label for each element.
        power (float): weighting factor for focusing more on least and most correct predictions; [gamma] in Eq. 7 of the paper
        pos_weight (float): weighting factor for object samples
        neg_weight (float): weighting factor for empty samples

    Returns:
        pos_loss (float): total positive (object queries) loss
        neg_loss (float): total negative (empty queries) loss
    """

    _, h, w = preds.shape
    _preds = preds.flatten(1)
    _tgts = targets.flatten(1)

    tgt_area = (_tgts).sum(-1)

    pos_mask_det = torch.nonzero((tgt_area != 0), as_tuple=False)
    neg_mask_det = torch.nonzero((tgt_area == 0), as_tuple=False)

    # positive predictions
    _pp = _preds[pos_mask_det].squeeze()
    _tp = _tgts[pos_mask_det].squeeze()

    if _pp.ndim == 1:
        _pp = _pp.unsqueeze(0)
    if _tp.ndim == 1:
        _tp = _tp.unsqueeze(0)

    numerator = (2 * (_pp * _tp).sum(1))
    denominator = (_pp).sum(-1) + (_tp).sum(-1)
    loss_pos = 1 - ((numerator + 1) / (denominator + 1))

    # negative samples (no object) -> invert masks
    _p = (1 - _preds[neg_mask_det].squeeze())
    _t = (1 - _tgts[neg_mask_det].squeeze())

    numerator = 2 * (_p * _t).mean(1)
    denominator = (_p + _t).mean(-1)
    loss_neg = torch.pow(-torch.log(((numerator + 1) / (denominator + 1))) + 1e-4, power)

    return loss_pos.mean() * pos_weight, loss_neg.mean() * neg_weight


def bipartite_matching_segmentation_loss(pred, target, power=0.2, pos_weight=1., neg_weight=1.):
    """
    Performs matching and loss computation.
    Args:
        pred (torch.tensor): float tensor of arbitrary shape - predictions for each sample.
        target (torch.tensor): tensor of same shape as pred - stores binary classification label for each element.
        power (float): weighting factor for focusing more on least and most correct predictions; [gamma] in Eq. 7 of the paper
        pos_weight (float): weighting factor for object samples
        neg_weight (float): weighting factor for empty samples

    Returns:
        pos_loss (float): total positive (object queries) loss
        neg_loss (float): total negative (empty queries) loss
    """

    preds = pred
    targets = target

    bsize, n_queries, h, w = preds.shape
    preds = torch.sigmoid(preds)

    # one-hot encoding
    tgts_onehot = F.one_hot(targets.long(), num_classes=-1)
    tgts_onehot = tgts_onehot[..., 1:]  # skip background
    tgts_onehot = tgts_onehot.squeeze().to(dtype=torch.float32)

    # extend tar with some zero tensor channels
    if tgts_onehot.ndim == 3:
        tgts_onehot = tgts_onehot.unsqueeze(-1)
    tgts_onehot = tgts_onehot.permute(0, 3, 1, 2)  # [bsize, n_targets, height, width]
    tgts_extend = torch.zeros_like(preds).to(preds.device)
    n_obj_tar = tgts_onehot.shape[1]
    tgts_extend[:, :n_obj_tar, :, :] = tgts_onehot

    # Retrieve the matching between the outputs and the targets
    indices = hungarian_matcher(preds.clone().detach(), tgts_extend)
    preds_idx = get_preds_permutation_idx(indices, n_queries=n_queries)
    tgt_idx = get_tgt_permutation_idx(indices, n_queries=n_queries)

    # restructure both targets and predictions for efficient loss calculation
    tgts_extend = tgts_extend.flatten(0, 1)  # [bsize*n_queries,height, width]
    preds = preds.flatten(0, 1)  # [bsize*n_queries,height, width]

    # reorder both tensors to match each other based on self.matcher result
    preds = preds[preds_idx]
    tgts_extend = tgts_extend[tgt_idx]

    # main loss calculation
    pos, neg = _dice_loss(preds, tgts_extend, power=power, pos_weight=pos_weight, neg_weight=neg_weight)
    return pos + neg
