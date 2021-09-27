"""
Adapted from https://github.com/facebookresearch/detr.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

import torch


@torch.no_grad()
def hungarian_matcher(preds, targets):
    """
    Computes an assignment between the targets and the predictions of the network.

    Args:
        preds (torch.tensor): Tensor of dim [bsize, n_queries, height, width] with the segmentation masks proposals
        targets (torch.tensor): Tensor of dim [bsize, n_queries, height, width] with the segmentation masks proposals padded with zeros if necessary

    Returns:
        A list of size bsize, containing tuples of (index_i, index_j) where:
            - index_i are the indices of the selected predictions (in order)
            - index_j are the indices of the corresponding selected targets (in order)
        For each batch element, it holds:
            len(index_i) = len(index_j) = min(n_queries, n_targets)
    """
    bsize, n_queries, height, width = preds.shape

    indices = []

    for b in range(bsize):
        n_targets, _, _ = targets[b].shape

        out = preds[b].flatten(1, 2)  # [num_queries, height * width]
        tgt = targets[b].flatten(1, 2)  # [num_targets, height * width]

        # compute true positive for every output-target combination
        tp = torch.mm(out, tgt.transpose(0, 1))  # [num_queries, num_targets]

        # compute area sum for every output-target combination
        out_area = torch.sum(out, dim=1).expand(n_targets, n_queries).transpose(0, 1)  # [num_queries, num_targets]
        tgt_area = torch.sum(tgt, dim=1).expand(n_queries, n_targets)  # [num_queries, num_targets]

        denominator = out_area + tgt_area
        tp = 2 * tp

        # make sure division by 0 is handeled; can happen since the one-hot encoding is over a whole batch
        denominator[denominator == 0] = 1e-10  # add epsilon to prevent 0-divison

        # (Dice) coefficient matrix
        C = -torch.div(tp, denominator).cpu()

        gt_ind, pred_ind = linear_sum_assignment(np.transpose(C))
        indices.append([torch.as_tensor(pred_ind).squeeze(), torch.as_tensor(gt_ind).squeeze()])

    return indices


# simple forward example
if __name__ == '__main__':
    preds = torch.randn(2, 10, 100, 100).cuda()
    tars = torch.randn_like(preds)

    indices = hungarian_matcher(preds, tars)
    print(indices)  # [[tens, tens], [tens, tens]]
