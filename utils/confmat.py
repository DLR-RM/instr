"""
ConfusionMatrix class for calculating instance iou.
"""

import torch
import torch.nn.functional as F

from model.matcher import hungarian_matcher
from model.loss import get_preds_permutation_idx, get_tgt_permutation_idx


class ConfusionMatrix:
    """
    Wrapper to calculate instance iou.
    Does matching internally.
    """
    def __init__(self, threshold=0.5):
        self.tp = 0.
        self.fp = 0.
        self.fn = 0.

        self.threshold = threshold

    def reset(self):
        self.tp = 0.
        self.fp = 0.
        self.fn = 0.

    def __call__(self, preds, targets):
        bsize, n_queries, h, w = preds.shape
        preds = torch.sigmoid(preds)

        preds[preds < self.threshold] = 0
        preds[preds >= self.threshold] = 1

        # one-hot encoding
        tgts_onehot = F.one_hot(targets.long(), num_classes=-1)
        tgts_onehot = tgts_onehot[..., 1:]  # skip background
        tgts_onehot = tgts_onehot.squeeze().to(dtype=torch.float32)

        # extend tar with some zero tensor channels
        if bsize == 1:
            tgts_onehot = tgts_onehot.unsqueeze(0)
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

        tp, fp, fn = self._calculate(preds, tgts_extend)
        self.tp += tp
        self.fp += fp
        self.fn += fn

        return 0

    def _calculate(self, preds, tgts):
        temp = preds == 1
        temp_l = tgts == 1

        # tp = np.logical_and(temp, temp_l)
        tp = temp & temp_l
        temp[temp_l] = True

        # fp = np.logical_xor(temp, temp_l)
        fp = temp ^ temp_l

        temp = preds == 1
        temp[fp] = False

        # fn = np.logical_xor(temp, temp_l)
        fn = temp ^ temp_l

        return tp.sum(), fp.sum(), fn.sum()

    def get_iou(self):
        return self.tp / (self.tp + self.fp + self.fn + 1e-10)

    def get_pre(self):
        return self.tp / (self.tp + self.fp + 1e-10)

    def get_rec(self):
        return self.tp / (self.tp + self.fn + 1e-10)

    def get_f1(self):
        pre = self.get_pre()
        rec = self.get_rec()
        return 2 * ((pre * rec) / (pre + rec + 1e-10))

    def tb(self, writer, descr, suffix, step):
        if descr is not None:
            descr = descr + '_'
        else:
            descr = ""
        writer.add_scalar(tag=f"{descr}obj_iou/{suffix}", scalar_value=self.get_iou(), global_step=step)
        writer.add_scalar(tag=f"{descr}obj_pre/{suffix}", scalar_value=self.get_pre(), global_step=step)
        writer.add_scalar(tag=f"{descr}obj_rec/{suffix}", scalar_value=self.get_rec(), global_step=step)
        writer.add_scalar(tag=f"{descr}obj_f1/{suffix}", scalar_value=self.get_f1(), global_step=step)


if __name__ == '__main__':
    mat = ConfusionMatrix()
    pred = torch.randn(2, 10, 480, 640).cuda()
    tar = torch.abs(torch.randn(2, 480, 640)).cuda()

    mat(pred, tar)
    print(mat.get_iou())
