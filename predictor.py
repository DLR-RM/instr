"""
Main predictor class.
"""

import numpy as np
import cv2

import torch

from utils.tensorboard_utils import colorize_tensor
from utils.pred_utils import stuff_from_state_dict_path, process_im


class Predictor:
    def __init__(self, state_dict_path, focal_length=541.15, baseline=0.065, device=torch.device('cuda'), return_depth=True):
        cfg, net = stuff_from_state_dict_path(state_dict_path)
        net = net.to(device).eval()
        net.adapt_to_new_intrinsics(f_new=focal_length, b_new=baseline)
        self.net = net
        self.device = device
        self.focal_length = focal_length
        self.baseline = baseline
        self.return_depth = return_depth

    def predict(self, left, right):
        t_left, t_right = process_im(im=left, device=self.device), process_im(im=right, device=self.device)
        with torch.no_grad():
            preds = self.net({'color_0': t_left, 'color_1': t_right})
        if self.return_depth:
            return self._preds_to_map(preds['predictions_0']), self.disp_to_depth(preds['disp_pred'].cpu().squeeze().numpy())
        return self._preds_to_map(preds['predictions_0']), preds['disp_pred'].cpu().squeeze().numpy()

    def disp_to_depth(self, disp):
        disp = disp.squeeze().astype(np.float32)
        depth = np.zeros_like(disp)
        depth[disp != 0] = self.baseline * self.focal_length / disp[disp != 0]
        return depth

    def _preds_to_map(self, preds):
        preds = torch.sigmoid(preds)

        preds[preds < 0.5] = 0
        # select based on argmax
        valid_maps = preds.argmax(dim=1, keepdim=False)
        # don't automatically select pixels where all channels have predicted 0
        valid_maps[torch.all(preds == 0, dim=1)] = 0

        return valid_maps.cpu().squeeze().numpy()

    def colorize_preds(self, preds, rgb=None, alpha=0.5):
        # colorize, if any instances have been found
        if preds.max() != 0:
            preds = colorize_tensor(preds, num_classes=50)
        else:
            preds = torch.zeros(preds.shape[0], 3, preds.shape[-2], preds.shape[-1], dtype=torch.uint8)
        preds = preds.squeeze().permute(1, 2, 0).numpy()
        if rgb is not None:
            preds = cv2.addWeighted(rgb, alpha, preds, 1 - alpha, 0.0)
        return preds


if __name__ == '__main__':
    predictor = Predictor('./pretrained_instr/models/pretrained_model.pth')
    preds, disp = predictor.predict(np.zeros((480, 640, 3), dtype=np.uint8), np.zeros((480, 640, 3), dtype=np.uint8))
    print(preds.shape, disp.shape)
