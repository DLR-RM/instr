"""
Prediction script for STIOS dataset.
Website: https://www.dlr.de/rm/en/desktopdefault.aspx/tabid-17628/#gallery/36367
Download link: https://zenodo.org/record/4706907#.YROCeDqxVhE
Code utilities: https://github.com/DLR-RM/stios-utils
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import time

from utils.pred_utils import load_data, process_im, stuff_from_state_dict_path
from utils.confmat import ConfusionMatrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-dict', type=str, default='./pretrained_instr/models/pretrained_model.pth', help="Path to INSTR checkpoint")
    parser.add_argument('--root', type=str, required=True, help="STIOS root")
    parser.add_argument('--rcvisard', default=False, action='store_true', help="Run on rc_visard images")
    parser.add_argument('--zed', default=False, action='store_true', help="Run on ZED images")
    args = parser.parse_args()

    assert args.rcvisard or args.zed
    assert not (args.rcvisard and args.zed)
    assert os.path.isfile(args.state_dict)

    cfg, net = stuff_from_state_dict_path(args.state_dict)
    if args.zed:
        print('Modifying subpixel correlation layer to fit ZED intrinsics')
        time.sleep(1)
        net.adapt_to_new_intrinsics(f_new=1390.0277099609375 / (2208/640), b_new=0.12)
    else:
        print('Modifying subpixel correlation layer to fit rc_visard intrinsics')
        time.sleep(1)
        net.adapt_to_new_intrinsics(f_new=1082.28 / (1280/640), b_new=0.0650206)

    net = net.cuda().eval()

    paths = load_data(root=args.root)

    results = {}

    for sensor in paths.keys():
        if sensor == 'zed' and not args.zed:
            continue
        if sensor == 'rc_visard' and not args.rcvisard:
            continue

        # go through all folders
        for folder in tqdm(paths[sensor].keys()):

            # reset confusion matrix per folder
            ious = []
            f1s = []
            num_dets_diff = []
            det_75_ratio = []
            # load images
            for (left, right, depth, gt) in tqdm(paths[sensor][folder]):
                mat = ConfusionMatrix(threshold=0.5)

                left = Image.open(left)
                right = Image.open(right)
                left_t = process_im(left)
                right_t = process_im(right)

                gt = np.array(Image.open(gt))
                gt = torch.from_numpy(gt).unsqueeze(0)

                with torch.no_grad():
                    preds = net({'color_0': left_t, 'color_1': right_t})
                pred = preds['predictions_0'][0].unsqueeze(0)

                mat(pred, targets=gt)
                ious.append(mat.get_iou().item())
                f1s.append(mat.get_f1().item())

            results[folder] = {
                'ious': ious,
                'f1s': f1s,
                'num_dets_diff': num_dets_diff,
                'det_75': det_75_ratio
            }

    mean = []
    for f, folder in enumerate(results.keys()):
        print(f"{sensor}: {folder}: {np.mean(results[folder]['ious'])*100:.4f}")
        mean.extend(results[folder]['ious'])
    print(f"---\nmean: {np.mean(mean)*100:.4f}\n")
    mean = []
    for f, folder in enumerate(results.keys()):
        print(f"{sensor}: {folder}: {np.mean(results[folder]['f1s'])*100:.4f}")
        mean.extend(results[folder]['f1s'])
    print(f"---\nmean: {np.mean(mean)*100:.4f}\n")


if __name__ == '__main__':
    main()
