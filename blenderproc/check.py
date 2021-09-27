"""
Dummy script to check generated training data.
"""

import cv2
import argparse

from utils.utils import load_hdf5
from utils.pred_utils import overlay_im_with_masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('hdf5_file_path', type=str)
    args = parser.parse_args()

    data = load_hdf5(args.hdf5_file_path)
    left_overlay = overlay_im_with_masks(data['colors'][0], ma=data['segmap'], alpha=0.3)

    cv2.imshow('left', data['colors'][0])
    cv2.imshow('left_overlay', left_overlay)
    cv2.imshow('right', data['colors'][1])
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
