"""
Various prediction utilities.
"""

import numpy as np
import os
import torch
from PIL import Image
import torchvision.transforms.functional as ttf
import yaml
from yacs.config import CfgNode
import cv2
from model.instr import INSTR
from utils.colormap import get_spaced_colors


YCB_OBJECTS = [
    '003_cracker_box',
    '005_tomato_soup_can',
    '006_mustard_bottle',
    '007_tuna_fish_can',
    '008_pudding_box',
    '010_potted_meat_can',
    '011_banana',
    '019_pitcher_base',
    '021_bleach_cleanser',
    '024_bowl',
    '025_mug',
    '035_power_drill',
    '037_scissors',
    '052_extra_large_clamp',
    '061_foam_brick',
]


def stuff_from_state_dict_path(path):
    cfg_path = '/'.join(path.split('/')[:-2]) + '/config.yaml'
    with open(cfg_path, 'r') as f:
        cfg = CfgNode(yaml.load(f))

    net = INSTR(cfg)
    state_dict = torch.load(path)
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']

    rets = net.load_state_dict(state_dict, strict=False)

    print(f"Loaded state dict from {path}: {rets}")
    return cfg, net


def process_im(im, device=torch.device('cuda')):
    im = np.array(im)[:, :, :3]
    im = ttf.to_pil_image(im)
    im = ttf.resize(im, [480, 640], interpolation=Image.LINEAR)
    im = ttf.to_tensor(im)
    im = ttf.normalize(im, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return im.unsqueeze(0).to(device=device)


def process_depth(depth, thresh=10.):
    depth = np.nan_to_num(depth)
    depth[depth > thresh] = thresh
    return torch.from_numpy(depth)


def process_disp(disp):
    return torch.from_numpy(disp.astype(np.float32))


def disp_to_depth(disp, f, b):
    disp = disp.squeeze().cpu().numpy().astype(np.float32)
    depth = np.zeros_like(disp)
    depth[disp != 0] = b * f / disp[disp != 0]
    return depth


def load_data(root=''):
    sensors = ['rc_visard', 'zed']
    data = {
        'rc_visard': {},
        'zed': {}
    }

    for sensor in sensors:
        folders = os.listdir(os.path.join(root, sensor))
        for folder in folders:
            data[sensor][folder] = load_folder(root=root, sensor=sensor, folder=folder)
    return data


def load_folder(root='', sensor='rc_visard', folder='black_table'):
    root = os.path.join(root, sensor)
    suffs = sorted(os.listdir(os.path.join(root, folder, 'left_rgb')))
    data = []

    for suff in suffs:
        left = os.path.join(root, folder, 'left_rgb', suff)
        right = os.path.join(root, folder, 'right_rgb', suff)
        depth = os.path.join(root, folder, 'depth', suff.split('.')[0] + '.npy')
        gt = os.path.join(root, folder, 'gt', suff)

        assert os.path.isfile(left)
        assert os.path.isfile(right)
        assert os.path.isfile(depth)
        assert os.path.isfile(gt)
        data.append([left, right, depth, gt])

    return data


def overlay_im_with_masks(im, ma, alpha=0.5):
    """
    Overlays an image with corresponding annotations.
    Args:
        im (uint8 np.array): image of shape h, w, 3
        ma (uint8 np array): mask of shape h, w; expects unique integers for object instances
        alpha (float): see cv2.addWeighted() for more information
    Returns:
        uint8 np.array: colorized image of shape h, w, 3
    """

    if ma.max() == 0:
        return im
    colors = get_spaced_colors(50)
    im_col = im.copy()
    for ctr, i in enumerate(np.unique(ma)[1:]):
        a, b = np.where(ma == i)
        if a != []:
            im_col [a, b, :] = colors[ctr]
    im_overlay = im.copy()
    im_overlay = cv2.addWeighted(im_overlay, alpha, im_col, 1 - alpha, 0.0)
    return im_overlay
