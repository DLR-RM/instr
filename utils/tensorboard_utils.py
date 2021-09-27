"""
Converter utilities for tensorboard logging.
"""

import torch
import torch.nn.functional as F

import utils.colormap as colormap


def _convert_rgb(tens):
    tens -= tens.min()
    tens = tens / tens.max()
    return tens


def _convert_disp(tens):
    tens -= tens.min()
    tens = tens / tens.max()
    tens *= 255
    if tens.ndim == 3:
        tens = tens.unsqueeze(1)
    return tens.to(dtype=torch.uint8)


def _convert_instanceseg_to_map(tens):
    if tens.ndim == 3:
        tens = tens.unsqueeze(1)

    if tens.dtype == torch.uint8:
        tens = colorize_tensor(tens, num_classes=50)
        return tens
    tens = torch.sigmoid(tens)
    bsize, ch, h, w = tens.shape

    # create a tile map for all queries
    tm = tens.clone()
    new_h, new_w = tm.shape[-2:]
    if 16 < ch <= 20:
        rows, cols = 4, 5
    elif ch == 15:
        rows, cols = 3, 5
    elif ch == 21:
        rows, cols = 5, 5
    elif ch == 50:
        rows, cols = 7, 8
    elif ch == 15:
        rows, cols = 3, 5
    elif ch == 16:
        rows, cols = 4, 4
    grid = torch.zeros(bsize, 1, new_h*rows, new_w*cols)
    qid = 0
    if tm.max() != 0:
        tm = tm / tm.max()
    for r in range(rows):
        for c in range(cols):
            grid[:, 0, new_h*r:new_h*r+new_h, new_w*c:new_w*c+new_w] = tm[:, qid, :, :]
            grid[:, 0, new_h * r:new_h * r + 1, :] = tm.max()
            grid[:, 0, new_h * r + new_h - 1:new_h * r + new_h, :] = tm.max()
            grid[:, 0, :, new_w * c:new_w * c + 1] = tm.max()
            grid[:, 0, :, new_w * c + new_w - 1:new_w * c + new_w] = tm.max()
            qid += 1
            if qid >= ch:
                break

    # threshold to discard everything below 0.5
    tens[tens < 0.5] = 0

    # select based on argmax
    valid_maps = tens.argmax(dim=1, keepdim=False)

    # don't automatically select pixels where all channels have predicted 0
    valid_maps[torch.all(tens == 0, dim=1)] = 0

    # colorize, if any instances have been found
    if valid_maps.max() != 0:
        valid_maps = colorize_tensor(valid_maps, num_classes=50)
    else:
        valid_maps = torch.zeros(bsize, 3, h, w, dtype=torch.uint8)
    return valid_maps


def _convert_instanceseg(tens):
    if tens.ndim == 3:
        tens = tens.unsqueeze(1).to(dtype=torch.uint8)

    if tens.dtype == torch.uint8:
        tens = colorize_tensor(tens, num_classes=50)
        return tens
    tens = torch.softmax(tens, dim=1)
    bsize, ch, h, w = tens.shape
    # create a tile map for all queries
    tm = tens.clone()
    new_h, new_w = tm.shape[-2:]
    if 16 < ch <= 20:
        rows, cols = 4, 5
    elif ch == 21:
        rows, cols = 5, 5
    elif ch == 50:
        rows, cols = 7, 8
    elif ch == 15:
        rows, cols = 3, 5
    elif ch == 16:
        rows, cols = 4, 4
    grid = torch.zeros(bsize, 1, new_h*rows, new_w*cols)
    qid = 0
    if tm.max() != 0:
        tm = tm / tm.max()
    for r in range(rows):
        for c in range(cols):
            grid[:, 0, new_h*r:new_h*r+new_h, new_w*c:new_w*c+new_w] = tm[:, qid, :, :]
            grid[:, 0, new_h * r:new_h * r + 1, :] = tm.max()
            grid[:, 0, new_h * r + new_h - 1:new_h * r + new_h, :] = tm.max()
            grid[:, 0, :, new_w * c:new_w * c + 1] = tm.max()
            grid[:, 0, :, new_w * c + new_w - 1:new_w * c + new_w] = tm.max()
            qid += 1
            if qid >= ch:
                break

    # threshold to discard everything below 0.5
    tens[tens < 0.5] = 0

    # select based on argmax
    valid_maps = tens.argmax(dim=1, keepdim=False)

    # don't automatically select pixels where all channels have predicted 0
    valid_maps[torch.all(tens == 0, dim=1)] = 0

    # colorize, if any instances have been found
    if valid_maps.max() != 0:
        valid_maps = colorize_tensor(valid_maps, num_classes=50)
    else:
        valid_maps = torch.zeros(bsize, 3, h, w, dtype=torch.uint8)
    return valid_maps, grid


def _convert_instanceseg_to_grid(tens):
    if tens.ndim == 3:
        tens = tens.unsqueeze(1)

    if tens.dtype == torch.uint8:
        tens = colorize_tensor(tens, num_classes=50)
        return tens
    tens = torch.sigmoid(tens)
    bsize, ch, h, w = tens.shape

    # create a tile map for all queries
    tm = tens.clone()
    new_h, new_w = tm.shape[-2:]
    if 16 < ch <= 20:
        rows, cols = 4, 5
    elif ch == 21:
        rows, cols = 5, 5
    elif ch == 50:
        rows, cols = 7, 8
    elif ch == 15:
        rows, cols = 3, 5
    elif ch == 16:
        rows, cols = 4, 4
    grid = torch.zeros(bsize, 1, new_h*rows, new_w*cols)
    qid = 0
    if tm.max() != 0:
        tm = tm / tm.max()
    for r in range(rows):
        for c in range(cols):
            grid[:, 0, new_h*r:new_h*r+new_h, new_w*c:new_w*c+new_w] = tm[:, qid, :, :]
            grid[:, 0, new_h * r:new_h * r + 1, :] = tm.max()
            grid[:, 0, new_h * r + new_h - 1:new_h * r + new_h, :] = tm.max()
            grid[:, 0, :, new_w * c:new_w * c + 1] = tm.max()
            grid[:, 0, :, new_w * c + new_w - 1:new_w * c + new_w] = tm.max()
            qid += 1
            if qid >= ch:
                break

    return grid


def colorize_tensor(tens, num_classes=50):
    if tens.ndim == 4:
        assert tens.shape[1] == 1, f"Tensor is 4d but second dim is not 1 - ndims: {tens.ndim}"
    else:
        assert tens.ndim == 3, f"Weird tensor shape:  {tens.shape}"

    tens = tens.squeeze()

    if tens.ndim == 2:
        tens = tens.unsqueeze(0)

    n, h, w = tens.shape

    colorized_tens = torch.zeros(n, 3, h, w).to(dtype=torch.uint8)
    if num_classes is None:
        num_classes = len(torch.unique(tens))

    colors = colormap.get_spaced_colors(num_classes)
    spaced_colors = torch.tensor(colors).to(dtype=torch.uint8)

    for ctr, i in enumerate(torch.unique(tens)[1:]):
        a, b, c = torch.where(tens == i)
        if a != []:
            colorized_tens[a, :, b, c] = spaced_colors[ctr]

    return colorized_tens


def _convert_queries(queries):
    # converts queries in shape b, q, h*w to tensorboard
    b, q, hw = queries.shape
    if hw == 1200:
        h, w = 30, 40
    else:
        h, w = 15, 20
    queries = queries.sigmoid().view(b, q, h, w)
    queries = F.interpolate(queries, size=(30, 40), mode='nearest')

    # create a tile map for all queries
    grid = torch.zeros(b, 1, 30 * 4, 40 * 5)
    new_h, new_w = 30, 40
    qid = 0
    if q == 15:
        rr, cc = 3, 5
    else:
        rr, cc = 4, 5
    for r in range(rr):
        for c in range(cc):
            grid[:, 0, new_h * r:new_h * r + new_h, new_w * c:new_w * c + new_w] = queries[:, qid, :, :]
            grid[:, 0, new_h * r:new_h * r + 1, :] = queries.max()
            grid[:, 0, new_h * r + new_h - 1:new_h * r + new_h, :] = queries.max()
            grid[:, 0, :, new_w * c:new_w * c + 1] = queries.max()
            grid[:, 0, :, new_w * c + new_w - 1:new_w * c + new_w] = queries.max()
            qid += 1
            if qid >= 20:
                break

    return grid
