"""
Various utility functions.
"""

import os
import logging
import sys
import yaml
from pathlib import Path
from yacs.config import CfgNode
from time import gmtime, strftime
import argparse
from yacs.config import CfgNode as CN
import warnings
import json
import numpy as np
from pathlib import PosixPath

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader as TorchLoader
from torch.utils.data._utils.collate import container_abcs, default_collate
import h5py


def setup():
    # obtain complete cfg node
    cfg = get_cfg()

    # make experiment dirs
    setup_exp(cfg.EXP)

    logging.getLogger(__name__).info(f"STARTING TRAINING")

    cfg.freeze()

    print(f"Cfg contains the following values:")
    print(cfg)

    # dump the cfg
    save_config(path=os.path.join(cfg.EXP.OUTPUT_PATH, 'config.yaml'), cfg=cfg)

    return cfg


def save_config(path, cfg):
    with open(os.path.join(path), 'w') as f:
        f.write(cfg.dump())


def setup_exp(cfg):
    logger = logging.getLogger(__name__)
    logger.info(f"Trying to create experiment dir ...")

    cfg = parse_exp_to_output_path(cfg)
    cfg = create_output_path(cfg)

    # save argv
    cfg.ARGV = ' '.join(sys.argv)


def parse_exp_to_output_path(cfg):
    if cfg.OUTPUT_PATH == "":
        assert cfg.ROOT != "", f"Please provide an experiment root!"

        timestamp = ""
        if cfg.WITH_STRFTIME:
            timestamp = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
        cfg.OUTPUT_PATH = os.path.join(cfg.ROOT, cfg.NAME + timestamp)
        logging.getLogger(__name__).debug(f"Parsing ROOT {cfg.ROOT} and NAME {cfg.NAME} to PATH {cfg.OUTPUT_PATH}")
        
    return cfg


def create_output_path(cfg):
    logger = logging.getLogger(__name__)
    # cfg doesn't have to have equal keys to base cfg, but should contain OUTPUT_DIR
    assert cfg.get("OUTPUT_PATH") is not None, logger.exception(f"cfg node has no key 'OUTPUT_PATH'")

    path = Path(cfg.OUTPUT_PATH)

    # create new tag folder if necessary
    if not os.path.exists(path.parent):
        os.makedirs(path.parent, exist_ok=True)
        logger.debug(f"Created TAG DIR (parent: {path.parent.as_posix()})")
    assert os.path.exists(path.parent.as_posix()), logger.exception(f"Parent of {path} does not exist!")
    assert path.as_posix() != "" and path.as_posix() != ".", logger.exception(f"Path is invalid: '{path}'")

    # create main dir
    os.makedirs(path, exist_ok=True)

    # create logs, models, tbx
    os.makedirs(path.joinpath('logs'), exist_ok=True)
    os.makedirs(path.joinpath('models'), exist_ok=True)

    logger.info(f"Created experiment dir in '{path.as_posix()}'")
    
    return cfg


def get_cfg():
    logger = logging.getLogger(__name__)
    # first: get base cfg
    cfg = get_base_cfg()

    # second: argparse
    args = get_default_argparse_args()

    # third: try to load config from list
    if args.config_file:
        logger.info(f"Loading config from file: {args.config_file}")
        cfg.merge_from_file(args.config_file)

    # fourth: merge opt args
    cfg.merge_from_list(args.OPTS)

    return cfg


def get_default_argparse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config-file", type=str, default="", metavar="FILE", help="path to config file(s)", required=True)
    parser.add_argument("-t", "--test", default=False, action='store_true', help="Testing mode")
    parser.add_argument("OPTS", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()

    return args


def get_base_cfg():
    _C = CN(new_allowed=True)

    # EXP
    _C.EXP = get_exp_cfg()

    # DEVICE
    _C.DEVICE = 'cuda'

    # DATA
    _C.DATA = CN(new_allowed=True)
    _C.DATA.TRAIN = CN(new_allowed=True)
    _C.DATA.VAL = CN(new_allowed=True)
    _C.DATA.TEST = CN(new_allowed=True)

    # MODEL
    _C.MODEL = CN(new_allowed=True)

    # OPTIMIZER
    _C.OPTIMIZER = CN(new_allowed=True)
    _C.OPTIMIZER.TYPE = 'optim.AdamW'
    _C.OPTIMIZER.LR = 0.001
    _C.OPTIMIZER.WEIGHT_DECAY = 0.01
    # do stuff like
    # _C.OPTIMIZER.DECODER.LR = 0.0001

    # SCHEDULER
    _C.SCHEDULER = CN(new_allowed=True)
    # initialize a scheduler like this
    # _C.SCHEDULER.TYPE = 'optim.lr_scheduler.StepLR'
    # pass kwargs like this
    # _C.SCHEDULER.step_size = 30

    return _C.clone()


def get_exp_cfg():
    _EXP = CN()

    _EXP.ROOT = ""
    _EXP.NAME = "test"
    _EXP.OUTPUT_PATH = ""

    _EXP.WITH_STRFTIME = True
    _EXP.ARGV = ""  # sys.argv placeholder

    return _EXP.clone()


class DataLoader(TorchLoader):
    def __init__(self, dataset, *args, **kwargs):
        kwargs["collate_fn"] = kwargs.get("collate_fn", ccollate)
        super().__init__(dataset, *args, **kwargs)


def ccollate(batch):
    # copied and adapted from the original collate_fn
    elem = batch[0]
    if isinstance(elem, container_abcs.Mapping):
        rets = pekdict({key: ccollate([d[key] for d in batch]) for key in elem})
        for key in rets.keys():
            elem[key] = rets[key]
        return elem
    else:
        return default_collate(batch)


def create_optimizer(cfg, net):
    assert isinstance(cfg, (dict, CfgNode)), f"Please provide an instance of type [dict, yacs.config:CfgNode]"
    assert isinstance(net, (nn.Module, nn.ModuleDict, nn.ModuleList)), f"Please provide an instance of type nn.[Module, ModuleList, ModuleDict]"

    ccfg = cfg.copy().pop("OPTIMIZER", cfg).copy()
    if ccfg is not None:
        print(f"Found OPTIMIZER configuration. Creating optimizer:")
    else:
        warnings.warn(f"Didn't find an optimizer - will run with the default values!")
        ccfg = {}

    param_list = []

    try:
        optimizer = eval(ccfg.pop("TYPE", "optim.AdamW"))
        lr_base = ccfg.pop("LR", 0.001)
        weight_decay_base = ccfg.pop("WEIGHT_DECAY", 0.01)
    except Exception as e:
        raise e

    children = net.named_children()
    names = []

    for (name, layer) in children:
        lr = lr_base
        if ccfg.get(name.upper()) is not None:
            lr = ccfg[name.upper()].get("LR")
            assert lr is not None, f"Layer '{name}' defined in config but lr missing"

        if lr != 0.:
            param_list.append({"params": layer.parameters(), "lr": lr})
            names.append(name)
        else:
            for param in layer.parameters():
                param.requires_grad = False

    # create optimizer
    optimizer = optimizer(param_list, lr=lr_base, weight_decay=weight_decay_base)
    logging.info(msg=f"Created optimizer: ")
    for group, name in zip(optimizer.param_groups, names):
        print(f"Layer '{name}': lr={group['lr']}, weight_decay={group['weight_decay']}")

    return optimizer


def create_scheduler(cfg, optimizer):
    assert isinstance(cfg, (dict, CfgNode)), f"Please provide an instance of type [dict, yacs.config:CfgNode]"

    ccfg = cfg.get("SCHEDULER", None)
    ccfg.defrost()

    try:
        scheduler = eval(ccfg.pop("TYPE"))
        # initialize
        scheduler = scheduler(optimizer=optimizer, **ccfg)
        logging.info(msg=f"Created scheduler: ")
        print(f"{scheduler}")
    except Exception as e:
        print(f"WARNING: Exception during scheduler creation: {e}")
        scheduler = None

    return scheduler


class pekdict(dict):
    def __init__(self, dct=None):
        super(pekdict, self).__init__()

        self._types = []
        self._descrs = []

        # conversion functions
        self._tbs = []
        self._prints = []

        if dct is not None:
            if type(dct) is dict:
                for key, value in dct.items():
                    self.add(key=key, value=value, tb=None, cprint=None)
            else:
                raise NotImplementedError
        super(pekdict, self).__init__()

        self.device = torch.device('cpu')
        if dct is not None:
            self._set_device()

    def add(self, key, value, descr=None, tb=None, cprint=None):
        # add new item
        self[key] = value

        # add metadata
        self._descrs.append(descr)
        self._tbs.append(tb)
        self._prints.append(cprint)

    def tb(self, writer, suffix='train', step=1):
        for key, value, descr, func in zip(self.keys(), self.values(), self._descrs, self._tbs):
            if descr is None:
                descr = key
            try:
                value.tb(writer, suffix=suffix, step=step)
            except AttributeError:
                logging.getLogger(__name__).debug(msg=f"Key {key} has no attribute '.tb()'")
                if func is not None:
                    logging.getLogger(__name__).debug(msg=f"Converting to tensorboard key {key} with {str(func)}")
                    tb_value = func(value)
                    writer.add_images(tag=os.path.join(descr, suffix), img_tensor=tb_value, global_step=step, dataformats='NCHW')
                else:
                    logging.getLogger(__name__).debug(msg=f"No '.tb()' set")
            except Exception as e:
                raise e

    def cpu(self):
        for key in self.keys():
            try:
                self[key] = self[key].cpu()
            except:
                logging.getLogger(__name__).warning(f"Invalid call of .cpu() ({dict_str(key, self[key])})")
        self._set_device()

        return self

    def cuda(self):
        for key in self.keys():
            try:
                self[key] = self[key].cuda()
            except:
                logging.getLogger(__name__).warning(f"Invalid call of .cuda() ({dict_str(key, self[key])})")
        self._set_device()

        return self

    def to(self, device):
        assert isinstance(device, torch.device), f"Invalid device: {device}"

        if device == torch.device('cpu'):
            self.cpu()
        else:
            self.cuda()

        return self

    def update(self, __m, **kwargs) -> None:
        super().update(__m, **kwargs)

        # also update hidden attributes
        for attr, value in __m.__dict__.items():
            if attr.startswith('_') and not attr.startswith('__'):
                logging.getLogger(__name__).debug(msg=f"Extending '{attr}' with {len(value)} values")
                eval(f"self.{attr}").extend(value)

    def _set_device(self):
        device_list = []

        if len(self.keys()) != 0:

            for key in self.keys():
                try:
                    device_list.append(self[key].device)
                except:
                    device_list.append(torch.device('cpu'))

        device_list = list(set(device_list))

        assert len(device_list) <= 2, f"Unknown devices: {device_list}"

        if len(device_list) == 2:
            self.device = 'mixed'
        else:
            self.device = device_list[0]

    def default(self):
        return dict(self)


def dict_str(key, value):
    return f"KEY: {key}, TYPE: {value.__class__.__name__}"


def load_hdf5(path, keys=None):
    if type(path) is PosixPath:
        path = path.as_posix()
    assert os.path.isfile(path), f"File {path} does not exist"
    assert '.hdf5' in path, f"File {path} is not a hdf5 file"

    with h5py.File(path, 'r') as data:
        data_keys = [key for key in data.keys()]
        if keys is None:
            keys = data_keys
        else:
            assert [key in data_keys for key in keys], f"Invalid keys ({keys}) for data keys: {data_keys}"

        hdf5_data = {}
        for key in keys:
            if data[key].dtype.char == 'S':
                try:
                    hdf5_data[key] = json.loads(bytes(np.array(data[key])))[0]
                except:
                    hdf5_data[key] = data[key]
            else:
                hdf5_data[key] = np.array(data[key])

    return hdf5_data


def load_config(path):
    assert path.endswith('.yaml'), f"File is not a .yaml file: {path}"

    with open(path, 'r') as f:
        cfg = CfgNode(yaml.safe_load(f))

    return cfg
