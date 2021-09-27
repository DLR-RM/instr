"""
Main training script.
"""

import os

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from data_io.data_loader import DatasetHDF5, worker_init_fn
from model.instr import INSTR
from utils.utils import setup, DataLoader


def train():
    cfg = setup()

    # initialize dataset and dataloader
    train_set = DatasetHDF5(base_path=cfg.DATA.TRAIN.ROOT, split='train', apply_augmentation=cfg.DATA.TRAIN.TRANSFORMS)
    val_set = DatasetHDF5(base_path=cfg.DATA.VAL.ROOT, split='val', apply_augmentation=cfg.DATA.VAL.TRANSFORMS)

    train_loader = DataLoader(train_set, batch_size=cfg.DATA.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.DATA.NUM_WORKERS, drop_last=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_set, batch_size=cfg.DATA.VAL.BATCH_SIZE, shuffle=False, num_workers=cfg.DATA.NUM_WORKERS, drop_last=True, worker_init_fn=worker_init_fn)

    print(f"Training samples: {len(train_loader) * cfg.DATA.TRAIN.BATCH_SIZE}")
    print(f"Validation samples: {len(val_loader) * cfg.DATA.VAL.BATCH_SIZE}")

    # initialize model
    model = INSTR(cfg=cfg)

    # give rights to the output dir
    os.system(f"chmod -R a+rwx {cfg.EXP.OUTPUT_PATH}")

    # initialize tensorboard logger and checkpoint callbacks
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(cfg.EXP.OUTPUT_PATH, 'logs'))
    callbacks = [
        ModelCheckpoint(dirpath=os.path.join(cfg.EXP.OUTPUT_PATH, 'models'), filename='{epoch}-{val_loss:.4f}',
                        monitor='val_loss', mode='min', save_last=True, save_top_k=1),
        ModelCheckpoint(dirpath=os.path.join(cfg.EXP.OUTPUT_PATH, 'models'), filename='{epoch}-{val_iou:.4f}',
                        monitor='val_disp_loss', mode='min', save_last=False, save_top_k=1)
    ]

    # train the network
    trainer = pl.Trainer(gpus=1, logger=[tb_logger], default_root_dir=cfg.EXP.OUTPUT_PATH, callbacks=callbacks)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    import sys
    assert sys.version_info >= (3, 7), f"Python version has to be >= 3.7 because we use the preserving of item order in dictionaries"
    train()
