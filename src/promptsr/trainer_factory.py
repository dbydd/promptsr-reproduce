from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from .config import PromptSRConfig


def get_accelerator() -> str:
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'gpu'
    return 'cpu'


def build_trainer(cfg: PromptSRConfig, max_steps: int | None = None) -> pl.Trainer:
    output_dir = Path(cfg.output_dir)
    checkpoint = ModelCheckpoint(
        dirpath=output_dir / 'checkpoints',
        save_top_k=1,
        monitor='val/psnr',
        mode='max',
        save_last=True,
        every_n_train_steps=cfg.checkpoint_every_n_train_steps,
    )
    logger = TensorBoardLogger(save_dir=str(output_dir), name='tensorboard')
    return pl.Trainer(
        accelerator=get_accelerator(),
        devices=1,
        max_epochs=cfg.max_epochs,
        max_steps=max_steps if max_steps is not None else -1,
        logger=logger,
        callbacks=[checkpoint],
        precision='32-true',
        log_every_n_steps=cfg.log_every_n_steps,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
    )
