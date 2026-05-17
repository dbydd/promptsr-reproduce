from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from .config import PromptSRConfig
from .data import PromptSRDataModule
from .lit_module import PromptSRLightningModule


def get_accelerator() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "gpu"
    return "cpu"


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PromptSR reproduction")
    parser.add_argument("--train-hr-dir", default="data/DIV2K_train_HR")
    parser.add_argument("--train-lr-dir", default="data/DIV2K_train_LR_bicubic/X4")
    parser.add_argument("--val-hr-dir", default="data/Set5/HR")
    parser.add_argument("--val-lr-dir", default="data/Set5/LR_bicubic/X4")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    args = build_parser().parse_args()
    cfg = PromptSRConfig(
        train_hr_dir=args.train_hr_dir,
        train_lr_dir=args.train_lr_dir,
        val_hr_dir=args.val_hr_dir,
        val_lr_dir=args.val_lr_dir,
        scale=args.scale,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        max_epochs=args.max_epochs,
        num_workers=args.num_workers,
        lr=args.lr,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    seed_everything(cfg.seed)
    output_dir = cfg.ensure_output_dir()
    datamodule = PromptSRDataModule(cfg)
    model = PromptSRLightningModule(cfg)
    checkpoint = ModelCheckpoint(
        dirpath=Path(output_dir) / "checkpoints",
        save_top_k=1,
        monitor="val/psnr",
        mode="max",
        save_last=True,
    )
    logger = CSVLogger(save_dir=output_dir, name="logs")
    trainer = pl.Trainer(
        accelerator=get_accelerator(),
        devices=1,
        max_epochs=cfg.max_epochs,
        logger=logger,
        callbacks=[checkpoint],
        precision="32-true",
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
