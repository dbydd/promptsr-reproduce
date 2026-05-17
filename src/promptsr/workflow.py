from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch

from .config import PromptSRConfig
from .data import PromptSRDataModule
from .lit_module import PromptSRLightningModule
from .optim import build_optimizer_and_scheduler
from .trainer_factory import build_trainer


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)


def _load_compatible_state(module: PromptSRLightningModule, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    incoming = checkpoint.get('state_dict', checkpoint)
    current = module.state_dict()
    filtered = {}
    skipped = []
    for k, v in incoming.items():
        if k in current and current[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)
    result = module.load_state_dict(filtered, strict=False)
    print('loaded compatible init checkpoint', checkpoint_path)
    print('loaded keys', len(filtered), 'skipped keys', len(skipped))
    if skipped:
        print('first skipped keys', skipped[:10])
    print('missing keys', len(result.missing_keys), 'unexpected keys', len(result.unexpected_keys))


def run_training(cfg: PromptSRConfig, max_steps: int | None = None):
    seed_everything(cfg.seed)
    cfg.ensure_output_dir()
    datamodule = PromptSRDataModule(cfg)
    model = PromptSRLightningModule(cfg)

    if cfg.init_checkpoint:
        _load_compatible_state(model, cfg.init_checkpoint)

    optimizer, scheduler = build_optimizer_and_scheduler(model.parameters(), cfg)
    model.attach_optimizers(optimizer, scheduler)
    trainer = build_trainer(cfg, max_steps=max_steps)
    trainer.fit(model, datamodule=datamodule)
    return trainer, model
