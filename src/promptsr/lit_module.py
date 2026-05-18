from __future__ import annotations

import math
from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from .config import PromptSRConfig
from .model import PromptSRModel
from .optim import build_optimizer_and_scheduler


def rgb_to_y_channel(x: torch.Tensor) -> torch.Tensor:
    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]
    y = 16.0 / 255.0 + (64.738 / 256.0) * r + (129.057 / 256.0) * g + (25.064 / 256.0) * b
    return y.clamp(0.0, 1.0)


def shave_tensor(x: torch.Tensor, border: int) -> torch.Tensor:
    if border <= 0:
        return x
    return x[..., border:-border, border:-border]


class PromptSRLightningModule(pl.LightningModule):
    def __init__(self, cfg: PromptSRConfig):
        super().__init__()
        self.cfg = cfg
        self.model = PromptSRModel(cfg)
        self.criterion = nn.L1Loss()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.save_hyperparameters(cfg.asdict())
        self._external_optimizers: tuple[Any, Any] | None = None

    def attach_optimizers(self, optimizer, scheduler) -> None:
        self._external_optimizers = (optimizer, scheduler)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        return self.model(lr)

    def training_step(self, batch, batch_idx):
        sr = self(batch["lr"])
        loss = self.criterion(sr, batch["hr"])
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch["lr"].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        sr = self(batch["lr"]).clamp(0.0, 1.0)
        hr = batch["hr"]
        loss = self.criterion(sr, hr)
        sr_y = shave_tensor(rgb_to_y_channel(sr), self.cfg.shave_border)
        hr_y = shave_tensor(rgb_to_y_channel(hr), self.cfg.shave_border)
        self.psnr.update(sr_y, hr_y)
        self.ssim.update(sr_y, hr_y)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, batch_size=1)
        return loss

    def on_validation_epoch_end(self):
        self.log("val/psnr", self.psnr.compute(), prog_bar=True)
        self.log("val/ssim", self.ssim.compute(), prog_bar=True) # type: ignore
        self.psnr.reset()
        self.ssim.reset()

    def configure_optimizers(self): # type: ignore
        if self._external_optimizers is None:
            optimizer, scheduler = build_optimizer_and_scheduler(self.parameters(), self.cfg)
        else:
            optimizer, scheduler = self._external_optimizers
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
