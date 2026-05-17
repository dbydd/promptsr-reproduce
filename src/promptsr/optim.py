from __future__ import annotations

import torch

from .config import PromptSRConfig


# Paper schedules are iteration-based and vary by scale.
# This helper supports both the x2-from-scratch and x3/x4-finetune regimes.
def build_optimizer_and_scheduler(params, cfg: PromptSRConfig):
    optimizer = torch.optim.Adam(
        params,
        lr=cfg.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=cfg.weight_decay,
    )
    if cfg.train_mode == "x2_scratch":
        milestones = [250_000, 400_000, 450_000, 475_000, 490_000]
    else:
        milestones = [150_000, 200_000, 225_000, 240_000]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    return optimizer, scheduler
