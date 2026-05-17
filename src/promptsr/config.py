from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class PromptSRConfig:
    scale: int = 4
    in_channels: int = 3
    embed_dim: int = 48
    num_residual_groups: int = 4
    num_cpbs_per_group: int = 3
    prompt_dim: int = 48
    num_categories: int = 128
    anchor_downsample: int = 8
    anchor_update_alpha: float = 0.01
    num_heads: int = 4
    window_size: int = 16
    mlp_ratio: float = 1.0
    lr: float = 2e-4
    weight_decay: float = 0.0
    batch_size: int = 48
    num_workers: int = 0
    val_batch_size: int = 16
    max_epochs: int = 100
    patch_size: int = 64
    train_hr_dir: str = "data/DIV2K_train_HR"
    train_lr_dir: str = "data/DIV2K_train_LR_bicubic/X4"
    val_hr_dir: str = "data/DIV2K_valid_HR"
    val_lr_dir: str = "data/DIV2K_valid_LR_bicubic/X4"
    benchmark_root: str = "data/benchmarks"
    output_dir: str = "outputs"
    seed: int = 42
    train_mode: str = "x4_finetune"
    shave_border: int = 4
    init_checkpoint: str = ""
    check_val_every_n_epoch: int = 5
    log_every_n_steps: int = 10
    checkpoint_every_n_train_steps: int = 100

    def asdict(self) -> dict:
        return asdict(self)

    def ensure_output_dir(self) -> Path:
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
