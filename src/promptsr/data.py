from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from .config import PromptSRConfig


IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        # 灰度图: [H, W] -> [H, W, 1]
        array = np.expand_dims(array, axis=-1)
    # PIL/NumPy 默认是 HWC，这里转成 PyTorch 常用的 CHW
    # [H, W, C] -> [C, H, W]
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array)


def random_crop_pair(lr: Image.Image, hr: Image.Image, patch_size: int, scale: int):
    lr_w, lr_h = lr.size
    lr_patch = patch_size // scale
    if lr_w < lr_patch or lr_h < lr_patch:
        raise ValueError(f'LR image too small for patch: {(lr_w, lr_h)} vs {lr_patch}')
    left = random.randint(0, lr_w - lr_patch)
    top = random.randint(0, lr_h - lr_patch)
    lr_crop = lr.crop((left, top, left + lr_patch, top + lr_patch))
    hr_left, hr_top = left * scale, top * scale
    hr_crop = hr.crop((hr_left, hr_top, hr_left + patch_size, hr_top + patch_size))
    return lr_crop, hr_crop


def center_crop_pair(lr: Image.Image, hr: Image.Image, patch_size: int, scale: int):
    lr_patch = patch_size // scale
    lr_w, lr_h = lr.size
    left = max(0, (lr_w - lr_patch) // 2)
    top = max(0, (lr_h - lr_patch) // 2)
    lr_crop = lr.crop((left, top, left + lr_patch, top + lr_patch))
    hr_left, hr_top = left * scale, top * scale
    hr_crop = hr.crop((hr_left, hr_top, hr_left + patch_size, hr_top + patch_size))
    return lr_crop, hr_crop


class PairedSRDataset(Dataset):
    def __init__(self, lr_dir: str, hr_dir: str, scale: int, patch_size: int, train: bool):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.scale = scale
        self.patch_size = patch_size
        self.train = train
        self.samples = self._collect_pairs()

    def _collect_pairs(self):
        if not self.lr_dir.exists() or not self.hr_dir.exists():
            return []
        hr_map = {p.stem: p for p in self.hr_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
        samples = []
        for lr_path in self.lr_dir.iterdir():
            if lr_path.suffix.lower() not in IMG_EXTS:
                continue
            key = lr_path.stem.replace(f'x{self.scale}', '').replace(f'X{self.scale}', '')
            hr_path = hr_map.get(key) or hr_map.get(lr_path.stem)
            if hr_path is not None:
                samples.append((lr_path, hr_path))
        return sorted(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        lr_path, hr_path = self.samples[index]
        lr = Image.open(lr_path).convert('RGB')
        hr = Image.open(hr_path).convert('RGB')
        crop_fn = random_crop_pair if self.train else center_crop_pair
        lr, hr = crop_fn(lr, hr, self.patch_size, self.scale)
        return {'lr': pil_to_tensor(lr), 'hr': pil_to_tensor(hr), 'name': hr_path.stem}


class PromptSRDataModule(pl.LightningDataModule):
    def __init__(self, cfg: PromptSRConfig):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str | None = None):
        self.train_dataset = PairedSRDataset(self.cfg.train_lr_dir, self.cfg.train_hr_dir, self.cfg.scale, self.cfg.patch_size, True)
        self.val_dataset = PairedSRDataset(self.cfg.val_lr_dir, self.cfg.val_hr_dir, self.cfg.scale, self.cfg.patch_size, False)

    def train_dataloader(self):
        kwargs = dict(
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=self.cfg.num_workers > 0,
        )
        if self.cfg.num_workers > 0:
            kwargs['prefetch_factor'] = 8
        return DataLoader(self.train_dataset, **kwargs) # type: ignore

    def val_dataloader(self):
        kwargs = dict(
            batch_size=self.cfg.val_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=self.cfg.num_workers > 0,
        )
        if self.cfg.num_workers > 0:
            kwargs['prefetch_factor'] = 4
        return DataLoader(self.val_dataset, **kwargs) # type: ignore
