from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def save_image(path: Path, array: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)


def main():
    root = Path("data")
    train_hr = root / "DIV2K_train_HR"
    train_lr = root / "DIV2K_train_LR_bicubic" / "X4"
    val_hr = root / "Set5" / "HR"
    val_lr = root / "Set5" / "LR_bicubic" / "X4"
    for directory in [train_hr, train_lr, val_hr, val_lr]:
        directory.mkdir(parents=True, exist_ok=True)

    for index, size in enumerate([128, 144, 160, 176], start=1):
        grid_y, grid_x = np.mgrid[0:size, 0:size]
        hr = np.stack([
            (grid_x * 255 / max(1, size - 1)),
            (grid_y * 255 / max(1, size - 1)),
            ((grid_x + grid_y) % 32) * 8,
        ], axis=-1)
        hr = np.clip(hr, 0, 255)
        hr_img = Image.fromarray(hr.astype(np.uint8))
        lr_img = hr_img.resize((size // 4, size // 4), Image.BICUBIC)
        name = f"sample{index:03d}"
        save_image(train_hr / f"{name}.png", np.asarray(hr_img))
        save_image(train_lr / f"{name}x4.png", np.asarray(lr_img))

    size = 128
    grid_y, grid_x = np.mgrid[0:size, 0:size]
    hr = np.stack([
        ((np.sin(grid_x / 6.0) + 1) * 127.5),
        ((np.cos(grid_y / 7.0) + 1) * 127.5),
        ((np.sin((grid_x + grid_y) / 10.0) + 1) * 127.5),
    ], axis=-1)
    hr = np.clip(hr, 0, 255)
    hr_img = Image.fromarray(hr.astype(np.uint8))
    lr_img = hr_img.resize((size // 4, size // 4), Image.BICUBIC)
    save_image(val_hr / "baby.png", np.asarray(hr_img))
    save_image(val_lr / "babyx4.png", np.asarray(lr_img))
    print("minimal dataset prepared under data/")


if __name__ == "__main__":
    main()
