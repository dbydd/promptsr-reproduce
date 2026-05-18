from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from PIL import Image

from .config import PromptSRConfig
from .lit_module import rgb_to_y_channel


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_image_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    # NumPy/PIL: [H, W, C]
    # transpose(2, 0, 1) 后: [C, H, W]
    tensor = torch.from_numpy(__import__("numpy").asarray(image, dtype=__import__("numpy").float32).transpose(2, 0, 1)) / 255.0
    # unsqueeze(0): [C, H, W] -> [1, C, H, W]
    return tensor.unsqueeze(0)


def shave_tensor(x: torch.Tensor, border: int) -> torch.Tensor:
    if border <= 0:
        return x
    return x[..., border:-border, border:-border]


def paired_benchmark_paths(dataset_dir: Path, scale: int):
    hr_dir = dataset_dir / "HR"
    lr_dir = dataset_dir / "LR_bicubic" / f"X{scale}"
    if not hr_dir.exists() or not lr_dir.exists():
        return []
    hr_map = {p.stem: p for p in hr_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    pairs = []
    for lr_path in lr_dir.iterdir():
        if lr_path.suffix.lower() not in IMG_EXTS:
            continue
        key = lr_path.stem.replace(f"x{scale}", "").replace(f"X{scale}", "")
        hr_path = hr_map.get(key) or hr_map.get(lr_path.stem)
        if hr_path is not None:
            pairs.append((lr_path, hr_path))
    return sorted(pairs)


def compute_psnr(sr: torch.Tensor, hr: torch.Tensor) -> float:
    mse = torch.mean((sr - hr) ** 2).item()
    if mse == 0:
        return 99.0
    import math
    return 10.0 * math.log10(1.0 / mse)


def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype)
    coords = coords - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g)
    return kernel_2d.view(1, 1, window_size, window_size)


def compute_ssim(sr: torch.Tensor, hr: torch.Tensor) -> float:
    sr = sr.to(torch.float32)
    hr = hr.to(torch.float32)
    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)
    window = _gaussian_kernel(device=sr.device, dtype=sr.dtype)

    mu1 = F.conv2d(sr, window, stride=1, padding=0)
    mu2 = F.conv2d(hr, window, stride=1, padding=0)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(sr * sr, window, stride=1, padding=0) - mu1_sq
    sigma2_sq = F.conv2d(hr * hr, window, stride=1, padding=0) - mu2_sq
    sigma12 = F.conv2d(sr * hr, window, stride=1, padding=0) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(ssim_map.mean().item())


def evaluate_dataset(model, dataset_dir: Path, cfg: PromptSRConfig, device: torch.device):
    pairs = paired_benchmark_paths(dataset_dir, cfg.scale)
    rows = []
    model.eval()
    with torch.no_grad():
        for lr_path, hr_path in pairs:
            lr = load_image_tensor(lr_path).to(device)
            hr = load_image_tensor(hr_path).to(device)
            sr = model(lr).clamp(0.0, 1.0)
            _, _, sh, sw = sr.shape
            hr_center = hr[..., :sh, :sw]
            sr_y = shave_tensor(rgb_to_y_channel(sr), cfg.shave_border)
            hr_y = shave_tensor(rgb_to_y_channel(hr_center), cfg.shave_border)
            rows.append({
                "name": hr_path.stem,
                "psnr": compute_psnr(sr_y, hr_y),
                "ssim": compute_ssim(sr_y, hr_y),
            })
    return rows


def evaluate_benchmarks(model, cfg: PromptSRConfig, output_csv: str | None = None):
    device = next(model.parameters()).device
    root = Path(cfg.benchmark_root)
    results = {}
    for name in ["Set5", "Set14", "BSD100", "Urban100", "Manga109"]:
        dataset_dir = root / name
        if dataset_dir.exists():
            rows = evaluate_dataset(model, dataset_dir, cfg, device)
            if rows:
                results[name] = rows
    if output_csv is not None:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset", "name", "psnr", "ssim"])
            for dataset, rows in results.items():
                for row in rows:
                    writer.writerow([dataset, row["name"], row["psnr"], row["ssim"]])
    return results
