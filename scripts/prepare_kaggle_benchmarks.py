from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_lr(hr_path: Path, lr_path: Path, scale: int):
    img = Image.open(hr_path).convert('RGB')
    w, h = img.size
    lr = img.resize((w // scale, h // scale), Image.BICUBIC)
    ensure_dir(lr_path.parent)
    lr.save(lr_path)


def build_dataset(src_dir: Path, dst_root: Path, scale: int):
    hr_dir = dst_root / 'HR'
    lr_dir = dst_root / 'LR_bicubic' / f'X{scale}'
    ensure_dir(hr_dir)
    ensure_dir(lr_dir)
    for src in sorted(src_dir.iterdir()):
        if not src.is_file():
            continue
        if src.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}:
            continue
        hr_name = src.name
        lr_name = f"{src.stem}x{scale}{src.suffix}"
        hr_dst = hr_dir / hr_name
        if not hr_dst.exists():
            hr_dst.write_bytes(src.read_bytes())
        lr_dst = lr_dir / lr_name
        if not lr_dst.exists():
            save_lr(src, lr_dst, scale)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-root', default='data/kaggle-benchmarks/super-resolution/SR_testing_datasets')
    parser.add_argument('--dst-root', default='data/benchmarks')
    parser.add_argument('--scale', type=int, default=4)
    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    for name in ['Set5', 'Set14', 'BSDS100', 'Urban100', 'Manga109']:
        src_name = 'BSDS100' if name == 'BSDS100' else name
        src_dir = src_root / src_name
        if not src_dir.exists():
            print(f'skip missing {src_dir}')
            continue
        dst_name = 'BSD100' if name == 'BSDS100' else name
        build_dataset(src_dir, dst_root / dst_name, args.scale)
        print(f'prepared {dst_name}')


if __name__ == '__main__':
    main()
