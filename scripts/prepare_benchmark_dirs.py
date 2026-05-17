from __future__ import annotations

from pathlib import Path

from PIL import Image
import numpy as np


def main():
    root = Path('data')
    root.mkdir(exist_ok=True)
    bench = root / 'benchmarks'
    for name in ['Set5', 'Set14', 'BSD100', 'Urban100', 'Manga109']:
        (bench / name).mkdir(parents=True, exist_ok=True)
    print('benchmark directories ready under data/benchmarks')


if __name__ == '__main__':
    main()
