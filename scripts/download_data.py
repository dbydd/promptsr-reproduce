from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path

DIV2K_URLS = {
    "DIV2K_train_HR.zip": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "DIV2K_valid_HR.zip": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
    "DIV2K_train_LR_bicubic_X2.zip": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip",
    "DIV2K_train_LR_bicubic_X3.zip": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X3.zip",
    "DIV2K_train_LR_bicubic_X4.zip": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip",
    "DIV2K_valid_LR_bicubic_X2.zip": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip",
    "DIV2K_valid_LR_bicubic_X3.zip": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X3.zip",
    "DIV2K_valid_LR_bicubic_X4.zip": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip",
}

BENCHMARK_HF = {
    "Set5.zip": ("keanteng/bsd100-set5-set14", "Set5.zip"),
    "Set14.zip": ("keanteng/bsd100-set5-set14", "Set14.zip"),
    "BSD100.zip": ("keanteng/bsd100-set5-set14", "BSD100.zip"),
}


def run(cmd: list[str], cwd: Path | None = None):
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def aria2_download(url: str, output: Path, connections: int):
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        print(f"skip existing {output}")
        return
    run([
        "aria2c",
        "-x", str(connections),
        "-s", str(connections),
        "-k", "1M",
        "--continue=true",
        "--max-tries=10",
        "--retry-wait=5",
        "-o", output.name,
        url,
    ], cwd=output.parent)


def unzip_file(zip_path: Path, target_dir: Path):
    marker = target_dir / f".{zip_path.stem}.extracted"
    if marker.exists():
        print(f"skip extracted {zip_path.name}")
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
    marker.write_text("ok\n")


def download_div2k(data_root: Path, scale: int, connections: int):
    archives = data_root / "archives"
    names = [
        "DIV2K_train_HR.zip",
        "DIV2K_valid_HR.zip",
        f"DIV2K_train_LR_bicubic_X{scale}.zip",
        f"DIV2K_valid_LR_bicubic_X{scale}.zip",
    ]
    for name in names:
        aria2_download(DIV2K_URLS[name], archives / name, connections)
    for name in names:
        unzip_file(archives / name, data_root)


def download_hf_file(repo_id: str, filename: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "huggingface-cli",
        "download",
        repo_id,
        filename,
        "--repo-type", "dataset",
        "--local-dir", str(output_dir),
    ]
    run(cmd)


def download_benchmarks(data_root: Path):
    archives = data_root / "archives" / "benchmarks"
    for local_name, (repo_id, filename) in BENCHMARK_HF.items():
        target = archives / filename
        if not target.exists():
            download_hf_file(repo_id, filename, archives)
        if target.exists():
            unzip_file(target, data_root / "benchmarks")
        else:
            print(f"missing {target}; download manually if huggingface mirror changed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--scale", type=int, default=4, choices=[2, 3, 4])
    parser.add_argument("--connections", type=int, default=16)
    parser.add_argument("--div2k", action="store_true")
    parser.add_argument("--benchmarks", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not args.div2k and not args.benchmarks:
        args.div2k = True
        args.benchmarks = True
    if args.div2k:
        download_div2k(data_root, args.scale, args.connections)
    if args.benchmarks:
        download_benchmarks(data_root)


if __name__ == "__main__":
    main()
