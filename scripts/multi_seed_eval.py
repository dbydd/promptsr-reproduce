from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
from pathlib import Path
from typing import Any

PAPER = {
    "Set5": {"psnr": 32.56, "ssim": 0.8993},
    "Set14": {"psnr": 28.91, "ssim": 0.7889},
    "BSD100": {"psnr": 27.78, "ssim": 0.7435},
    "Urban100": {"psnr": 27.02, "ssim": 0.8116},
    "Manga109": {"psnr": 31.50, "ssim": 0.9198},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-mode", required=True, choices=["x2_scratch", "x3_finetune", "x4_finetune"])
    parser.add_argument("--base-output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--check-val-every-n-epoch", type=int, default=10)
    parser.add_argument("--log-every-n-steps", type=int, default=50)
    parser.add_argument("--checkpoint-every-n-train-steps", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--benchmark-root", default="data/benchmarks")
    parser.add_argument("--results-dir", default="outputs/multi-seed-eval")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--python", default="uv run python")
    return parser


def run_command(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def python_prefix(raw: str) -> list[str]:
    return raw.strip().split()


def seed_output_dir(base_output: Path, seed: int) -> Path:
    return base_output / f"seed-{seed}"


def seed_ckpt(seed_dir: Path) -> Path:
    return seed_dir / "checkpoints" / "last.ckpt"


def seed_csv(results_dir: Path, seed: int) -> Path:
    return results_dir / f"seed-{seed}.csv"


def train_one(args: argparse.Namespace, project_root: Path, seed: int) -> Path:
    out_dir = seed_output_dir(Path(args.base_output), seed)
    cmd = python_prefix(args.python) + [
        "scripts/run_paper_training.py",
        "--train-mode",
        args.train_mode,
        "--output-dir",
        str(out_dir),
        "--seed",
        str(seed),
        "--batch-size",
        str(args.batch_size),
        "--val-batch-size",
        str(args.val_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--check-val-every-n-epoch",
        str(args.check_val_every_n_epoch),
        "--log-every-n-steps",
        str(args.log_every_n_steps),
        "--checkpoint-every-n-train-steps",
        str(args.checkpoint_every_n_train_steps),
    ]
    if args.max_steps >= 0:
        cmd += ["--max-steps", str(args.max_steps)]
    if args.init_checkpoint:
        cmd += ["--init-checkpoint", args.init_checkpoint]
    run_command(cmd, cwd=project_root)
    ckpt = seed_ckpt(out_dir)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt}")
    return ckpt


def eval_one(args: argparse.Namespace, project_root: Path, seed: int, checkpoint: Path) -> Path:
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = seed_csv(results_dir, seed)
    cmd = python_prefix(args.python) + [
        "scripts/evaluate_benchmarks.py",
        "--checkpoint",
        str(checkpoint),
        "--benchmark-root",
        args.benchmark_root,
        "--output-csv",
        str(csv_path),
        "--scale",
        str(4 if args.train_mode == "x4_finetune" else 2 if args.train_mode == "x2_scratch" else 3),
    ]
    run_command(cmd, cwd=project_root)
    if not csv_path.exists():
        raise FileNotFoundError(f"eval csv missing: {csv_path}")
    return csv_path


def load_csv(path: Path) -> dict[str, dict[str, float]]:
    acc: dict[str, list[dict[str, float]]] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row["dataset"]
            acc.setdefault(dataset, []).append(
                {
                    "psnr": float(row["psnr"]),
                    "ssim": float(row["ssim"]),
                }
            )
    result: dict[str, dict[str, float]] = {}
    for dataset, rows in acc.items():
        result[dataset] = {
            "psnr": sum(r["psnr"] for r in rows) / len(rows),
            "ssim": sum(r["ssim"] for r in rows) / len(rows),
        }
    return result


def summarize(seed_results: dict[int, dict[str, dict[str, float]]]) -> list[dict[str, Any]]:
    datasets = sorted({dataset for result in seed_results.values() for dataset in result})
    table: list[dict[str, Any]] = []
    for dataset in datasets:
        psnrs = [seed_results[seed][dataset]["psnr"] for seed in seed_results if dataset in seed_results[seed]]
        ssims = [seed_results[seed][dataset]["ssim"] for seed in seed_results if dataset in seed_results[seed]]
        psnr_mean = statistics.mean(psnrs)
        ssim_mean = statistics.mean(ssims)
        psnr_std = statistics.stdev(psnrs) if len(psnrs) > 1 else 0.0
        ssim_std = statistics.stdev(ssims) if len(ssims) > 1 else 0.0
        paper_psnr = PAPER.get(dataset, {}).get("psnr")
        paper_ssim = PAPER.get(dataset, {}).get("ssim")
        psnr_in_2sigma = abs(paper_psnr - psnr_mean) <= 2 * psnr_std if paper_psnr is not None else False
        ssim_in_2sigma = abs(paper_ssim - ssim_mean) <= 2 * ssim_std if paper_ssim is not None else False
        table.append(
            {
                "dataset": dataset,
                "n": len(psnrs),
                "paper_psnr": paper_psnr,
                "paper_ssim": paper_ssim,
                "mean_psnr": psnr_mean,
                "std_psnr": psnr_std,
                "mean_ssim": ssim_mean,
                "std_ssim": ssim_std,
                "delta_psnr": None if paper_psnr is None else psnr_mean - paper_psnr,
                "delta_ssim": None if paper_ssim is None else ssim_mean - paper_ssim,
                "paper_psnr_in_2sigma": psnr_in_2sigma,
                "paper_ssim_in_2sigma": ssim_in_2sigma,
                "paper_both_in_2sigma": psnr_in_2sigma and ssim_in_2sigma,
            }
        )
    return table


def write_summary(table: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "n",
                "paper_psnr",
                "paper_ssim",
                "mean_psnr",
                "std_psnr",
                "mean_ssim",
                "std_ssim",
                "delta_psnr",
                "delta_ssim",
                "paper_psnr_in_2sigma",
                "paper_ssim_in_2sigma",
                "paper_both_in_2sigma",
            ],
        )
        writer.writeheader()
        writer.writerows(table)


def print_table(table: list[dict[str, Any]]) -> None:
    print(
        "dataset,n,paper_psnr,mean_psnr,std_psnr,delta_psnr,paper_ssim,mean_ssim,std_ssim,delta_ssim,both_in_2sigma"
    )
    for row in table:
        print(
            f"{row['dataset']},{row['n']},{row['paper_psnr']:.4f},{row['mean_psnr']:.4f},{row['std_psnr']:.4f},{row['delta_psnr']:.4f},"
            f"{row['paper_ssim']:.4f},{row['mean_ssim']:.4f},{row['std_ssim']:.4f},{row['delta_ssim']:.4f},{row['paper_both_in_2sigma']}"
        )


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    seed_results: dict[int, dict[str, dict[str, float]]] = {}

    for seed in args.seeds:
        out_dir = seed_output_dir(Path(args.base_output), seed)
        ckpt = seed_ckpt(out_dir)
        if args.skip_train:
            if not ckpt.exists():
                raise FileNotFoundError(f"--skip-train set but checkpoint missing: {ckpt}")
        else:
            ckpt = train_one(args, project_root, seed)
        csv_path = eval_one(args, project_root, seed, ckpt)
        seed_results[seed] = load_csv(csv_path)

    table = summarize(seed_results)
    summary_path = Path(args.results_dir) / "summary.csv"
    write_summary(table, summary_path)
    print_table(table)
    print(f"summary_csv={summary_path}")


if __name__ == "__main__":
    main()
