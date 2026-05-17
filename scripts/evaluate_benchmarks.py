from __future__ import annotations

import argparse
from pathlib import Path

import torch

from promptsr.config import PromptSRConfig
from promptsr.eval import evaluate_benchmarks
from promptsr.workflow import run_training


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--output-csv', default='outputs/eval/results.csv')
    parser.add_argument('--benchmark-root', default='data/benchmarks')
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--quick-train-steps', type=int, default=0)
    return parser


def main():
    args = build_parser().parse_args()
    cfg = PromptSRConfig(scale=args.scale, benchmark_root=args.benchmark_root, batch_size=1, num_workers=0)

    if args.checkpoint:
        from promptsr.lit_module import PromptSRLightningModule
        module = PromptSRLightningModule.load_from_checkpoint(args.checkpoint, cfg=cfg, strict=False)
        model = module.model
    elif args.quick_train_steps > 0:
        trainer, module = run_training(cfg, max_steps=args.quick_train_steps)
        model = module.model
    else:
        raise SystemExit('Need --checkpoint or --quick-train-steps')

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)
    results = evaluate_benchmarks(model, cfg, output_csv=args.output_csv)
    for dataset, rows in results.items():
        if not rows:
            continue
        mean_psnr = sum(r['psnr'] for r in rows) / len(rows)
        mean_ssim = sum(r['ssim'] for r in rows) / len(rows)
        print(dataset, f'PSNR={mean_psnr:.4f}', f'SSIM={mean_ssim:.4f}', f'N={len(rows)}')


if __name__ == '__main__':
    main()
