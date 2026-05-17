from __future__ import annotations

import argparse

from promptsr.config import PromptSRConfig
from promptsr.workflow import run_training


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='outputs/notebook-run')
    parser.add_argument('--max-steps', type=int, default=4)
    parser.add_argument('--train-mode', default='x4_finetune', choices=['x2_scratch', 'x3_finetune', 'x4_finetune'])
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--init-checkpoint', default='')
    return parser


def main():
    args = build_parser().parse_args()
    lr = 5e-4 if args.train_mode == 'x2_scratch' else 2e-4
    cfg = PromptSRConfig(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_mode=args.train_mode,
        lr=lr,
        init_checkpoint=args.init_checkpoint,
    )
    run_training(cfg, max_steps=args.max_steps)


if __name__ == '__main__':
    main()
