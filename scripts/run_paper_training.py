from __future__ import annotations

import argparse

from promptsr.config import PromptSRConfig
from promptsr.workflow import run_training


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--train-mode', required=True, choices=['x2_scratch', 'x3_finetune', 'x4_finetune'])
    parser.add_argument('--max-steps', type=int, default=-1)
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--val-batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--init-checkpoint', default='')
    parser.add_argument('--check-val-every-n-epoch', type=int, default=5)
    parser.add_argument('--log-every-n-steps', type=int, default=10)
    parser.add_argument('--checkpoint-every-n-train-steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-hr-dir', default='data/DIV2K_train_HR')
    parser.add_argument('--train-lr-dir', default='')
    parser.add_argument('--val-hr-dir', default='data/DIV2K_valid_HR')
    parser.add_argument('--val-lr-dir', default='')
    return parser


def infer_lr_dir(train_mode: str, override: str) -> str:
    if override:
        return override
    if train_mode == 'x2_scratch':
        return 'data/DIV2K_train_LR_bicubic/X2'
    if train_mode == 'x3_finetune':
        return 'data/DIV2K_train_LR_bicubic/X3'
    return 'data/DIV2K_train_LR_bicubic/X4'


def infer_val_lr_dir(train_mode: str, override: str) -> str:
    if override:
        return override
    if train_mode == 'x2_scratch':
        return 'data/DIV2K_valid_LR_bicubic/X2'
    if train_mode == 'x3_finetune':
        return 'data/DIV2K_valid_LR_bicubic/X3'
    return 'data/DIV2K_valid_LR_bicubic/X4'


def infer_scale(train_mode: str) -> int:
    if train_mode == 'x2_scratch':
        return 2
    if train_mode == 'x3_finetune':
        return 3
    return 4


def main():
    args = build_parser().parse_args()
    scale = infer_scale(args.train_mode)
    lr = 5e-4 if args.train_mode == 'x2_scratch' else 2e-4
    cfg = PromptSRConfig(
        scale=scale,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        train_mode=args.train_mode,
        lr=lr,
        init_checkpoint=args.init_checkpoint,
        train_hr_dir=args.train_hr_dir,
        train_lr_dir=infer_lr_dir(args.train_mode, args.train_lr_dir),
        val_hr_dir=args.val_hr_dir,
        val_lr_dir=infer_val_lr_dir(args.train_mode, args.val_lr_dir),
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        log_every_n_steps=args.log_every_n_steps,
        checkpoint_every_n_train_steps=args.checkpoint_every_n_train_steps,
        seed=args.seed,
    )
    run_training(cfg, max_steps=None if args.max_steps < 0 else args.max_steps)


if __name__ == '__main__':
    main()
