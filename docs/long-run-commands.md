# Long-run Commands

## x2 scratch
```bash
uv run python scripts/run_paper_training.py \
  --train-mode x2_scratch \
  --output-dir outputs/paper-x2
```

## x4 finetune from x2 checkpoint
```bash
uv run python scripts/run_paper_training.py \
  --train-mode x4_finetune \
  --output-dir outputs/paper-x4 \
  --init-checkpoint outputs/paper-x2/checkpoints/last.ckpt
```

## Notes
- `max_steps` 默认不设上限，由 Lightning 按外部停止条件/中断控制。
- 如果要做短跑验证，可加：`--max-steps 100`。
- x2 使用 `DIV2K_train_LR_bicubic/X2` 与 `DIV2K_valid_LR_bicubic/X2`。
- x4 finetune 使用 `DIV2K_train_LR_bicubic/X4` 与 `DIV2K_valid_LR_bicubic/X4`。
