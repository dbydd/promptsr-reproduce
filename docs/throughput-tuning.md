# Throughput Tuning Notes

## Current high-throughput profile
- x4 finetune can be run with:
  - `batch_size=96`
  - `val_batch_size=32`
  - `num_workers=8`
  - `check_val_every_n_epoch=5`
  - `log_every_n_steps=10`
  - `checkpoint_every_n_train_steps=100`

## More aggressive overnight profile for M5 Max
Try this if memory allows:

```bash
uv run python scripts/run_paper_training.py \
  --train-mode x4_finetune \
  --output-dir outputs/paper-x4-m5max \
  --batch-size 128 \
  --val-batch-size 32 \
  --num-workers 12 \
  --check-val-every-n-epoch 5 \
  --log-every-n-steps 10 \
  --checkpoint-every-n-train-steps 100 \
  --init-checkpoint outputs/paper-x4-fast/checkpoints/last.ckpt
```

If this OOMs or destabilizes, step back to batch 96 / workers 8.
