# Multi-seed statistical validation

在协议先稳定后，用这个脚本批量跑多 seed 训练、benchmark 和统计汇总。

## 典型命令

```bash
uv run python scripts/multi_seed_eval.py \
  --train-mode x4_finetune \
  --base-output outputs/multi-seed-atd-csa \
  --init-checkpoint outputs/paper-x2/checkpoints/last.ckpt \
  --seeds 42 43 44 \
  --batch-size 128 \
  --val-batch-size 32 \
  --num-workers 12 \
  --check-val-every-n-epoch 10 \
  --log-every-n-steps 50 \
  --checkpoint-every-n-train-steps 200 \
  --results-dir outputs/multi-seed-eval
```

## 只做统计，不重跑训练

```bash
uv run python scripts/multi_seed_eval.py \
  --train-mode x4_finetune \
  --base-output outputs/multi-seed-atd-csa \
  --seeds 42 43 44 \
  --results-dir outputs/multi-seed-eval \
  --skip-train
```

## 产物

- `outputs/multi-seed-eval/seed-<seed>.csv`：每个 seed 的 benchmark 明细
- `outputs/multi-seed-eval/summary.csv`：按数据集聚合后的 mean/std、与论文差值、是否落在 2σ 内

## 判定口径

脚本内置论文 x4 表：

- Set5: 32.56 / 0.8993
- Set14: 28.91 / 0.7889
- BSD100: 27.78 / 0.7435
- Urban100: 27.02 / 0.8116
- Manga109: 31.50 / 0.9198

汇总时会计算：

- `mean_psnr`, `std_psnr`
- `mean_ssim`, `std_ssim`
- `delta_psnr`, `delta_ssim`
- `paper_psnr_in_2sigma`
- `paper_ssim_in_2sigma`
- `paper_both_in_2sigma`

注意：如果 `std≈0` 而均值和论文差很大，`paper_both_in_2sigma` 会直接是 `False`，这正是我们想要的怀疑性检验结果。

## 小步验证模式

先别一上来整夜跑。可以先用 1 个 seed + 很小的 `max-steps` 验证训练/评测/汇总链路：

```bash
uv run python scripts/multi_seed_eval.py \
  --train-mode x4_finetune \
  --base-output outputs/multi-seed-atd-csa-smoke \
  --init-checkpoint outputs/paper-x2/checkpoints/last.ckpt \
  --seeds 42 \
  --max-steps 5 \
  --results-dir outputs/multi-seed-eval-smoke
```
