# Evaluation protocol update log

Date: 2026-05-17
Branch merged: `experiment` -> `main`

## What changed

### 1. Benchmark SSIM implementation was replaced
Before:
- `src/promptsr/eval.py` used a custom global-statistics `compute_ssim_like()` implementation.
- This implementation does **not** match standard SR benchmark SSIM practice.

After:
- Replaced with a more standard SSIM implementation using:
  - 11x11 Gaussian window
  - sigma=1.5
  - valid convolution
  - Y-channel evaluation
  - crop border applied before metric computation

### 2. Validation and benchmark protocol were aligned better
Before:
- `validation_step()` in `src/promptsr/lit_module.py` computed Y-channel metrics **without** crop/shave.
- `eval.py` used Y-channel **with** `shave_border`.
- Therefore training-time validation numbers and benchmark numbers were not on the same protocol.

After:
- Validation now applies `shave_tensor(..., cfg.shave_border)` to Y-channel tensors before PSNR/SSIM updates.
- This makes training-time validation closer to final benchmark protocol.

## Why this matters

Previously observed symptom:
- PSNR looked low
- SSIM looked unusually high

After the protocol fix:
- PSNR remained almost unchanged
- SSIM dropped significantly on re-evaluation

Interpretation:
- The old SSIM values were inflated by a non-standard metric implementation.
- Earlier SSIM comparisons against the paper were not reliable.

## Re-evaluation summary

### Rank-1 q/k/v full checkpoint (`outputs/paper-x4-csa-qkv-rank1-full/checkpoints/last.ckpt`)

Old benchmark result:
- Set5: 28.3323 / 0.9753
- Set14: 25.6289 / 0.9438
- BSD100: 25.5891 / 0.9286
- Urban100: 23.1977 / 0.9289
- Manga109: 21.9664 / 0.9265

Re-evaluated with updated protocol:
- Set5: 28.3320 / 0.7974
- Set14: 25.6289 / 0.6852
- BSD100: 25.5890 / 0.6573
- Urban100: 23.1976 / 0.6563
- Manga109: 21.9664 / 0.7026

### Current baseline checkpoint (`outputs/checkpoints/last.ckpt`)
Re-evaluated:
- Set5: 10.9036 / 0.1317
- Set14: 10.5514 / 0.0879
- BSD100: 11.3102 / 0.1007
- Urban100: 10.3459 / 0.0946
- Manga109: 8.1976 / 0.0663

## Bottom line

This merge does **not** make the model closer to the paper by itself.
It fixes the evaluation protocol so that future comparisons are less misleading.

The main confirmed outcome is:
- Old SSIM numbers were not trustworthy.
- Future protocol should use the updated implementation.
