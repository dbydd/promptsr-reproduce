# PromptSR Reproduction Status

## Final checkpoint used
- `outputs/paper-x4-fast/checkpoints/last.ckpt`
- x4 fast run reached 100 epochs.

## Final benchmark snapshot
| Dataset | Ours PSNR | Ours SSIM | Paper PSNR | Paper SSIM | Delta PSNR |
|---|---:|---:|---:|---:|---:|
| Set5 | 29.1183 | 0.9800 | 32.56 | 0.8993 | -3.4417 |
| Set14 | 26.0318 | 0.9476 | 28.91 | 0.7889 | -2.8782 |
| BSD100 | 25.8429 | 0.9330 | 27.78 | 0.7435 | -1.9371 |
| Urban100 | 23.6125 | 0.9346 | 27.02 | 0.8116 | -3.4075 |
| Manga109 | 21.9403 | 0.9266 | 31.50 | 0.9198 | -9.5597 |

## Conclusion
- Engineering reproduction chain is complete: data -> training -> x2->x4 handoff -> benchmark works.
- Paper-level metric reproduction is NOT achieved yet.
- Biggest remaining explanation candidates:
  1. CSA is still only a lightweight approximation, not the true paper/ATD-style implementation.
  2. Evaluation protocol is still not fully aligned with standard SR implementations.
  3. The current training length (100 epochs) is far below the paper's 500K + 250K iteration schedule.
  4. Some architecture stabilizations and throughput-oriented edits likely changed the effective model behavior.
