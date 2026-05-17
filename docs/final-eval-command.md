# Final Evaluation Command

After x4 training finishes, run:

```bash
uv run python scripts/evaluate_benchmarks.py \
  --checkpoint outputs/paper-x4/checkpoints/last.ckpt \
  --benchmark-root data/benchmarks \
  --output-csv outputs/eval/paper-x4-results.csv
```

If you want to test the latest best/epoch checkpoint manually, replace `last.ckpt` with the desired file.
