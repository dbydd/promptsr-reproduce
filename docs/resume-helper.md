# Long-run resume helper

```python
from promptsr.checkpoint_utils import latest_checkpoint
ckpt = latest_checkpoint('outputs/paper-x2')
```

Then start x4 finetune:

```bash
uv run python scripts/run_paper_training.py \
  --train-mode x4_finetune \
  --output-dir outputs/paper-x4 \
  --init-checkpoint "$ckpt"
```
