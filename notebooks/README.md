# PromptSR reproduction notebook

主入口 notebook。按用户偏好，配置、数据准备、训练、日志检查都集中在这里。

## 建议流程
1. 先运行 `scripts/download_data.py` 准备 DIV2K 和 benchmark。
2. 在 notebook 里构造 `PromptSRConfig`。
3. 调 `run_training(cfg, max_steps=...)` 做短跑验证或正式训练。
4. 用 TensorBoard 查看 `outputs/.../tensorboard/`。

## 最小示例
```python
from promptsr.config import PromptSRConfig
from promptsr.workflow import run_training

cfg = PromptSRConfig(
    scale=4,
    batch_size=1,
    patch_size=64,
    num_workers=0,
    output_dir='outputs/notebook-run',
    train_mode='x4_finetune',
)
trainer, model = run_training(cfg, max_steps=4)
```
