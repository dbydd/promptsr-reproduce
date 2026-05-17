# 数据准备

## 自动下载

### 训练数据 + benchmark
```bash
uv run python scripts/download_data.py --scale 4 --connections 16
```

### 只下载 DIV2K
```bash
uv run python scripts/download_data.py --div2k --scale 4 --connections 16
```

### 只下载 benchmark
```bash
uv run python scripts/download_data.py --benchmarks
```

## 约定目录
- `data/DIV2K_train_HR/`
- `data/DIV2K_train_LR_bicubic/X4/`
- `data/DIV2K_valid_HR/`
- `data/DIV2K_valid_LR_bicubic/X4/`
- `data/benchmarks/Set5/`
- `data/benchmarks/Set14/`
- `data/benchmarks/BSD100/`
- `data/benchmarks/Urban100/`
- `data/benchmarks/Manga109/`

## 说明
- DIV2K 默认从 ETH 官方站用 `aria2c` 多线程下载。
- 部分 benchmark 先尝试 Hugging Face 镜像。
- 如果某些镜像失效，脚本会保留已下载内容，并提示手动补齐。
