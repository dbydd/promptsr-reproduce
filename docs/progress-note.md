# 继续推进记录

## 本轮新增
- `src/promptsr/eval.py`
- `scripts/evaluate_benchmarks.py`
- `docs/reproduction-scope.md`
- `docs/data-preparation.md`
- `docs/evaluation-plan.md`

## 新发现的问题
1. benchmark 下载脚本里 Hugging Face 镜像 `keanteng/bsd100-set5-set14` 当前下载失败，需要换镜像或改为更稳的直链。
2. 由于模型结构已经变更，旧 checkpoint 无法直接加载到新结构上。
3. 快速训练 1 step 曾出现 `train/loss_step=nan`。根因是 CPB/LPL 内连续无缩放残差叠加导致特征幅度逐组指数级放大。已通过残差分支缩放临时修复，1-step 训练 fresh verification 已无 NaN。

## 现在的优先级
1. 补稳定 benchmark 数据来源。
2. 完成正式 benchmark evaluation。
3. 继续把残差缩放参数、CSA 实现和参数量调到更贴论文。
4. 再进入更长训练。
