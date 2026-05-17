# PromptSR Reproduce

基于论文 *PromptSR: Cascade Prompting for Lightweight Image Super-Resolution* 的复现骨架。

## 已知论文要点
- 核心模块：Cascade Prompting Block (CPB)
- CPB 由 3 个串联 prompting layer 组成：1 个 GAPL + 2 个 LPL
- 目标：在轻量 SR 场景下扩大感受野，同时保持计算量低

## 当前实现策略
- 用 PyTorch Lightning 组织训练流程
- 用 uv 管理依赖与环境
- 先实现可运行的最小复现骨架，再按论文补齐细节
- 默认数据接口按 DIV2K / benchmark 的常见 SR 结构设计

## 目录
- `src/promptsr/`：模型、数据、训练入口
- `data/`：数据根目录（不提交大文件）
- `outputs/`：实验输出

## 快速检查
```bash
uv sync
uv run python -m promptsr.train --help
```
