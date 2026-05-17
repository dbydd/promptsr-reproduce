# PromptSR 未确认点与当前假设

## 已确认
- 论文主结构、GAPL 关键公式、LPL 的 coarse/fine 公式
- baseline 主要超参
- 训练与评测协议主干
- 真实工作目录已切换到 `/Users/dbydd/Documents/vibe-notes/groupshare/20260514`

## 当前代码与论文的偏差
1. `src/promptsr/model.py` 里现在的 `GlobalAnchorPromptingLayer` 还是 prompt bank 近似写法，不是论文中的 `A/K/V -> M_coarse -> P -> M_fine -> X_p`。
2. `LocalPromptingLayer` 还是 depthwise conv 风格近似，不是 `WSA + CSA` 组合。
3. 还没有显式 `ResidualGroup`。
4. `PromptSRLightningModule.configure_optimizers()` 仍在模块内部定义优化器，这不符合你要求的最佳实践。
5. 训练入口仍以 CLI 为主，不符合 notebook-first。
6. 当前 metrics 还没切到 Y-channel PSNR / SSIM。
7. 当前 scheduler 不是论文里的 milestone halve schedule。

## 接下来必须补的
- 抽出纯模型层 / 任务层 / trainer factory / optimizer factory 的职责边界
- notebook 作为主实验入口
- TensorBoard 作为默认日志后端
- 论文公式版 GAPL
- 至少一个可读、可跑的 WSA+CSA 近似实现
- Y-channel 验证指标
- iteration-based lr schedule

## 实现策略
- 先不追求一次性做出高性能精确版
- 先做“结构和训练协议尽量贴论文、代码可读可跑、notebook 友好”的版本
- 再逐步替换近似实现为更精确实现
