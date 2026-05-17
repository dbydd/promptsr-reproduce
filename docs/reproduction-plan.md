# PromptSR 复现计划

> 目标：把 `promptsr-reproduce` 从“可运行骨架”推进到“论文细节对齐的 notebook-first 复现工程”。

## 用户长期约定
- 主入口优先是 Jupyter notebook，而不是 CLI。
- Lightning 用官方推荐的组织方式：DataModule / ModelModule 分离；训练编排尽量外置；TensorBoard 记录实验。
- 非严肃约束、实验说明、假设、待确认点，优先写成 Markdown 文件，作为上下文与工程文档的一部分。

## 当前已确认事实
1. 论文：`PromptSR: Cascade Prompting for Lightweight Image Super-Resolution`
2. 上游 GitHub 仓库 `wenyang001/PromptSR` 目前只有空 README 和 LICENSE，没有源码可对照。
3. 本地已完成：
   - uv 环境
   - Lightning 最小训练骨架
   - 最小数据生成脚本
   - 1 epoch MPS 训练跑通

## 本轮要做的事
1. 深挖论文公开文本，尽量提取：
   - 网络总结构
   - CPB / GAPL / LPL 的更细描述
   - 训练策略与评测协议
   - 参数量 / 计算量 / 对比基线
2. 把工程改成 notebook-first：
   - `notebooks/` 下放主实验入口
   - CLI 仅保留兼容或调试用途
3. 按 Lightning 最佳实践重构：
   - data module
   - pure model module
   - task/lightning wrapper
   - trainer factory / optimizer factory
   - TensorBoard logger
4. 把“已确认 / 未确认 / 假设”落到 markdown 文档，避免实现细节混杂。

## 关键风险
- 没有作者代码，很多公式和超参只能从论文文字反推。
- 如果拿不到足够完整的论文正文，当前复现只能是 faithful approximation，不是精确复刻。
- 目前工程已经先写了代码，再补更严格测试；后续补细节时要把回归验证补齐。

## 交付物
- `docs/paper-notes.md`：论文细读笔记
- `docs/open-questions.md`：未确认点和假设
- `notebooks/`：主实验 notebook
- 更贴近 Lightning 最佳实践的代码结构
- TensorBoard 训练日志
