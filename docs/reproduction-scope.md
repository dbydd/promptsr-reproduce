# PromptSR 完整复现范围

## 目标
把本工程整理成 notebook-first 的 PromptSR 复现实验工程：

1. 可自动准备训练/评测数据。
2. 可在 notebook 中启动训练、查看 TensorBoard 日志、保存 checkpoint。
3. 可在 benchmark 数据集上按 Y-channel PSNR/SSIM 评测。
4. 模型结构尽量按论文公式实现：RG / CPB / GAPL / coarse LPL / fine LPL。

## “完整复现”的定义
由于作者 GitHub 仓库目前没有源码，完整复现不等于逐行复刻官方实现。这里定义为：

- 论文中明确写出的结构和训练协议必须落地。
- 论文没有展开的实现细节，必须在 `docs/open-questions.md` 写清楚当前假设。
- 每一步必须可运行、可验证。

## 训练数据
论文训练只用 DIV2K：

- `DIV2K_train_HR.zip`
- `DIV2K_train_LR_bicubic_X2.zip`
- `DIV2K_train_LR_bicubic_X3.zip`
- `DIV2K_train_LR_bicubic_X4.zip`

默认用 aria2c 多线程下载。若源站不可达，可切换到 Hugging Face / Kaggle 镜像。

## 评测数据
论文 benchmark：

- Set5
- Set14
- BSD100
- Urban100
- Manga109

当前工程先提供脚本化下载入口和目录规范。若公开直链不可用，会保留手动/镜像下载说明。

## Notebook 工作流
主入口：

- `notebooks/promptsr_reproduction.ipynb`
- `notebooks/promptsr_evaluation.ipynb`

CLI 只作为兼容和自动化验证入口，不作为主要交互方式。

## 验证层级
1. 小数据前向与最小训练：证明代码没坏。
2. 标准数据下载与目录检查：证明数据就绪。
3. DIV2K 短训练：证明训练流程可跑。
4. benchmark evaluation：证明评测流程可跑。
5. 长训结果：和论文表格对齐。

## 当前已知近似点
- WSA 当前用局部卷积近似，还不是严格 window multi-head self-attention。
- CSA 当前是基于 similarity map 的全局聚合近似，还不是 ATD 论文中的高性能分类窗口实现。
- 这些会继续替换。
