# PromptSR 论文细读笔记

## 真实目录
- 工作目录：`/Users/dbydd/Documents/vibe-notes/groupshare/20260514`
- 论文 PDF：`Liu 等 - 2025 - PromptSR Cascade Prompting for Lightweight Image Super-Resolution.pdf`

## 已确认结构
- 整体分三部分：
  1. shallow feature encoder
  2. deep feature extraction
  3. pixel-shuffle decoder
- 输入 `I_LR -> X0`
- deep feature extraction 由 `K` 个 residual groups 组成
- 每个 RG 内有 `N` 个 CPB，RG 尾部接一个 `3x3 conv`
- 最终输出：`I_SR = decoder(X_df + X0)`

## baseline PromptSR 超参
- RG 数 `K = 4`
- 每个 RG 的 CPB 数 `N = 3`
- PromptSR* 用 `N = 4`
- channel dim `C = 48`
- attention heads `4`
- MLP expansion ratio `1`
- downscale ratio `d = 8`
- AP update weight `alpha = 0.01`
- WSA window size `16`
- CSA sub-category size `128`

## CPB 总体
- 一个 CPB = `GAPL + LPL(coarse) + LPL(fine)` 的级联
- 设计目标：从 global-to-local、coarse-to-fine 提升感受野

## GAPL
### 1. Anchor Prompt Generation
- 输入特征 `X ∈ R^{HW × C}`
- 线性投影：
  - `Q = XW^Q`
  - `K = XW^K`
  - `V = XW^V`
- 先把 `X` 下采样成 `X_d ∈ R^{H/d × W/d × C}`
- 再投影为 anchor：
  - `A = X_d W^A`
- 粗相似图：
  - `M_coarse = A K^T / sqrt(C)`
  - shape: `HW/d^2 × HW`
- anchor prompts：
  - `P = softmax(M_coarse) · V`

### 2. Anchor Prompt Update
- AP 只在同一个 RG 内跨 CPB 更新，不跨 RG
- 更新式：
  - `P^i = alpha * P^(i-1) + (1 - alpha) * P^i`

### 3. Fine Prompting in GAPL
- 从 prompt 继续生成：
  - `K_p = P W^{K_p}`
  - `V_p = P W^{V_p}`
- 细相似图：
  - `M_fine = Q (K_p + A)^T / sqrt(C)`
  - shape: `HW × HW/d^2`
- prompted feature：
  - `X_p = softmax(M_fine) · V_p`
- 再接残差与 FFN：
  - `X_p = X_p + X`
  - `X_p = FFN(LN(X_p)) + X_p`

## LPL
- 两层 LPL，结构相同
- 每层都结合：
  - WSA（固定窗口）
  - CSA（基于相似图形成不规则类别窗口）

### coarse prompting
- 分类依据：`argmax_k(M_coarse^{kj}) = i`
- 公式：
  - `X_coarse = WSA(X_p) + CSA(X_p, M_coarse) + X_p`
  - `X_coarse = FFN(LN(X_coarse)) + X_coarse`

### fine prompting
- 分类依据沿不同维度取最大值，论文文字这里有一点书写混乱，但意图明确：使用 `M_fine`
- 公式：
  - `X_fine = WSA(X_coarse) + CSA(X_coarse, M_fine) + X_coarse`
  - `X_fine = FFN(LN(X_fine)) + X_fine`

## 训练设置
- loss：只用 `L1`
- 数据：DIV2K
  - 800 train
  - 100 test
  - 最后 100 val
- LR 由 bicubic downsampling 生成
- benchmark：
  - Set5
  - Set14
  - BSD100
  - Urban100
  - Manga109
- 指标：YCbCr 的 Y 通道上 PSNR / SSIM
- 优化器：Adam
  - `beta1=0.9`
  - `beta2=0.999`
  - `eps=1e-8`
- batch size：48
- crop size：`64x64`
- augmentation：random flip + rotation

## 训练日程
- x2：从头训练 500K iter
- x3 / x4：从已训练好的 x2 微调 250K iter
- x2 学习率：
  - init `5e-4`
  - halve at `[250K, 400K, 450K, 475K, 490K]`
- x3 / x4 学习率：
  - init `2e-4`
  - halve at `[150K, 200K, 225K, 240K]`

## 参数与复杂度摘录
- PromptSR x4：
  - params: `779K`
  - multi-adds: `53.5G`
  - Urban100: `27.02 / 0.8116`
  - Manga109: `31.50 / 0.9198`

## ablation 摘录
- 三重 prompting 都有效：anchor / coarse / fine 全开最好
- `d=8` 最优
- `alpha=0.01` 在 Manga109 最优
- `window size=16` 优于 `8`

## 当前还没精确落地到代码的点
1. 论文里的 WSA / CSA 具体张量实现细节还没完整写进当前代码
2. 当前工程里的 GAPL 还是近似实现，不是严格公式版
3. 当前工程还没有显式 residual group 抽象
4. 当前验证指标还没切到 Y-channel PSNR/SSIM
5. 当前训练调度还是 epoch 风格，不是论文的 iteration milestone 风格
