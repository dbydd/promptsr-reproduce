# PromptSR Reproduction Gap Analysis

## 主要不相似点
1. **CSA 实现仍是近似**：当前是可运行的 similarity-map guided attention 近似，不是论文中更精确的 category-based self-attention / categorization / uncategorization 路径。
2. **训练协议长度不一致**：论文是 x2 500K iterations + x4 250K iterations，当前工程是 epoch 驱动的收敛跑法，训练总步数远不足论文量级。
3. **评测协议不够标准化**：当前 PSNR/SSIM 评测是自实现版本，虽已对齐 Y-channel/shave/crop 的基本逻辑，但仍未完全对齐最常用 SR benchmark 复现实践。
4. **工程稳定化改动可能影响原模型行为**：为了 MPS 兼容和数值稳定，当前模型引入了 residual scaling、简化 attention projection 等改写。

## 下一步优先级
1. 先把评测协议再严格一点，减少“算分方式”差异。
2. 继续提高训练吞吐，但不要再改变会明显破坏模型语义的结构。
3. 如果还想继续逼近论文指标，下一次最大改动点应该是 CSA 的精确实现，而不是再堆 batch。
