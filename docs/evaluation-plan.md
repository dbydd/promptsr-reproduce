# PromptSR 评测计划

## 目标
在以下 benchmark 上按论文协议做 Y-channel PSNR/SSIM 评测：

- Set5
- Set14
- BSD100
- Urban100
- Manga109

## 当前状态
- Y-channel metric 已接到 Lightning validation
- 独立 benchmark evaluation 脚本已补上基础版：`scripts/evaluate_benchmarks.py`
- benchmark 自动下载链路已初步建立，但镜像可靠性和 Manga109 合规来源还需要继续确认

## 下一步
1. 拉齐 Set5 / Set14 / BSD100 / Urban100 / Manga109 的实际目录内容。
2. 补更严格的 SSIM 实现与论文常用 shave/eval 细节。
3. 为各 benchmark 输出 CSV / markdown 汇总表。
4. 在 notebook 中给出论文表格对比。
