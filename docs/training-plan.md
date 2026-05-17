# Training Plan

## Paper-aligned schedule

### x2 scratch
- train mode: `x2_scratch`
- lr: `5e-4`
- milestones: `250K, 400K, 450K, 475K, 490K`
- total iters: `500K`

### x3/x4 finetune
- train mode: `x3_finetune` or `x4_finetune`
- lr: `2e-4`
- milestones: `150K, 200K, 225K, 240K`
- total iters: `250K`

## Current implementation status
- optimizer: Adam (paper-aligned)
- scheduler: MultiStepLR with paper milestones
- checkpoint: TensorBoard + last/best checkpoint enabled
- notebook-first runner: `scripts/run_notebook_workflow.py`
- supports `--init-checkpoint` for x2 -> x4 handoff

## Recommended run order
1. x2 scratch
2. save checkpoint
3. x4 finetune with `--init-checkpoint <x2_ckpt>`
4. run benchmark evaluation

## Remaining gap to full reproduction
- no full long-run checkpoint yet
- CSA is still a lightweight approximation, not a line-by-line ATD-style implementation
- final paper-level metrics still need long training evidence
