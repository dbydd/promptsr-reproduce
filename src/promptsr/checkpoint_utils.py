from __future__ import annotations

from pathlib import Path


def latest_checkpoint(output_dir: str) -> str:
    ckpt_dir = Path(output_dir) / 'checkpoints'
    candidates = sorted(ckpt_dir.glob('*.ckpt'), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0]) if candidates else ''
