"""재현성 — 시드 고정 + 실행 스냅샷(설정/git hash/pip freeze) 기록."""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (1 << 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def snapshot_run(run_dir: str | Path, cfg: dict) -> None:
    """run 디렉토리에 config 사본, git rev, pip freeze 저장 (실패해도 학습은 계속)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    for name, cmd in (
        ("git_rev.txt", ["git", "rev-parse", "HEAD"]),
        ("pip_freeze.txt", [sys.executable, "-m", "pip", "freeze"]),
    ):
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60, check=False,
                cwd=str(Path(__file__).resolve().parents[3]),
            ).stdout
            (run_dir / name).write_text(out, encoding="utf-8")
        except Exception:
            pass
