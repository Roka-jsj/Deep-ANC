"""체크포인트 저장/로드 — 모델+옵티마이저+스케줄러+step+RNG 완전 재개."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    best_metric: float,
    cfg: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if hasattr(model, "module") else model
    state = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": int(step),
        "best_metric": float(best_metric),
        "cfg": cfg,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)                              # 원자적 교체


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    restore_rng: bool = True,
    map_location: str = "cpu",
) -> dict:
    state = torch.load(Path(path), map_location=map_location, weights_only=False)
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if restore_rng and "rng" in state:
        rng = state["rng"]
        try:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"].cpu() if hasattr(rng["torch"], "cpu") else rng["torch"])
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
        except Exception:
            pass
    return state
