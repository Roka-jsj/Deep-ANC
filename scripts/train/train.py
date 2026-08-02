#!/usr/bin/env python3
"""학습 진입점.

  python scripts/train/train.py --config configs/train_pretrain.yaml
  torchrun --nproc_per_node=2 scripts/train/train.py --config configs/train_pretrain.yaml
  python scripts/train/train.py --config configs/train_finetune.yaml --set stage=closed_loop
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.config import load_train_config          # noqa: E402
from deep_anc.train.trainer import Trainer             # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="재개할 체크포인트 경로")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        help="설정 오버라이드 (예: --set batch_size=4 --set optimizer.lr=1e-4)",
    )
    args = parser.parse_args()

    cfg = load_train_config(args.config, args.overrides)
    if args.resume:
        cfg["resume"] = args.resume

    Trainer(cfg).train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
