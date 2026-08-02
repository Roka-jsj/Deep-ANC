#!/usr/bin/env python3
"""duct.yaml 기하로 도메인 랜덤화 RIR 뱅크 생성.

  python scripts/data/build_rir_bank.py --n 300 --out data/rir_bank/duct_rirs_v1.npz
덕트 치수(duct.yaml)를 바꾸면 반드시 다시 실행해야 시뮬레이션에 반영된다.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.config import REPO_ROOT, validate_duct     # noqa: E402
from deep_anc.dsp.duct_sim import build_rir_bank         # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duct", default="configs/duct.yaml")
    parser.add_argument("--out", default="data/rir_bank/duct_rirs_v1.npz")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--ir-len", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()

    with open(REPO_ROOT / args.duct, encoding="utf-8") as f:
        duct = yaml.safe_load(f)
    validate_duct(duct)

    print(f"RIR 뱅크 생성: n={args.n}, ir_len={args.ir_len} @{args.sample_rate}Hz ...")
    bank = build_rir_bank(duct, args.sample_rate, n_variants=args.n, seed=args.seed, ir_len=args.ir_len)

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **bank, sample_rate=args.sample_rate, seed=args.seed)
    size_mb = out.stat().st_size / 1e6
    print(f"저장: {out} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
