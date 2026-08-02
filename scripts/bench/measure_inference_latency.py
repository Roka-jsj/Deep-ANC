#!/usr/bin/env python3
"""엔진 스텝 지연 벤치마크 — 배포 게이트: block 256 에서 P99 < 3.0ms.

  python scripts/bench/measure_inference_latency.py --config configs/runtime.yaml --steps 1000
  python scripts/bench/measure_inference_latency.py --config configs/runtime.yaml --set engine.type=ort
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.config import load_runtime_config          # noqa: E402
from deep_anc.realtime.engines import build_engine       # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/runtime.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--gate-p99-ms", type=float, default=3.0)
    args = parser.parse_args()

    cfg = load_runtime_config(args.config, args.overrides)
    engine = build_engine(cfg)
    hop = engine.hop
    budget_ms = 1000.0 * hop / int(cfg["hardware"]["audio"]["sample_rate"])

    rng = np.random.default_rng(0)
    ref = (rng.standard_normal(hop) * 0.02).astype(np.float32)
    err = (rng.standard_normal(hop) * 0.02).astype(np.float32)

    for _ in range(args.warmup):
        engine.step(ref, err)

    times = np.empty(args.steps)
    for i in range(args.steps):
        t0 = time.perf_counter()
        engine.step(ref, err)
        times[i] = (time.perf_counter() - t0) * 1000.0

    p50, p90, p99 = np.percentile(times, [50, 90, 99])
    kind = cfg.get("controller", "dl")
    if kind == "dl":
        kind = f"dl/{cfg.get('engine', {}).get('type')}"
    print(f"엔진 {kind} | hop {hop} (예산 {budget_ms:.2f}ms) | {args.steps}회")
    print(f"  P50 {p50:6.2f}ms | P90 {p90:6.2f}ms | P99 {p99:6.2f}ms | max {times.max():6.2f}ms")
    ok = p99 < args.gate_p99_ms
    print(f"  게이트 P99 < {args.gate_p99_ms}ms : {'통과 ✓' if ok else '미달 ✗ → tiny 모델/TRT 전환 검토'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
