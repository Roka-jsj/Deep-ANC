#!/usr/bin/env python3
"""스피커를 열지 않고 Jetson ERR/REF 마이크 2채널을 사전 점검한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.audio_io import capture_input_probe  # noqa: E402
from deep_anc.config import REPO_ROOT, load_yaml  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", default="configs/hardware_jetson.yaml")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--min-rms-dbfs", type=float, default=-80.0)
    parser.add_argument("--max-clip-ratio", type=float, default=0.005)
    parser.add_argument(
        "--require-both",
        action="store_true",
        help="기본은 FxLMS/digital-ref에 필수인 ERR ch0만 게이트; 지정 시 REF ch1도 필수",
    )
    args = parser.parse_args(argv)

    hardware = load_yaml(REPO_ROOT / args.hardware)["audio"]
    try:
        report = capture_input_probe(
            hardware,
            seconds=args.seconds,
            min_rms_dbfs=args.min_rms_dbfs,
            max_clip_ratio=args.max_clip_ratio,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[FAIL] 입력 probe 실패: {exc}", file=sys.stderr)
        return 2

    names = ("ERR", "REF")
    for item in report["channels"][:2]:
        index = int(item["channel"])
        verdict = "PASS" if item["valid"] else "FAIL"
        print(
            f"[{verdict}] {names[index]} ch{index}: RMS {item['rms_dbfs']:.2f}dBFS, "
            f"peak {item['peak']:.6f}, clip {item['clip_ratio']:.3%}, "
            f"unique {item['unique_codes']}, raw [{item['raw_min']}, {item['raw_max']}]"
        )

    required = report["channels"][:2] if args.require_both else report["channels"][:1]
    ok = all(bool(item["valid"]) for item in required)
    if not bool(report["channels"][1]["valid"]):
        print("[경고] REF ch1 무효 — acoustic-reference 수집/평가는 금지", file=sys.stderr)
    if not ok:
        print("[중단] 필수 마이크 입력이 무효이므로 스피커 실측을 시작하지 마세요.", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
