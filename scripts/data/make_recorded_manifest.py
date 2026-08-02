#!/usr/bin/env python3
"""실측 세션 → manifest (세션 단위 train/val/test 분할, 누수 방지).

  python scripts/data/make_recorded_manifest.py
"""

import argparse
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.config import REPO_ROOT                          # noqa: E402
from deep_anc.data.manifest import assign_splits, write_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/recorded")
    parser.add_argument("--out", default="data/manifests/recorded_train.jsonl")
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    root = REPO_ROOT / args.root
    entries = []
    for session in sorted(root.iterdir()) if root.exists() else []:
        mics = session / "mics.wav"
        if not mics.exists():
            continue
        info = sf.info(str(mics))
        entries.append(
            {
                "path": str(session),
                "duration_s": float(info.frames) / float(info.samplerate),
                "sample_rate": int(info.samplerate),
                "tag": "recorded",
            }
        )
    if not entries:
        print(f"세션 없음: {root} — 먼저 record_duct.py 로 수집하세요")
        return 1

    entries = assign_splits(entries, {"train": 0.8, "val": 0.1}, seed=args.seed)
    out = REPO_ROOT / args.out
    write_manifest(entries, out)
    total_min = sum(e["duration_s"] for e in entries) / 60.0
    print(f"{len(entries)}개 세션 ({total_min:.1f}분) → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
