#!/usr/bin/env python3
"""노이즈 풀 인덱싱 → JSONL manifest 생성 (파일 단위 train/val/test 분할).

  python scripts/data/prepare_noise_pool.py
리샘플/정규화는 학습 로더(NoisePool)가 실시간으로 수행하므로 여기서는 인덱스만 만든다.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.config import REPO_ROOT                         # noqa: E402
from deep_anc.data.manifest import (                          # noqa: E402
    assign_splits,
    scan_wavs,
    write_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/noise")
    parser.add_argument("--out", default="data/manifests")
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()

    root = REPO_ROOT / args.root
    out_dir = REPO_ROOT / args.out

    for tag, subdir in (("dns_fullband", "dns_fullband"), ("esc50", "esc50")):
        src = root / subdir
        if not src.exists():
            print(f"[skip] {src} 없음")
            continue
        entries = scan_wavs(src, tag)
        if not entries:
            print(f"[skip] {src} 에 오디오 없음")
            continue
        entries = assign_splits(entries, {"train": 0.9, "val": 0.05}, seed=args.seed)
        out = out_dir / f"{tag}.jsonl"
        write_manifest(entries, out)
        n_train = sum(1 for e in entries if e["split"] == "train")
        total_h = sum(e["duration_s"] for e in entries) / 3600.0
        print(f"{tag}: {len(entries)}개 파일 ({total_h:.1f}h), train {n_train} → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
