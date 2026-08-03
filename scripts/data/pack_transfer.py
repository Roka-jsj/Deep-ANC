#!/usr/bin/env python3
"""데이터 전송 패킹 — Elice 업로드용 tar 샤드 생성 (수만 개 파일 → 수 개 tar).

  .venv/bin/python scripts/data/pack_transfer.py --src data/raw/noise --shard-gb 2
Elice 쪽에서: for f in transfer/*.tar; do tar -xf "$f" ; done
"""

import argparse
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.config import REPO_ROOT                          # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="data/raw/noise")
    parser.add_argument("--out", default="transfer")
    parser.add_argument("--shard-gb", type=float, default=2.0)
    args = parser.parse_args()

    src = REPO_ROOT / args.src
    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    limit = int(args.shard_gb * 1e9)

    files = [p for p in sorted(src.rglob("*")) if p.is_file()]
    if not files:
        print(f"파일 없음: {src}")
        return 1

    shard_idx, size, tf = 0, 0, None
    try:
        for p in files:
            if tf is None or size >= limit:
                if tf is not None:
                    tf.close()
                shard_idx += 1
                size = 0
                shard_path = out_dir / f"noise_shard_{shard_idx:03d}.tar"
                tf = tarfile.open(shard_path, "w")
                print(f"[샤드] {shard_path}")
            tf.add(p, arcname=str(p.relative_to(REPO_ROOT)))
            size += p.stat().st_size
    finally:
        if tf is not None:
            tf.close()
    print(f"완료: {shard_idx}개 샤드 → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
