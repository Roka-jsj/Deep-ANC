"""JSONL manifest — 데이터 인덱스 (경로/분할/길이/태그).

분할 규칙 (누수 방지): 노이즈 풀은 원본 파일 단위, 실측 데이터는 세션 단위로
train/val/test 를 나눈다. tests/test_dataset.py 가 교차 누수를 검사한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf

VALID_SPLITS = ("train", "val", "test")


def write_manifest(entries: list[dict], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for entry in entries:
            if entry.get("split") not in VALID_SPLITS:
                raise ValueError(f"split 은 {VALID_SPLITS} 중 하나여야 합니다: {entry}")
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_manifest(path: str | Path, split: str | None = None) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"manifest 없음: {p}")
    entries: list[dict] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if split is None or entry.get("split") == split:
                entries.append(entry)
    return entries


def scan_wavs(root: str | Path, tag: str) -> list[dict]:
    """디렉토리의 wav/flac 을 스캔해 manifest 엔트리 생성 (split 은 호출자가 배정)."""
    root = Path(root)
    entries = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in {".wav", ".flac"}:
            continue
        try:
            info = sf.info(str(p))
        except RuntimeError:
            continue
        entries.append(
            {
                "path": str(p),
                "duration_s": float(info.frames) / float(info.samplerate),
                "sample_rate": int(info.samplerate),
                "channels": int(info.channels),
                "tag": tag,
            }
        )
    return entries


def assign_splits(
    entries: list[dict], ratios: dict[str, float], seed: int = 20260802
) -> list[dict]:
    """파일 단위 랜덤 분할 (재현 가능)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(entries))
    n = len(entries)
    n_train = int(round(n * float(ratios.get("train", 0.9))))
    n_val = int(round(n * float(ratios.get("val", 0.05))))
    out = [dict(e) for e in entries]
    for rank, idx in enumerate(order):
        if rank < n_train:
            out[idx]["split"] = "train"
        elif rank < n_train + n_val:
            out[idx]["split"] = "val"
        else:
            out[idx]["split"] = "test"
    return out
