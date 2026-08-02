"""실측 녹음 파인튜닝 데이터셋.

scripts/data/record_duct.py 가 저장한 세션 구조:
    data/recorded/<session_id>/
      mics.wav    : 2ch PCM_32 (ch0=err mic, ch1=ref mic)
      source.wav  : 재생한 디지털 소스 (1ch)
      session.json: 프로그램/레벨/설정 메타

ANC OFF 상태로 녹음했으므로 err 마이크 신호가 곧 d(t)이다.
digital-ref 모드에서는 source.wav 를 x_ref 로, acoustic-ref 모드에서는
ref 마이크(ch1)를 x_ref 로 사용한다. manifest(recorded_*.jsonl)는 세션 단위 분할.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .manifest import read_manifest
from .synth_dataset import _delay_np


class RecordedANCDataset(IterableDataset):
    def __init__(
        self,
        manifest_path: str | Path,
        data_cfg: dict,
        split: str = "train",
        seed: int = 20260803,
    ) -> None:
        super().__init__()
        self.entries = read_manifest(manifest_path, split=split)
        if not self.entries:
            raise ValueError(f"'{split}' split 세션이 없습니다: {manifest_path}")
        self.fs = int(data_cfg["sample_rate"])
        raw_segment = int(round(float(data_cfg["segment_seconds"]) * self.fs))
        self.segment = max(256, (raw_segment // 256) * 256)
        self.reference_mode = str(data_cfg.get("reference_mode", "digital"))
        fb = data_cfg.get("closed_loop", {}).get("feedback_delay_samples", [512, 1024])
        self.feedback_delay_range = (int(fb[0]), int(fb[1]))
        self.seed = int(seed)

    def _load_session(self, entry: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        session_dir = Path(entry["path"])
        mics, sr = sf.read(session_dir / "mics.wav", dtype="float32", always_2d=True)
        if sr != self.fs:
            raise ValueError(f"{session_dir}: 샘플레이트 {sr} != {self.fs}")
        err = mics[:, 0]
        ref = mics[:, 1]
        source_path = session_dir / "source.wav"
        if source_path.exists():
            source, _ = sf.read(source_path, dtype="float32", always_2d=True)
            source = source[:, 0]
        else:
            source = np.zeros_like(err)
        n = min(err.size, ref.size, source.size)
        return err[:n], ref[:n], source[:n]

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rng = np.random.default_rng(self.seed + worker_id * 1013)
        sessions = [self._load_session(e) for e in self.entries]
        meta = [json.loads(Path(e["path"], "session.json").read_text(encoding="utf-8"))
                if Path(e["path"], "session.json").exists() else {} for e in self.entries]
        del meta  # 세션 메타는 현재 학습에 미사용 (프로토콜 기록용)

        while True:
            err, ref, source = sessions[int(rng.integers(len(sessions)))]
            if err.size <= self.segment:
                continue
            start = int(rng.integers(0, err.size - self.segment))
            sl = slice(start, start + self.segment)
            d = err[sl].copy()
            x_ref = source[sl].copy() if self.reference_mode == "digital" else ref[sl].copy()
            fb_delay = int(rng.integers(*self.feedback_delay_range))
            err_in = _delay_np(d, fb_delay)
            x = np.stack([x_ref, err_in]).astype(np.float32)
            yield {
                "x": torch.from_numpy(x),
                "d": torch.from_numpy(d.astype(np.float32)).unsqueeze(0),
            }
