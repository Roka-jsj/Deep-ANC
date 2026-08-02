"""노이즈 wav 풀 — manifest 기반 랜덤 세그먼트 샘플러."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from .manifest import read_manifest


class NoisePool:
    def __init__(
        self,
        manifest_paths: list[str | Path],
        split: str,
        sample_rate: int,
        seed: int | None = None,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.rng = np.random.default_rng(seed)
        self.entries: list[dict] = []
        for mp in manifest_paths:
            self.entries.extend(read_manifest(mp, split=split))
        if not self.entries:
            raise ValueError(f"'{split}' split 에 해당하는 노이즈 파일이 없습니다: {manifest_paths}")
        durations = np.array([max(0.1, float(e["duration_s"])) for e in self.entries])
        self.weights = durations / durations.sum()

    def __len__(self) -> int:
        return len(self.entries)

    def sample_segment(self, n_samples: int) -> np.ndarray:
        """길이 가중 랜덤 파일에서 무작위 구간을 읽어 48kHz 모노로 반환."""
        entry = self.entries[int(self.rng.choice(len(self.entries), p=self.weights))]
        path = entry["path"]
        file_sr = int(entry.get("sample_rate", self.sample_rate))
        need_src = int(np.ceil(n_samples * file_sr / self.sample_rate)) + 16

        info = sf.info(path)
        total = int(info.frames)
        if total <= need_src:
            data, _ = sf.read(path, dtype="float32", always_2d=True)
        else:
            start = int(self.rng.integers(0, total - need_src))
            data, _ = sf.read(
                path, start=start, stop=start + need_src, dtype="float32", always_2d=True
            )
        mono = data.mean(axis=1)

        if file_sr != self.sample_rate:
            from math import gcd

            g = gcd(file_sr, self.sample_rate)
            mono = signal.resample_poly(mono, self.sample_rate // g, file_sr // g)

        if mono.size < n_samples:
            reps = int(np.ceil(n_samples / max(1, mono.size)))
            mono = np.tile(mono, reps)
        return mono[:n_samples].astype(np.float32)
