"""스트리밍 DSP 기본 블록 (numpy).

DCBlocker / SampleDelay 는 anc_project/fxlms_core.py 의 실기 검증 구현을 이식.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

_FLOAT32_ONE = np.array([1.0], dtype=np.float32)


class DCBlocker:
    """1차 DC 차단기: (1 - z^-1) / (1 - r z^-1), 스트리밍 상태 유지."""

    def __init__(self, r: float = 0.995) -> None:
        if not 0.0 < r < 1.0:
            raise ValueError("r 은 (0, 1) 범위여야 합니다")
        self.r = float(r)
        self._b = np.array([1.0, -1.0], dtype=np.float32)
        self._a = np.array([1.0, -self.r], dtype=np.float32)
        self._zi = np.zeros(1, dtype=np.float32)

    def reset(self) -> None:
        self._zi.fill(0.0)

    def process(self, block: np.ndarray) -> np.ndarray:
        values = np.asarray(block, dtype=np.float32).reshape(-1)
        output, self._zi = signal.lfilter(self._b, self._a, values, zi=self._zi)
        return np.asarray(output, dtype=np.float32)


class SampleDelay:
    """인과적 정수 샘플 지연 (스트리밍)."""

    def __init__(self, delay_samples: int) -> None:
        if delay_samples < 0:
            raise ValueError("delay_samples 는 음수일 수 없습니다")
        self.delay_samples = int(delay_samples)
        self._state = np.zeros(self.delay_samples, dtype=np.float32)

    def reset(self) -> None:
        self._state.fill(0.0)

    def process(self, block: np.ndarray) -> np.ndarray:
        values = np.asarray(block, dtype=np.float32).reshape(-1)
        if self.delay_samples == 0:
            return values.copy()
        joined = np.concatenate((self._state, values))
        output = joined[: values.size].copy()
        self._state = joined[values.size : values.size + self.delay_samples].copy()
        return output


class StreamingFIR:
    """스트리밍 FIR 필터 (lfilter 상태 유지). 오프라인 검증·베이스라인용."""

    def __init__(self, fir: np.ndarray, delay_samples: int = 0) -> None:
        self.fir = np.asarray(fir, dtype=np.float32).reshape(-1)
        if self.fir.size < 1:
            raise ValueError("빈 FIR")
        self.delay = SampleDelay(delay_samples)
        self._zi = np.zeros(max(0, self.fir.size - 1), dtype=np.float32)

    def reset(self) -> None:
        self.delay.reset()
        self._zi.fill(0.0)

    def process(self, block: np.ndarray) -> np.ndarray:
        delayed = self.delay.process(block)
        output, self._zi = signal.lfilter(self.fir, _FLOAT32_ONE, delayed, zi=self._zi)
        return np.asarray(output, dtype=np.float32)


def fft_filter(x: np.ndarray, fir: np.ndarray) -> np.ndarray:
    """오프라인 선형 컨볼루션 (앞부분 x 길이만 반환 = 인과 필터링)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    fir = np.asarray(fir, dtype=np.float64).reshape(-1)
    out = signal.fftconvolve(x, fir, mode="full")[: x.size]
    return out.astype(np.float32)
