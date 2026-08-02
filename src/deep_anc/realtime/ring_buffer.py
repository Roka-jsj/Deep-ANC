"""SPSC(단일 생산자-단일 소비자) 링버퍼 — 콜백↔추론 스레드 핸드오프.

콜백 쪽은 절대 블로킹하지 않는다: 데이터가 없으면 즉시 무음을 반환하고
언더런 카운터만 올린다 (지연 워치독이 감시).
인덱스는 파이썬 int(GIL 원자적)만 사용 — 락은 소비자 웨이크업 조건변수뿐.
"""

from __future__ import annotations

import threading

import numpy as np


class SPSCRing:
    def __init__(self, channels: int, capacity_samples: int) -> None:
        self.channels = int(channels)
        self.capacity = int(capacity_samples)
        self.buf = np.zeros((self.channels, self.capacity), dtype=np.float32)
        self.write_pos = 0            # 총 누적 샘플 (mod 는 접근 시)
        self.read_pos = 0
        self.underruns = 0
        self.overruns = 0
        self.cond = threading.Condition()

    def available(self) -> int:
        return self.write_pos - self.read_pos

    def push(self, block: np.ndarray) -> None:
        """생산자 전용. block: [channels, n]. 가득 차면 가장 오래된 데이터를 버린다."""
        n = block.shape[-1]
        if self.available() + n > self.capacity:
            drop = self.available() + n - self.capacity
            self.read_pos += drop
            self.overruns += 1
        start = self.write_pos % self.capacity
        end = start + n
        if end <= self.capacity:
            self.buf[:, start:end] = block
        else:
            k = self.capacity - start
            self.buf[:, start:] = block[:, :k]
            self.buf[:, : end - self.capacity] = block[:, k:]
        self.write_pos += n
        with self.cond:
            self.cond.notify()

    def pop(self, n: int) -> np.ndarray | None:
        """소비자 전용. n 샘플이 없으면 None (논블로킹)."""
        if self.available() < n:
            return None
        start = self.read_pos % self.capacity
        end = start + n
        if end <= self.capacity:
            out = self.buf[:, start:end].copy()
        else:
            k = self.capacity - start
            out = np.concatenate([self.buf[:, start:], self.buf[:, : end - self.capacity]], axis=1)
        self.read_pos += n
        return out

    def pop_or_silence(self, n: int) -> tuple[np.ndarray, bool]:
        """콜백용: 부족하면 무음 + underrun 표시."""
        out = self.pop(n)
        if out is None:
            self.underruns += 1
            return np.zeros((self.channels, n), dtype=np.float32), False
        return out, True

    def wait_for(self, n: int, timeout: float) -> bool:
        """소비자용: n 샘플이 쌓일 때까지 대기."""
        with self.cond:
            return self.cond.wait_for(lambda: self.available() >= n, timeout=timeout)

    def reset(self) -> None:
        self.read_pos = self.write_pos
