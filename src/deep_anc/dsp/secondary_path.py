"""측정된 2차경로 S(z) — NPZ 로더 + 미분가능(torch) 플랜트.

anc_project 의 캘리브레이션 산출물(NPZ: compact FIR + 순수지연)을 로드해
학습 손실 그래프에 넣는다. 극성 규약: 측정 FIR에 음향/전기 극성이 이미 포함되어
있으므로 추가 부호 반전을 하지 않는다 (e = d + S*y).  — fxlms_core.py 규약 계승

지연 구성 [설계 교차검증 C1]:
    총지연 = delay_samples(캘리브레이션 실측, I/O 버퍼 포함)
           + handoff_extra_samples(3-스레드 런타임의 1 hop 핸드오프)
           + jitter(학습 증강, 비대칭 [0, +512])
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class SecondaryPathData:
    fir: np.ndarray            # compact FIR (float32)
    delay_samples: int         # 순수지연 (캘리브레이션 실측)
    sample_rate: int
    fit_improvement_db: float
    coherence_median: float
    excitation_band_hz: tuple[float, float]
    source_path: str


def _npz_scalar(data: Any, key: str, default: Any) -> Any:
    if key not in data:
        return default
    value = data[key]
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(-1)[0].item()
    return value


def load_secondary_path(path: str | Path) -> SecondaryPathData:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"2차경로 모델이 없습니다: {p}")
    with np.load(p, allow_pickle=False) as data:
        if "fir" not in data:
            raise ValueError(f"{p} 에 'fir' 배열이 없습니다")
        fir = np.asarray(data["fir"], dtype=np.float32).reshape(-1)
        delay = int(_npz_scalar(data, "delay_samples", 0))
        sr = int(_npz_scalar(data, "sample_rate", 48000))
        fit = float(_npz_scalar(data, "fit_improvement_db", float("nan")))
        coh = float(_npz_scalar(data, "coherence_median", float("nan")))
        band = data["excitation_band_hz"] if "excitation_band_hz" in data else np.array([0.0, 0.0])
        band = tuple(float(v) for v in np.asarray(band).reshape(-1)[:2])

    if fir.size < 1 or not np.all(np.isfinite(fir)) or float(np.max(np.abs(fir))) <= 0.0:
        raise ValueError(f"잘못된 2차경로 FIR: {p}")
    if delay < 0:
        raise ValueError("2차경로 지연은 음수일 수 없습니다")
    return SecondaryPathData(
        fir=np.ascontiguousarray(fir),
        delay_samples=delay,
        sample_rate=sr,
        fit_improvement_db=fit,
        coherence_median=coh,
        excitation_band_hz=band,
        source_path=str(p),
    )


def fft_causal_filter(x: torch.Tensor, fir: torch.Tensor) -> torch.Tensor:
    """인과 선형 컨볼루션 (FFT) — x: [..., T], fir: [L]. 앞 T 샘플 반환."""
    T = x.shape[-1]
    L = fir.shape[-1]
    n = 1
    while n < T + L - 1:
        n *= 2
    X = torch.fft.rfft(x, n)
    H = torch.fft.rfft(fir, n)
    out = torch.fft.irfft(X * H, n)
    return out[..., :T]


def integer_delay(x: torch.Tensor, delay: int) -> torch.Tensor:
    """우측 시프트(과거로 지연), 길이 유지, 앞은 0."""
    if delay <= 0:
        return x
    return F.pad(x, (delay, 0))[..., : x.shape[-1]]


class DifferentiableSecondaryPath(nn.Module):
    """학습 그래프용 S(z): y[B,1,T] → S*y (지연 + FIR, 선택적 섭동 증강).

    섭동 증강은 coherence 0.40 인 측정 신뢰도에 대한 도메인 랜덤화이다:
    지연 지터(비대칭), 게인/틸트, 랜덤 올패스 위상.
    """

    def __init__(
        self,
        data: SecondaryPathData,
        handoff_extra_samples: int = 0,
        delay_jitter_range: tuple[int, int] = (0, 0),
        gain_db_range: tuple[float, float] = (0.0, 0.0),
        tilt_db_per_octave_range: tuple[float, float] = (0.0, 0.0),
        allpass_perturb: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("fir", torch.from_numpy(data.fir.copy()))
        self.base_delay = int(data.delay_samples) + int(handoff_extra_samples)
        self.sample_rate = data.sample_rate
        self.delay_jitter_range = (int(delay_jitter_range[0]), int(delay_jitter_range[1]))
        self.gain_db_range = tuple(map(float, gain_db_range))
        self.tilt_range = tuple(map(float, tilt_db_per_octave_range))
        self.allpass_perturb = bool(allpass_perturb)
        self.rng = np.random.default_rng(seed)

    def sample_perturbation(self) -> dict:
        lo, hi = self.delay_jitter_range
        jitter = int(self.rng.integers(lo, hi + 1)) if hi > lo else lo
        gain_db = float(self.rng.uniform(*self.gain_db_range)) if self.gain_db_range[1] > self.gain_db_range[0] else 0.0
        tilt = float(self.rng.uniform(*self.tilt_range)) if self.tilt_range[1] > self.tilt_range[0] else 0.0
        return {"jitter": jitter, "gain_db": gain_db, "tilt_db_per_octave": tilt, "allpass": self.allpass_perturb}

    def _perturbed_fir(self, perturb: dict, device: torch.device) -> torch.Tensor:
        fir = self.fir.to(device)
        gain_db = perturb.get("gain_db", 0.0)
        tilt = perturb.get("tilt_db_per_octave", 0.0)
        use_allpass = perturb.get("allpass", False)
        if gain_db == 0.0 and tilt == 0.0 and not use_allpass:
            return fir

        L = fir.shape[-1]
        n = 1
        while n < 2 * L:
            n *= 2
        H = torch.fft.rfft(fir, n)
        freqs = torch.fft.rfftfreq(n, d=1.0 / self.sample_rate).to(device)

        mag = torch.ones_like(freqs) * (10.0 ** (gain_db / 20.0))
        if tilt != 0.0:
            ref = 500.0  # 틸트 기준 주파수
            octaves = torch.log2(torch.clamp(freqs, min=20.0) / ref)
            mag = mag * (10.0 ** (tilt * octaves / 20.0))
        H = H * mag

        if use_allpass:
            # 완만한 랜덤 위상 (저차 랜덤 워크를 평활화) — 에너지 보존 위상 섭동
            steps = int(self.rng.integers(3, 7))
            knots = self.rng.uniform(-0.6, 0.6, size=steps)
            phase_np = np.interp(
                np.linspace(0.0, 1.0, freqs.shape[0]),
                np.linspace(0.0, 1.0, steps),
                knots,
            )
            phase = torch.from_numpy(phase_np.astype(np.float32)).to(device)
            H = H * torch.exp(1j * phase)

        out = torch.fft.irfft(H, n)[:L]
        return out

    def forward(self, y: torch.Tensor, perturb: dict | None = None) -> torch.Tensor:
        """y: [B, 1, T] (물리 스케일) → e 에 더해질 S(G(y)) 성분 [B, 1, T]."""
        if perturb is None:
            perturb = {"jitter": 0}
        fir = self._perturbed_fir(perturb, y.device)
        delayed = integer_delay(y, self.base_delay + int(perturb.get("jitter", 0)))
        return fft_causal_filter(delayed, fir)
