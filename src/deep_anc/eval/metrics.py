"""ANC 평가 지표 — NMSE(dB)와 옥타브밴드별 감쇠량.

부호 규약: 감쇠(attenuation) = 양수일수록 좋음 = 10·log10(P_d / P_e).
NMSE = −감쇠 (음수일수록 좋음). 두 값 모두 리포트에 표기한다.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import signal

_EPS = 1.0e-12


def nmse_db(d: np.ndarray, e: np.ndarray) -> float:
    """10·log10(mean(e²) / mean(d²)) — 음수일수록 좋음.

    d/e 길이가 같으면 기존의 ``Σe² / Σd²`` 정의와 동일하다.
    실기 평가의 OFF/ON 녹음 길이가 다른 경우에도 시간 길이 차이가
    감쇠량으로 잘못 계산되지 않도록 평균 전력을 사용한다.
    """
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    e = np.asarray(e, dtype=np.float64).reshape(-1)
    if d.size == 0 or e.size == 0:
        raise ValueError("NMSE 계산에는 비어 있지 않은 d/e 신호가 필요합니다")
    return float(10.0 * np.log10((np.mean(e**2) + _EPS) / (np.mean(d**2) + _EPS)))


def intersect_frequency_bands(
    first: tuple[float, float] | list[float],
    second: tuple[float, float] | list[float],
    nyquist_hz: float,
) -> tuple[float, float]:
    """두 유효 주파수 대역의 교집을 반환하고 빈 교집은 거부한다."""

    def _validate(name: str, band: tuple[float, float] | list[float]) -> tuple[float, float]:
        try:
            if len(band) != 2:
                raise ValueError
            lo, hi = (float(v) for v in band)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} 대역은 [lo, hi] 형식이어야 합니다: {band}") from exc
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError(f"{name} 대역은 유한한 값이어야 합니다: {band}")
        if not (0.0 <= lo < hi <= nyquist):
            raise ValueError(f"잘못된 {name} 대역: {band} (Nyquist={nyquist:g}Hz)")
        return lo, hi

    nyquist = float(nyquist_hz)
    if not math.isfinite(nyquist) or nyquist <= 0.0:
        raise ValueError(f"Nyquist 주파수는 유한한 양수여야 합니다: {nyquist_hz}")
    first_lo, first_hi = _validate("첫 번째", first)
    second_lo, second_hi = _validate("두 번째", second)
    lo, hi = max(first_lo, second_lo), min(first_hi, second_hi)
    if lo >= hi:
        raise ValueError(f"주파수 대역 교집이 비어 있습니다: {first} ∩ {second}")
    return lo, hi


def band_nmse_db(
    d: np.ndarray,
    e: np.ndarray,
    sample_rate: int,
    band_hz: tuple[float, float] | list[float],
) -> float:
    """주어진 대역의 one-sided Parseval 전력 NMSE(dB).

    신호별 FFT 에너지를 각 신호 길이로 나누므로 OFF/ON 구간의
    길이가 달라도 평균 대역 전력을 비교한다.
    """
    fs = float(sample_rate)
    if not math.isfinite(fs) or fs <= 0.0:
        raise ValueError(f"sample_rate는 유한한 양수여야 합니다: {sample_rate}")
    band = intersect_frequency_bands(band_hz, band_hz, fs / 2.0)
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    e = np.asarray(e, dtype=np.float64).reshape(-1)
    if d.size < 2 or e.size < 2:
        raise ValueError("대역 NMSE 계산에는 d/e 각각 2샘플 이상이 필요합니다")

    def _band_power(x: np.ndarray, name: str) -> float:
        spectrum = np.fft.rfft(x, norm="ortho")
        frequencies = np.fft.rfftfreq(x.size, d=1.0 / fs)
        mask = (frequencies >= band[0]) & (frequencies <= band[1])
        if not np.any(mask):
            raise ValueError(
                f"{name} {x.size}샘플 FFT에 {band[0]:g}–{band[1]:g}Hz 빈이 없습니다"
            )
        weights = np.full(spectrum.shape, 2.0, dtype=np.float64)
        weights[0] = 1.0
        if x.size % 2 == 0:
            weights[-1] = 1.0
        energy = np.sum(np.abs(spectrum[mask]) ** 2 * weights[mask])
        return float(energy / x.size)

    d_power = _band_power(d, "d")
    e_power = _band_power(e, "e")
    return float(10.0 * np.log10((e_power + _EPS) / (d_power + _EPS)))


def attenuation_db(d: np.ndarray, e: np.ndarray) -> float:
    """전대역 감쇠량 (양수 = 저감)."""
    return -nmse_db(d, e)


def octave_band_attenuation(
    d: np.ndarray,
    e: np.ndarray,
    sample_rate: int,
    centers_hz: list[float],
    trusted_band_hz: tuple[float, float] | None = None,
) -> list[dict]:
    """옥타브밴드(중심 f, 경계 f/√2~f·√2)별 감쇠량.

    trusted_band_hz 를 주면 S(z) 보정 유효대역 밖 밴드에 trusted=False 를 표기
    [설계 교차검증 L2 — 유효대역 밖 수치는 신뢰 낮음].
    """
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    e = np.asarray(e, dtype=np.float64).reshape(-1)
    if trusted_band_hz is not None:
        trusted_band_hz = intersect_frequency_bands(
            trusted_band_hz, trusted_band_hz, sample_rate / 2.0
        )
    out: list[dict] = []
    sqrt2 = np.sqrt(2.0)
    for fc in centers_hz:
        lo, hi = fc / sqrt2, fc * sqrt2
        if hi >= sample_rate / 2 * 0.98:
            continue
        sos = signal.butter(4, [lo, hi], btype="bandpass", fs=sample_rate, output="sos")
        d_band = signal.sosfilt(sos, d)
        e_band = signal.sosfilt(sos, e)
        att = attenuation_db(d_band, e_band)
        trusted = True
        if trusted_band_hz is not None:
            trusted = trusted_band_hz[0] <= fc <= trusted_band_hz[1]
        out.append({"center_hz": float(fc), "attenuation_db": att, "trusted": trusted})
    return out


def segment_stats(d: np.ndarray, e: np.ndarray, sample_rate: int, seg_seconds: float = 1.0) -> dict:
    """세그먼트별 감쇠 분포 (중앙값 / 최악 10%)."""
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    e = np.asarray(e, dtype=np.float64).reshape(-1)
    seg = int(seg_seconds * sample_rate)
    vals = []
    for start in range(0, d.size - seg + 1, seg):
        sl = slice(start, start + seg)
        vals.append(attenuation_db(d[sl], e[sl]))
    if not vals:
        vals = [attenuation_db(d, e)]
    arr = np.array(vals)
    return {
        "median_db": float(np.median(arr)),
        "worst10_db": float(np.percentile(arr, 10)),
        "n_segments": int(arr.size),
    }
