"""ANC 평가 지표 — NMSE(dB)와 옥타브밴드별 감쇠량.

부호 규약: 감쇠(attenuation) = 양수일수록 좋음 = 10·log10(P_d / P_e).
NMSE = −감쇠 (음수일수록 좋음). 두 값 모두 리포트에 표기한다.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

_EPS = 1.0e-12


def nmse_db(d: np.ndarray, e: np.ndarray) -> float:
    """10·log10(Σe² / Σd²) — 음수일수록 좋음."""
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    e = np.asarray(e, dtype=np.float64).reshape(-1)
    return float(10.0 * np.log10((np.sum(e**2) + _EPS) / (np.sum(d**2) + _EPS)))


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
