"""평가 지표의 해석적 검증."""

import numpy as np

from deep_anc.eval.metrics import (
    attenuation_db,
    nmse_db,
    octave_band_attenuation,
    segment_stats,
)

FS = 48000


def test_nmse_half_amplitude():
    rng = np.random.default_rng(0)
    d = rng.standard_normal(FS)
    e = 0.5 * d                                   # 진폭 절반 → -6.02 dB
    assert abs(nmse_db(d, e) + 6.0206) < 0.01
    assert abs(attenuation_db(d, e) - 6.0206) < 0.01


def test_octave_band_selective():
    """300Hz 성분만 제거된 경우 250Hz 밴드(177~354Hz)에서만 큰 감쇠."""
    t = np.arange(FS * 2) / FS
    tone300 = np.sin(2 * np.pi * 300 * t)
    tone1200 = np.sin(2 * np.pi * 1200 * t)
    d = tone300 + tone1200
    e = 0.01 * tone300 + tone1200                 # 300Hz 만 40dB 감쇠

    bands = octave_band_attenuation(d, e, FS, [125, 250, 500, 1000, 2000], (150, 600))
    by_center = {b["center_hz"]: b for b in bands}
    assert by_center[250]["attenuation_db"] > 30
    assert abs(by_center[1000]["attenuation_db"]) < 3
    assert by_center[250]["trusted"] is True
    assert by_center[2000]["trusted"] is False    # S(z) 유효대역(150~600) 밖


def test_segment_stats():
    rng = np.random.default_rng(1)
    d = rng.standard_normal(FS * 3)
    e = 0.1 * d
    stats = segment_stats(d, e, FS, seg_seconds=1.0)
    assert stats["n_segments"] == 3
    assert abs(stats["median_db"] - 20.0) < 0.5
