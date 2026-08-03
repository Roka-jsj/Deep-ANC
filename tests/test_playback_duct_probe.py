"""출력 전용 덕트 정성 진단의 무음 경계·안전 확인."""

import numpy as np

from scripts.bench.playback_duct_probe import main, stepped_tone


def test_stepped_tone_has_faded_boundaries_and_requested_peak():
    signal = stepped_tone(300.0, 48_000, 0.2, 0.002)
    assert signal.dtype == np.float32
    assert signal[0] == 0.0
    assert abs(float(signal[-1])) < 1.0e-5
    assert np.max(np.abs(signal)) <= 0.002001
    assert np.sqrt(np.mean(signal.astype(np.float64) ** 2)) > 0.0005


def test_playback_probe_requires_explicit_volume_confirmation():
    assert main([]) == 2
