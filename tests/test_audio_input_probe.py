"""출력 없는 I2S 입력 probe의 고정값·클리핑 판정."""

import numpy as np

from deep_anc.audio_io import analyze_int32_input_probe
from scripts.demo import evaluate_session


def _probe_report(err_valid: bool, ref_valid: bool) -> dict:
    channels = []
    for index, valid in enumerate((err_valid, ref_valid)):
        channels.append(
            {
                "channel": index,
                "rms_dbfs": -30.0 if valid else -186.64,
                "peak": 0.1 if valid else 0.0,
                "clip_ratio": 0.0,
                "unique_codes": 100 if valid else 1,
                "raw_min": -100 if valid else -1,
                "raw_max": 100 if valid else -1,
                "valid": valid,
            }
        )
    return {"frames": 1024, "channels": channels}


def test_input_probe_rejects_stuck_minus_one_channels():
    raw = np.full((1024, 2), -1, dtype=np.int32)
    report = analyze_int32_input_probe(raw)
    assert all(item["stuck"] and not item["valid"] for item in report["channels"])


def test_input_probe_accepts_low_level_dynamic_pcm():
    phase = np.linspace(0.0, 8.0 * np.pi, 4096, endpoint=False)
    signal = np.rint(np.sin(phase) * (2**22)).astype(np.int32)
    raw = np.stack([signal, np.roll(signal, 31)], axis=1)
    report = analyze_int32_input_probe(raw)
    assert all(item["valid"] for item in report["channels"])


def test_input_probe_rejects_excessive_fullscale_clipping():
    rng = np.random.default_rng(7)
    raw = rng.integers(-(2**22), 2**22, size=(4096, 2), dtype=np.int32)
    raw[:100, 0] = np.iinfo(np.int32).max
    report = analyze_int32_input_probe(raw, max_clip_ratio=0.005)
    assert not report["channels"][0]["valid"]
    assert report["channels"][1]["valid"]


def test_session_preflight_rejects_invalid_error_before_output(monkeypatch):
    monkeypatch.setattr(
        evaluate_session,
        "capture_input_probe",
        lambda *_args, **_kwargs: _probe_report(False, True),
    )
    cfg = {"reference": "digital", "hardware": {"audio": {}}}
    assert evaluate_session.input_preflight(cfg, seconds=0.1) is False


def test_session_preflight_reference_requirement_depends_on_mode(monkeypatch):
    monkeypatch.setattr(
        evaluate_session,
        "capture_input_probe",
        lambda *_args, **_kwargs: _probe_report(True, False),
    )
    digital_cfg = {"reference": "digital", "hardware": {"audio": {}}}
    mic_cfg = {"reference": "mic", "hardware": {"audio": {}}}
    assert evaluate_session.input_preflight(digital_cfg, seconds=0.1) is True
    assert evaluate_session.input_preflight(mic_cfg, seconds=0.1) is False
