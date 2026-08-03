"""실시간 런타임의 안전한 초기 상태 규약."""

import numpy as np
import pytest

from deep_anc.realtime import run_realtime
from deep_anc.realtime.engines import FxLMSEngine
from deep_anc.realtime.run_realtime import RealtimeANC, fxlms_adaptation_allowed


def test_start_on_true_is_rejected_before_hardware_initialization():
    """설정 오버라이드로 ANC ON 시작을 우회할 수 없어야 한다."""
    with pytest.raises(ValueError, match="start_on=true"):
        RealtimeANC({"start_on": True})


def test_fxlms_engine_includes_thread_handoff_and_starts_adaptation_off(tmp_path):
    secondary = tmp_path / "secondary.npz"
    np.savez(
        secondary,
        fir=np.array([1.0], dtype=np.float32),
        delay_samples=np.array(3),
        sample_rate=np.array(48_000),
    )
    engine = FxLMSEngine(
        str(secondary),
        {"control_length": 4, "mu": 0.01, "leakage": 0.0},
        hop=4,
        handoff_extra_samples=4,
    )
    assert engine.secondary_delay_samples == 7
    assert engine.secondary_total_length == 8
    assert engine.controller.secondary_delay_samples == 7
    assert engine.adapt is False

    ref = np.linspace(-0.1, 0.1, 4, dtype=np.float32)
    err = np.full(4, 0.02, dtype=np.float32)
    for _ in range(4):
        engine.step(ref, err)
    assert engine.controller.update_count == 0

    engine.set_adapt_enabled(True)
    engine.step(ref, err)
    assert engine.controller.update_count == 1
    engine.reset()
    assert engine.adapt is False


def test_fxlms_adaptation_gate_is_fail_closed():
    ready = {
        "requested": True,
        "full_anc_gain": True,
        "full_noise_gain": True,
        "hold_samples": 0,
        "output_clip_fraction": 0.0,
        "input_clip_fraction": 0.0,
        "reference_power": 1.0e-3,
        "stream_ok": True,
    }
    assert fxlms_adaptation_allowed(**ready)

    unsafe_values = {
        "requested": False,
        "full_anc_gain": False,
        "full_noise_gain": False,
        "hold_samples": 1,
        "output_clip_fraction": 0.01,
        "input_clip_fraction": 0.01,
        "reference_power": 0.0,
        "stream_ok": False,
    }
    for key, unsafe in unsafe_values.items():
        case = dict(ready)
        case[key] = unsafe
        assert not fxlms_adaptation_allowed(**case), key


def test_runtime_input_preflight_rejects_stuck_error_channel(monkeypatch):
    def fake_probe(*_args, **_kwargs):
        base = {
            "rms_dbfs": -186.64,
            "peak": 0.0,
            "clip_ratio": 0.0,
            "unique_codes": 1,
            "raw_min": -1,
            "raw_max": -1,
        }
        return {
            "channels": [
                dict(base, channel=0, valid=False),
                dict(base, channel=1, valid=False),
            ]
        }

    monkeypatch.setattr(run_realtime, "capture_input_probe", fake_probe)
    cfg = {"reference": "digital", "hardware": {"audio": {}}}
    assert run_realtime.input_preflight(cfg, seconds=0.1) is False
