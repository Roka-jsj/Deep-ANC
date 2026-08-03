"""광대역 P/S 경로 측정의 안전·승격 게이트."""

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from scripts.data import calibrate_wideband as calibration


def _args(**overrides) -> Namespace:
    values = {
        "confirm_volume_minimum": True,
        "repeats": 3,
        "amplitude": 0.005,
        "band": [80.0, 1600.0],
        "sweep_seconds": 4.0,
        "input_probe_seconds": 2.0,
        "fir_length": 2048,
        "pre_roll": 32,
        "max_delay_ms": 250.0,
        "max_delay_jitter_ms": 1.0,
        "block_size": 512,
        "output_channel": "cancel",
        "out": "results/test_calibration_model_never_created.npz",
        "diagnostics_root": "results/calibration_wideband",
    }
    values.update(overrides)
    return Namespace(**values)


def _channel_report(*, valid: bool = True, clip_ratio: float = 0.0) -> dict:
    return {
        "channel": 0,
        "valid": valid,
        "clip_ratio": clip_ratio,
    }


def _reports(*, valid: bool = True, clip_ratio: float = 0.0) -> dict:
    return {
        "channels": [
            _channel_report(valid=valid, clip_ratio=clip_ratio),
            {**_channel_report(valid=valid, clip_ratio=clip_ratio), "channel": 1},
        ]
    }


def test_defaults_are_low_level_banded_and_high_latency():
    args = calibration.build_parser().parse_args([])
    assert args.band == [80.0, 1600.0]
    assert args.amplitude == 0.005
    assert args.repeats == 3
    assert args.latency == "high"
    assert args.max_delay_jitter_ms == 1.0
    assert args.confirm_volume_minimum is False


def test_confirmation_is_checked_before_hardware_access(monkeypatch, capsys):
    monkeypatch.setattr(
        calibration,
        "load_yaml",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("hardware accessed")),
    )
    assert calibration.main([]) == 2
    assert "--confirm-volume-minimum" in capsys.readouterr().err


def test_options_reject_unsafe_amplitude_and_too_few_repeats():
    for args in (_args(amplitude=0.0201), _args(repeats=2), _args(block_size=0)):
        try:
            calibration.validate_options(args, 48_000)
        except ValueError:
            pass
        else:  # pragma: no cover - 회귀 시 읽기 쉬운 실패 메시지
            raise AssertionError("unsafe calibration options were accepted")


def test_options_reject_paths_outside_repository(tmp_path):
    args = _args(out=str(tmp_path / "model.npz"))
    try:
        calibration.validate_options(args, 48_000)
    except ValueError as exc:
        assert "Deep_ANC" in str(exc)
    else:
        raise AssertionError("outside-repository output was accepted")


def test_xrun_or_low_consistency_blocks_both_path_models():
    output = np.zeros((32, 2), dtype=np.float32)
    output[:, 1] = 0.005
    output_pcm = calibration.float32_to_pcm_int16(output)

    for xrun_count, consistency, expected in (
        (1, 0.99, "xrun_detected"),
        (0, 0.89, "repeat_consistency_below_0.9"),
    ):
        valid, reasons, _ = calibration.quality_gate(
            preflight_report=_reports(),
            measurement_report=_reports(),
            output_float=output,
            output_pcm=output_pcm,
            telemetry={
                "xrun_count": xrun_count,
                "unexpected_status_count": 0,
                "completed": True,
            },
            consistency=consistency,
            model_available=True,
        )
        assert valid is False
        assert expected in reasons


def test_repeat_delay_spread_above_one_ms_blocks_promotion():
    output = np.zeros((32, 2), dtype=np.float32)
    output[:, 1] = 0.005
    common = {
        "preflight_report": _reports(),
        "measurement_report": _reports(),
        "output_float": output,
        "output_pcm": calibration.float32_to_pcm_int16(output),
        "telemetry": {
            "xrun_count": 0,
            "unexpected_status_count": 0,
            "completed": True,
        },
        "consistency": 0.99,
        "model_available": True,
        "max_delay_jitter_samples": 48,
    }

    valid, reasons, _ = calibration.quality_gate(
        **common,
        delay_spread_samples=49,
    )
    assert valid is False
    assert "repeat_delay_spread_exceeds_limit" in reasons

    valid, reasons, _ = calibration.quality_gate(
        **common,
        delay_spread_samples=48,
    )
    assert valid is True
    assert "repeat_delay_spread_exceeds_limit" not in reasons


def _synthetic_repeat_irs(delays: list[int]) -> np.ndarray:
    rng = np.random.default_rng(20260803)
    length = 3000
    taps = np.arange(320, dtype=np.float64)
    response = np.exp(-taps / 110.0) * np.sin(2 * np.pi * (taps + 3) / 31.0)
    repeats = []
    for delay in delays:
        ir = rng.normal(0.0, 1e-4, length)
        # 기존 단일 5% 방식이면 이 spike를 onset으로 잘못 고른다.
        ir[80] += 0.06
        ir[delay : delay + response.size] += response
        repeats.append(ir)
    return np.stack(repeats)


def test_robust_energy_onset_ignores_single_spike_and_aligns_stable_repeats():
    irs = _synthetic_repeat_irs([700, 710, 692])
    model, consistency, correlations, error = calibration._model_from_repeat_irs(
        irs,
        max_delay_samples=1500,
        fir_length=512,
        pre_roll=32,
        max_delay_jitter_samples=48,
    )

    assert error is None
    assert model["stable_delay"] is True
    assert min(model["repeat_onset_samples"]) > 600
    assert model["delay_spread_samples"] <= 48
    assert model["delay_samples"] == int(np.median(model["repeat_delay_samples"]))
    assert model["fir"].shape == (512,)
    assert consistency > 0.99
    assert min(correlations) > 0.99


def test_unstable_synthetic_repeat_delays_do_not_build_fir():
    irs = _synthetic_repeat_irs([700, 710, 790])
    model, _consistency, _correlations, error = calibration._model_from_repeat_irs(
        irs,
        max_delay_samples=1500,
        fir_length=512,
        pre_roll=32,
        max_delay_jitter_samples=48,
    )

    assert model["stable_delay"] is False
    assert model["delay_spread_samples"] > 48
    assert "fir" not in model
    assert error is not None and "지터" in error


def test_existing_invalid_noise_raw_is_rejected_by_robust_delay_gate():
    raw_path = Path(
        "results/calibration_wideband/"
        "20260803_200859_977980_noise/raw_measurement.npz"
    )
    if not raw_path.exists():
        pytest.skip("현장 raw 진단 파일이 없는 환경")

    with np.load(raw_path, allow_pickle=False) as data:
        irs = data["repeat_irs"]
    model, consistency, correlations, error = calibration._model_from_repeat_irs(
        irs,
        max_delay_samples=12_000,
        fir_length=2048,
        pre_roll=32,
        max_delay_jitter_samples=48,
    )

    assert model["repeat_onset_samples"] == [2274, 2339, 2361]
    assert model["repeat_delay_samples"] == [2242, 2307, 2329]
    assert model["delay_spread_samples"] == 87
    assert model["stable_delay"] is False
    assert "fir" not in model
    assert consistency == pytest.approx(0.15201470917802323)
    assert correlations == pytest.approx(
        [0.068555936427638, 0.41453983316530385, -0.027051642058872158]
    )
    assert error is not None and "87 > 48" in error


def test_input_and_output_clipping_block_promotion():
    clipped_output = np.ones((16, 2), dtype=np.float32)
    valid, reasons, summary = calibration.quality_gate(
        preflight_report=_reports(),
        measurement_report=_reports(clip_ratio=0.01),
        output_float=clipped_output,
        output_pcm=calibration.float32_to_pcm_int16(clipped_output),
        telemetry={"xrun_count": 0, "unexpected_status_count": 0, "completed": True},
        consistency=0.99,
        model_available=True,
    )
    assert valid is False
    assert "input_clipping" in reasons
    assert "output_clipping" in reasons
    assert summary["pcm_saturation_ratio"] > 0.0


def test_invalid_result_never_creates_official_npz(tmp_path):
    target = tmp_path / "official.npz"
    saved = calibration.save_official_model(
        target,
        valid=False,
        arrays={"fir": np.ones(8, dtype=np.float32)},
    )
    assert saved is False
    assert not target.exists()


def test_diagnostics_always_include_output_err_ref_and_metadata(tmp_path):
    session = tmp_path / "diagnostic"
    raw = np.arange(40, dtype=np.int32).reshape(20, 2)
    output = np.zeros((20, 2), dtype=np.float32)
    output[:, 0] = 0.005
    npz_path, json_path = calibration.save_diagnostics(
        session,
        output=output,
        output_pcm=calibration.float32_to_pcm_int16(output),
        recorded_raw=raw,
        preflight_raw=raw[:4],
        repeat_irs=np.ones((3, 8)),
        metadata={"result": {"valid_for_model": False}, "error": "test"},
    )
    assert json_path.exists()
    with np.load(npz_path, allow_pickle=False) as data:
        assert {"output", "err", "ref", "metadata_json"} <= set(data.files)
        assert data["output"].shape == (20, 2)
        assert data["err"].shape == (20,)
        assert data["ref"].shape == (20,)
        assert "valid_for_model" in str(data["metadata_json"].item())


def test_stream_is_aborted_and_closed_when_start_raises():
    created = []

    class FakeStream:
        def __init__(self, **_kwargs):
            self.aborted = False
            self.closed = False
            created.append(self)

        def start(self):
            raise RuntimeError("fake start failure")

        def abort(self):
            self.aborted = True

        def close(self):
            self.closed = True

    class FakeSD:
        Stream = FakeStream

        class CallbackStop(Exception):
            pass

        class CallbackAbort(Exception):
            pass

    try:
        calibration._capture_measurement(
            FakeSD,
            fs=48_000,
            block_size=512,
            latency="high",
            in_dev=1,
            out_dev=2,
            output_float=np.zeros((32, 2), dtype=np.float32),
        )
    except RuntimeError as exc:
        assert "fake start failure" in str(exc)
    else:
        raise AssertionError("fake stream start failure was swallowed")

    assert len(created) == 1
    assert created[0].aborted is True
    assert created[0].closed is True


def test_diagnostics_refuse_overwrite(tmp_path):
    session = tmp_path / "diagnostic"
    session.mkdir()
    (session / "metadata.json").write_text("{}", encoding="utf-8")
    try:
        calibration.save_diagnostics(
            session,
            output=np.empty((0, 2), np.float32),
            output_pcm=np.empty((0, 2), np.int16),
            recorded_raw=np.empty((0, 2), np.int32),
            preflight_raw=np.empty((0, 2), np.int32),
            repeat_irs=np.empty((0, 0)),
            metadata={},
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("diagnostic overwrite was accepted")
