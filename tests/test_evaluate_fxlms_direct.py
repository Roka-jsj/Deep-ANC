"""direct-callback FxLMS 평가기의 안전·기록·품질 게이트."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.demo import evaluate_fxlms_direct as direct


def _program(**overrides):
    values = {
        "sample_rate": 100,
        "frequency": 10.0,
        "amplitude": 0.005,
        "noise_delay_ms": 70.0,
        "off_seconds": 1.5,
        "on_seconds": 2.0,
        "tail_off_seconds": 1.0,
        "pre_silence_seconds": 0.5,
        "post_silence_seconds": 0.5,
        "fade_seconds": 0.1,
    }
    values.update(overrides)
    return direct.build_program(**values)


def _preflight(valid=True):
    return {
        "channels": [
            {"channel": 0, "valid": valid, "clip_count": 0},
            {"channel": 1, "valid": valid, "clip_count": 0},
        ]
    }


def _telemetry(**overrides):
    values = direct._empty_telemetry(100)
    values.update(
        {
            "completed": True,
            "terminal": True,
            "stream_started": True,
            "stream_closed": True,
            "captured_frames": 100,
            "adaptation_enabled_frames": 100,
            "adaptation_adapted_frames": 100,
            "adaptation_update_segments": 2,
            "callback_timestamps": {"stable": True, "invalid_reasons": []},
        }
    )
    values.update(overrides)
    return values


def _flush(**overrides):
    values = {
        "attempted": True,
        "zero_blocks_requested": 4,
        "zero_blocks_written": 4,
        "underflow_blocks": 0,
        "both_channels_zero": True,
        "stream_closed": True,
        "error": None,
    }
    values.update(overrides)
    return values


def _metrics(attenuation=1.0):
    return {
        "available": True,
        "error": {
            "rms_attenuation_db": attenuation,
            "off_return_tone_change_db": 0.0,
        },
        "reference_mic": {"off_return_tone_change_db": 0.0},
        "control": {"on_peak": 0.001},
    }


def _secondary(valid=True):
    return {"valid_for_performance_claim": valid, "invalid_reasons": []}


class _FakeController:
    def __init__(self):
        self.w = np.zeros(8, dtype=np.float32)
        self.calls = []

    def generate_block(self, source):
        return np.asarray(source, dtype=np.float32) * np.float32(0.5)

    def adapt_block(self, error, enabled=True):
        self.calls.append((bool(enabled), int(np.asarray(error).size)))
        if enabled:
            self.w[0] += np.float32(1.0e-5)
        return SimpleNamespace(
            adapted=bool(enabled),
            weight_limited=False,
            weight_norm=float(np.linalg.norm(self.w)),
        )


class _Status:
    def __init__(self, *, xrun=False):
        self.input_underflow = False
        self.input_overflow = False
        self.output_underflow = bool(xrun)
        self.output_overflow = False
        self.priming_output = False

    def __bool__(self):
        return any(
            (
                self.input_underflow,
                self.input_overflow,
                self.output_underflow,
                self.output_overflow,
                self.priming_output,
            )
        )

    def __str__(self):
        return "output underflow" if self.output_underflow else ""


class _FakeSD:
    class CallbackStop(Exception):
        pass

    class CallbackAbort(Exception):
        pass

    def __init__(self, *, xrun_first=False, clipped_first=False):
        owner = self

        class Stream:
            def __init__(self, **kwargs):
                self.callback = kwargs["callback"]
                self.blocksize = int(kwargs["blocksize"])
                self.aborted = False
                self.closed = False
                self.outputs = []
                owner.stream = self

            def start(self):
                block_index = 0
                while True:
                    raw = np.zeros((self.blocksize, 2), dtype=np.int32)
                    raw[:, 0] = 100_000 + np.arange(self.blocksize, dtype=np.int32)
                    raw[:, 1] = -200_000 - np.arange(self.blocksize, dtype=np.int32)
                    if owner.clipped_first and block_index == 0:
                        raw[0, 0] = np.iinfo(np.int32).max
                    output = np.full((self.blocksize, 2), 123, dtype=np.int16)
                    status = _Status(xrun=owner.xrun_first and block_index == 0)
                    adc_time = 10.0 + block_index * self.blocksize / 100.0
                    time_info = SimpleNamespace(
                        inputBufferAdcTime=adc_time,
                        currentTime=adc_time + 0.05,
                        outputBufferDacTime=adc_time + 0.10,
                    )
                    try:
                        self.callback(raw, output, self.blocksize, time_info, status)
                    except owner.CallbackStop:
                        self.outputs.append(output.copy())
                        break
                    except owner.CallbackAbort:
                        self.outputs.append(output.copy())
                        break
                    self.outputs.append(output.copy())
                    block_index += 1
                    if block_index > 1000:  # pragma: no cover - 회귀 안전망
                        raise RuntimeError("callback did not terminate")

            def abort(self):
                self.aborted = True

            def close(self):
                self.closed = True

        self.Stream = Stream
        self.xrun_first = xrun_first
        self.clipped_first = clipped_first
        self.stream = None


def _run_fake(fake_sd, controller=None):
    return direct.run_direct_session(
        fake_sd,
        controller=controller or _FakeController(),
        sample_rate=100,
        block_size=20,
        latency="high",
        input_device=1,
        output_device=2,
        noise_output_channel=0,
        cancel_output_channel=1,
        program=_program(),
        control_limit=0.10,
        dc_block_r=0.995,
        max_timestamp_jitter_seconds=0.001,
    )


def test_defaults_match_low_level_legacy_direct_protocol():
    args = direct.build_parser().parse_args([])
    assert args.secondary_path == "assets/measured/secondary_path_legacy_512high.npz"
    assert args.frequency == 300.0
    assert args.amplitude == 0.005
    assert args.noise_delay_ms == 70.0
    assert args.mu == 0.001
    assert args.control_limit == 0.10
    assert args.control_len == 256
    assert args.block_size == 512
    assert args.latency == "high"
    assert args.max_timestamp_jitter_ms == 1.0
    assert (args.off_seconds, args.on_seconds, args.tail_off_seconds) == (10.0, 30.0, 5.0)
    assert args.confirm_user_present_volume_minimum is False


def test_confirmation_precedes_config_and_hardware_access(monkeypatch, capsys):
    monkeypatch.setattr(
        direct,
        "load_yaml",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("config accessed")),
    )
    assert direct.main([]) == 2
    assert "--confirm-user-present-volume-minimum" in capsys.readouterr().err


def test_validation_rejects_amplitude_above_hard_maximum_and_external_paths(tmp_path):
    args = direct.build_parser().parse_args(["--amplitude", "0.0201"])
    with pytest.raises(ValueError, match="amplitude"):
        direct.validate_options(args, 48_000)
    with pytest.raises(ValueError, match="Deep_ANC"):
        direct._repo_path(tmp_path / "outside.npz")
    args = direct.build_parser().parse_args(["--max-timestamp-jitter-ms", "0"])
    with pytest.raises(ValueError, match="timestamp"):
        direct.validate_options(args, 48_000)


def test_program_has_silence_fades_exact_protocol_and_70ms_playback_delay():
    program = _program()
    reference = program["reference"]
    noise = program["noise_playback"]
    gain = program["scheduled_gain"]
    bounds = program["bounds"]
    assert program["delay_samples"] == 7
    assert np.all(reference[: bounds["pre_silence"][1]] == 0.0)
    assert np.all(reference[bounds["post_silence"][0] :] == 0.0)
    assert np.all(noise[:7] == 0.0)
    np.testing.assert_array_equal(noise[7:], reference[:-7])
    assert np.all(gain[: bounds["on"][0]] == 0.0)
    assert np.all(gain[bounds["on"][1] :] == 0.0)
    assert 0.0 <= gain[bounds["on"][0]] < 1.0
    assert np.max(gain[bounds["on"][0] : bounds["on"][1]]) == 1.0


def test_direct_callback_completes_and_adapts_only_scheduled_on_frames():
    fake_sd = _FakeSD()
    controller = _FakeController()
    capture = _run_fake(fake_sd, controller)
    telemetry = capture["telemetry"]
    assert telemetry["completed"] is True
    assert telemetry["stream_closed"] is True
    assert telemetry["xrun_count"] == 0
    assert telemetry["input_clip_count"] == 0
    enabled_frames = sum(frames for enabled, frames in controller.calls if enabled)
    assert enabled_frames == int(np.count_nonzero(_program()["scheduled_on"]))
    assert telemetry["adaptation_enabled_frames"] == enabled_frames
    assert telemetry["adaptation_adapted_frames"] == enabled_frames
    assert telemetry["callback_timestamps"]["stable"] is True
    assert len(telemetry["callback_input_buffer_adc_time"]) == telemetry["callback_count"]
    assert all(np.all(block[:, 0] != 123) for block in fake_sd.stream.outputs)


def _timestamp_telemetry(*, adc, current, dac, frame_start=None, frames=None):
    count = len(adc)
    return {
        "callback_count": count,
        "callback_frame_start": list(
            np.arange(count) * 20 if frame_start is None else frame_start
        ),
        "callback_frames": list(np.full(count, 20) if frames is None else frames),
        "callback_input_buffer_adc_time": list(adc),
        "callback_current_time": list(current),
        "callback_output_buffer_dac_time": list(dac),
    }


def test_timestamp_summary_accepts_stable_progression_and_offset():
    telemetry = _timestamp_telemetry(
        adc=[10.0, 10.2, 10.4],
        current=[10.05, 10.25, 10.45],
        dac=[10.1, 10.3, 10.5],
    )
    report = direct.summarize_callback_timestamps(
        telemetry, sample_rate=100, max_jitter_seconds=0.001
    )
    assert report["stable"] is True
    assert report["dac_minus_adc_seconds"]["median"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("telemetry", "reason"),
    [
        (
            _timestamp_telemetry(
                adc=[0.0, 0.0, 0.0],
                current=[0.0, 0.0, 0.0],
                dac=[0.0, 0.0, 0.0],
            ),
            "callback_timestamps_non_positive",
        ),
        (
            _timestamp_telemetry(
                adc=[10.0, 10.0, 10.0],
                current=[10.05, 10.05, 10.05],
                dac=[10.1, 10.1, 10.1],
            ),
            "callback_timestamps_non_progressing",
        ),
        (
            _timestamp_telemetry(
                adc=[10.0, float("nan"), 10.4],
                current=[10.05, 10.25, 10.45],
                dac=[10.1, 10.3, 10.5],
            ),
            "callback_timestamps_missing_or_non_finite",
        ),
        (
            _timestamp_telemetry(
                adc=[10.0, 10.205, 10.4],
                current=[10.05, 10.255, 10.45],
                dac=[10.1, 10.305, 10.5],
            ),
            "callback_timestamp_progression_jitter_exceeded",
        ),
        (
            _timestamp_telemetry(
                adc=[10.0, 10.2, 10.4],
                current=[10.05, 10.25, 10.45],
                dac=[10.1, 10.302, 10.5],
            ),
            "callback_dac_adc_offset_jitter_exceeded",
        ),
        (
            _timestamp_telemetry(
                adc=[10.0, 10.2, 10.4],
                current=[10.05, 10.25, 10.45],
                dac=[10.1, 10.3, 10.5],
                frame_start=[0, 20, 50],
            ),
            "callback_frame_sequence_discontinuous",
        ),
        (
            {
                **_timestamp_telemetry(
                    adc=[10.0, 10.2, 10.4],
                    current=[10.05, 10.25, 10.45],
                    dac=[10.1, 10.3, 10.5],
                ),
                "callback_count": 4,
            },
            "callback_timestamp_count_mismatch",
        ),
    ],
)
def test_timestamp_summary_rejects_invalid_or_unstable_clocks(telemetry, reason):
    report = direct.summarize_callback_timestamps(
        telemetry, sample_rate=100, max_jitter_seconds=0.001
    )
    assert report["stable"] is False
    assert reason in report["invalid_reasons"]


def test_timestamp_instability_invalidates_quality_gate():
    telemetry = _telemetry(
        callback_timestamps={
            "stable": False,
            "invalid_reasons": ["callback_timestamps_non_progressing"],
        }
    )
    result = direct.quality_gate(
        preflight_report=_preflight(),
        telemetry=telemetry,
        final_flush=_flush(),
        on_duty=0.99,
        metrics=_metrics(),
        secondary_path=_secondary(),
    )
    assert result["measurement_valid"] is False
    assert "callback_timestamps_unstable" in result["measurement_invalid_reasons"]
    assert "callback_timestamps_non_progressing" in result["measurement_invalid_reasons"]


@pytest.mark.parametrize(
    ("fake_sd", "reason"),
    [
        (_FakeSD(xrun_first=True), "xrun_detected"),
        (_FakeSD(clipped_first=True), "input_clipping"),
    ],
)
def test_xrun_or_clipping_immediately_forces_both_outputs_zero(fake_sd, reason):
    capture = _run_fake(fake_sd)
    telemetry = capture["telemetry"]
    assert telemetry["completed"] is False
    assert telemetry["safety_latched"] is True
    assert reason in telemetry["safety_reasons"]
    assert np.all(fake_sd.stream.outputs[0] == 0)
    assert np.all(capture["control"] == 0.0)


def test_quality_gate_requires_duty_xrun_clip_and_complete_termination():
    valid = direct.quality_gate(
        preflight_report=_preflight(),
        telemetry=_telemetry(),
        final_flush=_flush(),
        on_duty=0.99,
        metrics=_metrics(),
        secondary_path=_secondary(),
    )
    assert valid["measurement_valid"] is True
    assert valid["performance_success"] is True

    for telemetry, duty, flush, expected in (
        (_telemetry(xrun_count=1), 0.99, _flush(), "xrun_detected"),
        (_telemetry(input_clip_count=1), 0.99, _flush(), "input_clipping"),
        (_telemetry(completed=False), 0.99, _flush(), "measurement_incomplete"),
        (
            _telemetry(control_limit_count=1),
            0.99,
            _flush(),
            "control_output_hard_limited",
        ),
        (
            _telemetry(adaptation_adapted_frames=94),
            0.99,
            _flush(),
            "adaptation_duty_below_95_percent",
        ),
        (_telemetry(), 0.949, _flush(), "on_duty_below_95_percent"),
        (_telemetry(), 0.99, _flush(stream_closed=False), "final_zero_flush_incomplete"),
    ):
        result = direct.quality_gate(
            preflight_report=_preflight(),
            telemetry=telemetry,
            final_flush=flush,
            on_duty=duty,
            metrics=_metrics(),
            secondary_path=_secondary(),
        )
        assert result["measurement_valid"] is False
        assert expected in result["measurement_invalid_reasons"]
        assert result["performance_success"] is False


def test_invalid_secondary_path_blocks_claim_but_preserves_diagnostic_reduction():
    result = direct.quality_gate(
        preflight_report=_preflight(),
        telemetry=_telemetry(),
        final_flush=_flush(),
        on_duty=0.99,
        metrics=_metrics(attenuation=2.0),
        secondary_path={
            "valid_for_performance_claim": False,
            "invalid_reasons": ["secondary_path_delay_stability_unverified"],
        },
    )
    assert result["measurement_valid"] is True
    assert result["fxlms_reduction_observed"] is True
    assert result["performance_claim_allowed"] is False
    assert result["performance_success"] is False
    assert "secondary_path_not_validated_for_performance" in result[
        "performance_claim_block_reasons"
    ]


def test_default_legacy_secondary_path_is_explicitly_diagnostic_only():
    path = direct.REPO_ROOT / direct.DEFAULT_SECONDARY_PATH
    model = direct.load_secondary_path(path)
    report = direct.assess_secondary_path(
        path,
        model=model,
        block_size=512,
        latency="high",
        frequency=300.0,
    )
    assert report["valid_for_performance_claim"] is False
    assert "secondary_path_not_official_ess" in report["invalid_reasons"]
    assert "성능 주장에는 사용 금지" in report["interpretation"]


def test_final_zero_flush_runs_when_direct_session_raises(monkeypatch):
    calls = []

    def fail(*_args, **_kwargs):
        raise RuntimeError("fake measurement failure")

    def flush(*_args, **_kwargs):
        calls.append("flush")
        return _flush()

    monkeypatch.setattr(direct, "run_direct_session", fail)
    monkeypatch.setattr(direct, "flush_output_silence", flush)
    with pytest.raises(RuntimeError, match="fake measurement failure"):
        direct.collect_with_final_flush(object(), run_kwargs={}, flush_kwargs={})
    assert calls == ["flush"]


def test_matched_metrics_use_equal_late_off_and_on_windows():
    sample_rate = 100
    program = _program()
    total = int(program["total_frames"])
    raw = np.zeros((total, 2), dtype=np.int32)
    off_start, off_stop = program["bounds"]["initial_off"]
    on_start, on_stop = program["bounds"]["on"]
    phase = np.arange(total) / sample_rate
    raw[:, 0] = np.rint(0.02 * np.sin(2 * np.pi * 10 * phase) * 2**31).astype(np.int32)
    raw[on_start:on_stop, 0] //= 2
    raw[:, 1] = raw[:, 0]
    capture = {
        "raw_input": raw,
        "reference": program["reference"],
        "control": np.zeros(total, dtype=np.float32),
    }
    metrics = direct.compute_matched_metrics(
        capture,
        bounds=program["bounds"],
        sample_rate=sample_rate,
        frequency=10.0,
        analysis_seconds=0.5,
        guard_seconds=0.1,
    )
    assert metrics["available"] is True
    assert metrics["off_indices"][1] - metrics["off_indices"][0] == metrics[
        "on_indices"
    ][1] - metrics["on_indices"][0]
    assert metrics["error"]["rms_attenuation_db"] == pytest.approx(6.02, abs=0.1)


def test_results_store_all_raw_traces_weights_and_refuse_overwrite(tmp_path):
    session = tmp_path / "session"
    capture = {
        "raw_input": np.arange(40, dtype=np.int32).reshape(20, 2),
        "output_pcm": np.zeros((20, 2), dtype=np.int16),
        "reference": np.ones(20, dtype=np.float32),
        "noise_playback": np.ones(20, dtype=np.float32),
        "control": np.zeros(20, dtype=np.float32),
        "control_unlimited": np.zeros(20, dtype=np.float32),
        "gain": np.zeros(20, dtype=np.float32),
        "scheduled_on": np.zeros(20, dtype=np.bool_),
        "weights": np.arange(8, dtype=np.float32),
        "telemetry": {
            "callback_frame_start": [0, 10],
            "callback_frames": [10, 10],
            "callback_input_buffer_adc_time": [1.0, 1.1],
            "callback_current_time": [1.05, 1.15],
            "callback_output_buffer_dac_time": [1.1, 1.2],
        },
    }
    npz_path, json_path, weights_path = direct.save_results(
        session,
        preflight_raw=np.zeros((4, 2), dtype=np.int32),
        capture=capture,
        metadata={"quality": {"performance_claim_allowed": False}},
    )
    assert json_path.exists() and weights_path.exists()
    with np.load(npz_path, allow_pickle=False) as data:
        assert {
            "err_raw_int32",
            "ref_raw_int32",
            "source",
            "control",
            "gain",
            "weights",
            "callback_frame_start",
            "callback_input_buffer_adc_time",
            "callback_output_buffer_dac_time",
            "metadata_json",
        } <= set(data.files)
        np.testing.assert_allclose(data["callback_input_buffer_adc_time"], [1.0, 1.1])
    with pytest.raises(FileExistsError):
        direct.save_results(
            session,
            preflight_raw=np.zeros((4, 2), dtype=np.int32),
            capture=capture,
            metadata={},
        )
