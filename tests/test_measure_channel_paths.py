"""저레벨 출력-마이크 채널 경로 진단의 안전 경계와 분석 테스트."""

import numpy as np
import pytest

from scripts.bench import measure_channel_paths as paths


def test_program_drives_exactly_one_channel_with_silent_boundaries():
    output, bounds = paths.build_output_program(
        sample_rate=48_000,
        frequency=300.0,
        amplitude=0.005,
        output_channel=0,
        pre_seconds=0.5,
        tone_seconds=1.0,
        post_seconds=0.5,
    )

    assert output.dtype == np.int16
    assert np.all(output[bounds["pre"][0] : bounds["pre"][1]] == 0)
    assert np.all(output[bounds["post"][0] : bounds["post"][1]] == 0)
    assert np.all(output[:, 1] == 0)
    assert np.any(output[bounds["tone"][0] : bounds["tone"][1], 0] != 0)
    assert np.max(np.abs(output[:, 0].astype(np.int32))) <= round(0.005 * 32767)


def test_amplitude_has_hard_safety_ceiling():
    with pytest.raises(ValueError, match="0.020"):
        paths.validate_probe_settings(
            sample_rate=48_000,
            frequency=300.0,
            amplitude=0.02001,
            block_size=512,
            pre_seconds=1.0,
            tone_seconds=2.0,
            post_seconds=1.0,
        )


def test_analysis_detects_tone_coupling_without_calling_it_performance():
    fs = 8_000
    pre = fs
    tone = 2 * fs
    post = fs
    total = pre + tone + post
    rng = np.random.default_rng(12)
    floating = rng.normal(0.0, 2.0e-5, size=(total, 2))
    t = np.arange(tone, dtype=np.float64) / fs
    floating[pre : pre + tone, 0] += 0.01 * np.sin(2 * np.pi * 300.0 * t)
    raw = np.rint(np.clip(floating, -1.0, 1.0) * (2**31)).astype(np.int32)
    bounds = {"pre": (0, pre), "tone": (pre, pre + tone), "post": (pre + tone, total)}

    report = paths.analyze_path_capture(
        raw,
        bounds=bounds,
        frequency=300.0,
        sample_rate=fs,
        status_report={"status_blocks": 0},
    )

    assert report[0]["coupling_detected"] is True
    assert report[0]["tone_bin_change_db"] > 30.0
    assert report[0]["tone_steady"]["clip_ratio"] == 0.0
    assert report[1]["coupling_detected"] is False


def test_callback_status_invalidates_coupling_verdict():
    fs = 8_000
    bounds = {"pre": (0, fs), "tone": (fs, 3 * fs), "post": (3 * fs, 4 * fs)}
    raw = np.zeros((4 * fs, 2), dtype=np.int32)
    t = np.arange(2 * fs) / fs
    raw[fs : 3 * fs, 0] = np.rint(
        0.01 * np.sin(2 * np.pi * 300.0 * t) * (2**31)
    ).astype(np.int32)

    report = paths.analyze_path_capture(
        raw,
        bounds=bounds,
        frequency=300.0,
        sample_rate=fs,
        status_report={"status_blocks": 1, "xrun_blocks": 1},
    )
    assert report[0]["tone_bin_change_db"] > 50.0
    assert report[0]["coupling_detected"] is False


def test_finally_flushes_both_outputs_when_second_probe_fails(monkeypatch):
    calls = []

    def fake_run(_sd, **kwargs):
        calls.append("run")
        if len(calls) == 2:
            raise RuntimeError("simulated callback failure")
        frames = kwargs["output_pcm"].shape[0]
        return np.zeros((frames, 2), dtype=np.int32), {}, 0.1

    def fake_flush(_sd, **_kwargs):
        calls.append("flush")
        return {"both_channels_zero": True}

    monkeypatch.setattr(paths, "run_output_probe", fake_run)
    monkeypatch.setattr(paths, "flush_output_silence", fake_flush)
    program = np.zeros((1024, 2), dtype=np.int16)

    with pytest.raises(RuntimeError, match="simulated"):
        paths.collect_channel_paths(
            object(),
            input_device=1,
            output_device=2,
            sample_rate=48_000,
            block_size=512,
            latency="high",
            programs=[("noise_out", 0, program), ("cancel_out", 1, program)],
        )
    assert calls == ["run", "run", "flush"]


def test_cli_requires_user_presence_and_minimum_volume():
    assert paths.main([]) == 2
