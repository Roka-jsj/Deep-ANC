#!/usr/bin/env python3
"""P(z)와 S(z)를 **한 번의 재생으로 동시에** 측정한다.

왜 순차 ESS 로는 안 되는가
--------------------------
재생은 USB(AB13X), 녹음은 Tegra APE I²S 다. 클록 도메인이 서로 달라 "출력 샘플 번호 ↔
녹음 샘플 번호" 대응이 시간에 따라 흔들린다(wander). 저장된 측정 4건을 재분석한 결과가
이를 못박는다 — 재생 프로그램 기준 반복 간 coherence 0.08~0.17, 같은 녹음을 ERR/REF
기준으로 보면 0.9915~0.9976, |H| 반복 std 0.08dB. 즉 **깨진 것은 시간축 대응 하나뿐**이다.
자극 진폭을 4배 올려도 개선이 없었다는 사실이 레벨 가설을 직접 반증한다.

``calibrate_wideband.py`` 는 P 와 S 를 별도 실행으로 잰다. 두 측정이 수십 초 떨어지면
그 사이의 wander 가 **두 경로의 상대 지연**에 그대로 실린다. ANC 가 실제로 요구하는 양이
바로 그 상대 지연(``lead = S_delay + handoff − P_delay``)이므로, 순차 측정은 우리가 가장
필요로 하는 숫자를 가장 크게 틀린다.

해법
----
두 출력 채널은 **같은 DAC·같은 스트림**을 지나므로 warp D(t) 가 동일하다. 정확히 같은
시각에 두 경로를 구동하면 D 는 두 경로에 공통으로 실리고 상대 관계에서 상쇄된다.
동시 재생 상태로 두 응답을 분리하기 위해 주파수를 번갈아 나눈다(guard=1).

    ch0(소음 스피커) → 짝수번째 톤,  ch1(상쇄 스피커) → 홀수번째 톤

정수 주기 FFT 라 빈 집합이 정확히 서로소이고 누설이 0 이다. 시뮬레이션 검증 결과
(``tests/test_interleaved_probe.py``) 실측 wander 3.2샘플에서 상대오차 −26.9dB —
같은 조건의 순차 측정은 −4dB 다.

산출물
------
게이트를 통과하면 P/S NPZ 두 개를 **같은 capture_id** 로 함께 저장한다. 같은 capture 에서
나왔다는 사실이 파일에 박혀 있어야 파인튜닝 진입 감사가 "두 경로가 같은 조건"임을
파일만 보고 확인할 수 있다(``finetune_readiness.audit_official_path_model``).

사용자 입회·앰프 볼륨 최저에서만 실행한다::

  .venv/bin/python scripts/data/measure_paths_interleaved.py --confirm-volume-minimum
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

import calibrate_wideband as cw  # noqa: E402

from deep_anc.audio_io import (  # noqa: E402
    pcm_int32_to_float32,
    resolve_alsa_portaudio_device,
)
from deep_anc.config import REPO_ROOT, load_yaml  # noqa: E402
from deep_anc.dsp.interleaved_probe import (  # noqa: E402
    DEFAULT_TRACK_WINDOW,
    build_interleaved_probe,
    channel_impulse_response,
    dewarp_recording,
    estimate_transfer,
    tone_snr_db,
    track_warp,
)

METHOD = "interleaved_multitone"

# 설계 대역은 필수 대역 [80,1600] 보다 넓게 잡는다. 채널마다 톤이 한 칸씩 어긋나므로
# 딱 맞춰 잡으면 한 채널의 마지막 톤이 상한 안쪽으로 떨어져 대역을 덮지 못한다.
DEFAULT_BAND_HZ = (70.0, 1610.0)
# None = 주파수 분해능 그대로. 그래야 인접 빈이 서로 다른 채널이 되어 guard=1 이 된다.
# guard 를 넓히면 두 경로를 **서로 다른 주파수에서** 보게 되어 동시 측정의 이점을 깎는다.
DEFAULT_TONE_SPACING_HZ = None

# 창 길이는 wander 와의 정면 트레이드오프다. 실측 위상 잔차: 0.77s→0.12rad,
# 1.36s→0.24rad, 2.26s→2.33rad, 3.70s→3.99rad. 2초를 넘기면 급격히 무너진다.
DEFAULT_PERIOD_SECONDS = 1.0
DEFAULT_WARMUP_PERIODS = 2      # 순환 정상상태 도달 전 주기는 버린다
DEFAULT_REPEATS = 8

MIN_TONE_SNR_DB = 12.0          # 톤 중앙값 SNR 하한
MIN_TONE_SNR_FRACTION = 0.9     # 이 비율 이상의 톤이 하한을 넘어야 한다
MAX_CREST_DB = 14.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", default="configs/hardware_jetson.yaml")
    parser.add_argument("--band", type=float, nargs=2, default=list(DEFAULT_BAND_HZ))
    parser.add_argument("--required-band", type=float, nargs=2, default=[80.0, 1600.0])
    parser.add_argument(
        "--tone-spacing-hz",
        type=float,
        default=DEFAULT_TONE_SPACING_HZ,
        help="채널별 톤 간격(Hz). 생략하면 guard=1 이 되는 최소 간격을 쓴다",
    )
    parser.add_argument("--period-seconds", type=float, default=DEFAULT_PERIOD_SECONDS)
    parser.add_argument("--warmup-periods", type=int, default=DEFAULT_WARMUP_PERIODS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--amplitude", type=float, default=cw.MAX_AMPLITUDE)
    parser.add_argument("--fir-length", type=int, default=2048)
    parser.add_argument("--pre-roll", type=int, default=256)
    parser.add_argument("--max-delay-ms", type=float, default=100.0)
    parser.add_argument("--max-delay-jitter-ms", type=float, default=1.0)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--latency", choices=["low", "high"], default="high")
    parser.add_argument("--input-probe-seconds", type=float, default=3.0)
    parser.add_argument("--primary-out", default="assets/measured/primary_path_il.npz")
    parser.add_argument("--secondary-out", default="assets/measured/secondary_path_il.npz")
    parser.add_argument("--diagnostics-root", default="results/calibration_interleaved")
    parser.add_argument(
        "--dewarp",
        action="store_true",
        help=(
            "주기 분석 전에 warp 궤적을 추적해 녹음을 재생 타임베이스로 되돌린다. "
            "실측에서 반복 일관성 0.05 → 0.85 (게이트 0.90 에는 아직 미달)"
        ),
    )
    parser.add_argument("--track-window", type=int, default=DEFAULT_TRACK_WINDOW)
    parser.add_argument("--track-min-peak", type=float, default=0.2)
    parser.add_argument("--confirm-volume-minimum", action="store_true")
    return parser


def analyse_channel(
    *,
    err: np.ndarray,
    probe,
    drive: str,
    period_starts: list[int],
    fir_length: int,
    pre_roll: int,
    max_delay_samples: int,
    max_delay_jitter_samples: int,
) -> tuple[dict[str, Any], np.ndarray, float, list[float], str | None]:
    """주기별 IR 을 뽑아 ``calibrate_wideband`` 와 **같은** 정렬·평균 규칙을 적용한다.

    같은 규칙을 쓰는 것이 중요하다. 게이트가 읽는 ``delay_samples`` /
    ``delay_spread_samples`` 의 정의가 두 측정 도구 사이에서 달라지면, 같은 이름의
    숫자가 다른 뜻을 갖게 되어 lead 계산이 조용히 틀린다.
    """

    irs: list[np.ndarray] = []
    for start in period_starts:
        segment = err[start : start + probe.period_samples]
        _, transfer = estimate_transfer(segment, probe, drive=drive)
        irs.append(
            channel_impulse_response(probe, transfer, drive=drive, pre_roll=pre_roll)
        )
    stack = np.stack(irs)
    model, consistency, correlations, error = cw._model_from_repeat_irs(
        stack,
        max_delay_samples=max_delay_samples + pre_roll,
        fir_length=fir_length,
        pre_roll=pre_roll,
        max_delay_jitter_samples=max_delay_jitter_samples,
    )
    return model, stack, consistency, correlations, error


def channel_quality(
    *,
    model: dict[str, Any],
    consistency: float,
    snr_db: np.ndarray,
    min_consistency: float,
) -> list[str]:
    reasons: list[str] = []
    if not model.get("stable_delay"):
        reasons.append("delay_unstable")
    if not np.isfinite(consistency) or consistency < min_consistency:
        reasons.append(f"consistency_{consistency:.4f}")
    finite = snr_db[np.isfinite(snr_db)]
    if finite.size != snr_db.size or finite.size == 0:
        reasons.append("tone_snr_not_finite")
    else:
        good = float(np.mean(finite >= MIN_TONE_SNR_DB))
        if good < MIN_TONE_SNR_FRACTION:
            reasons.append(f"tone_snr_coverage_{good:.3f}")
    return reasons


def _official_arrays(
    *,
    model: dict[str, Any],
    fs: int,
    consistency: float,
    band_hz: tuple[float, float],
    amplitude: float,
    block_size: int,
    latency: str,
    output_channel: str,
    repeats: int,
    xrun_count: int,
    capture_id: str,
    probe,
    drive: str,
    snr_db: np.ndarray,
    period_seconds: float,
) -> dict[str, Any]:
    return {
        "fir": np.asarray(model["fir"], dtype=np.float32),
        "delay_samples": np.int64(model["delay_samples"]),
        "sample_rate": np.int64(fs),
        "coherence_median": np.float64(consistency),
        "excitation_band_hz": np.asarray(band_hz, dtype=np.float64),
        "calibration_block_size": np.int64(block_size),
        "calibration_latency": np.str_(latency),
        "output_channel": np.str_(output_channel),
        "method": np.str_(METHOD),
        "repeats": np.int64(repeats),
        "amplitude": np.float64(amplitude),
        "xrun_count": np.int64(xrun_count),
        "delay_spread_samples": np.int64(model["delay_spread_samples"]),
        "max_delay_jitter_samples": np.int64(model["max_delay_jitter_samples"]),
        # --- interleaved 전용 (게이트가 method 별로 추가 검사한다) ---
        "capture_id": np.str_(capture_id),
        "interleave_guard_bins": np.int64(probe.guard_bins()),
        "analysis_period_seconds": np.float64(period_seconds),
        "tone_count": np.int64(probe.bins_for(drive).size),
        "tone_snr_median_db": np.float64(float(np.median(snr_db))),
        "tone_snr_min_db": np.float64(float(np.min(snr_db))),
        "tone_frequencies_hz": (
            probe.bins_for(drive) * probe.sample_rate / probe.period_samples
        ).astype(np.float64),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.confirm_volume_minimum:
        print(
            "[중단] 스피커가 울립니다. 사용자 입회와 앰프 볼륨 최저를 확인한 뒤 "
            "--confirm-volume-minimum 을 지정하세요.",
            file=sys.stderr,
        )
        return 2

    try:
        hardware = load_yaml(REPO_ROOT / args.hardware)["audio"]
        fs = int(hardware["sample_rate"])
        block_size = int(args.block_size or hardware["block_size"])
        if not 0.0 < args.amplitude <= cw.MAX_AMPLITUDE:
            raise ValueError(f"--amplitude 는 0 초과 {cw.MAX_AMPLITUDE} 이하여야 합니다")
        if args.repeats < cw.MIN_REPEATS:
            raise ValueError(f"--repeats 는 {cw.MIN_REPEATS} 이상이어야 합니다")
        primary_out = cw._repo_path(args.primary_out)
        secondary_out = cw._repo_path(args.secondary_out)
        for path in (primary_out, secondary_out):
            if path.exists():
                raise FileExistsError(f"기존 정식 모델은 덮어쓰지 않습니다: {path}")
        if primary_out == secondary_out:
            raise ValueError("P 와 S 는 다른 파일이어야 합니다")
        diagnostics_root = cw._repo_path(args.diagnostics_root, require_results=True)
        max_delay = int(round(args.max_delay_ms / 1000.0 * fs))
        max_jitter = int(round(args.max_delay_jitter_ms / 1000.0 * fs))
        probe = build_interleaved_probe(
            sample_rate=fs,
            period_seconds=args.period_seconds,
            band_hz=(float(args.band[0]), float(args.band[1])),
            amplitude=float(args.amplitude),
            tone_spacing_hz=(
                float(args.tone_spacing_hz) if args.tone_spacing_hz else None
            ),
        )
        if probe.guard_bins() != 1:
            raise ValueError(
                f"guard={probe.guard_bins()} bin — 게이트는 1 을 요구합니다. "
                "--tone-spacing-hz 를 지우거나 --period-seconds 를 조정하세요"
            )
    except (KeyError, OSError, ValueError, FileExistsError) as exc:
        print(f"[중단] {exc}", file=sys.stderr)
        return 2

    need_lo, need_hi = float(args.required_band[0]), float(args.required_band[1])
    resolution = fs / probe.period_samples
    channel_band = {}
    for drive in ("noise", "cancel"):
        bins = probe.bins_for(drive)
        low, high = float(bins[0]) * resolution, float(bins[-1]) * resolution
        channel_band[drive] = (low, high)
        if low > need_lo or high < need_hi:
            print(
                f"[중단] {drive} 톤 대역 {low:.1f}-{high:.1f}Hz 가 필수 대역 "
                f"{need_lo:.0f}-{need_hi:.0f}Hz 를 덮지 못합니다. --band 를 넓히세요.",
                file=sys.stderr,
            )
            return 2

    crest_noise, crest_cancel = probe.crest_db()
    if max(crest_noise, crest_cancel) > MAX_CREST_DB:
        print(
            f"[중단] 크레스트 {crest_noise:.1f}/{crest_cancel:.1f} dB 가 "
            f"{MAX_CREST_DB} dB 를 넘습니다 — 같은 피크에서 음향 에너지를 잃습니다.",
            file=sys.stderr,
        )
        return 2

    capture_id = uuid.uuid4().hex
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = diagnostics_root / f"{stamp}_{capture_id[:8]}"
    session_dir.mkdir(parents=True, exist_ok=False)

    print(
        f"동시 인터리브 측정 {args.band[0]:.0f}-{args.band[1]:.0f}Hz · "
        f"톤 간격 {probe.bin_step('noise') * resolution:.2f}Hz · "
        f"주기 {args.period_seconds:.2f}s\n"
        f"  톤 수 noise {probe.noise_bins.size} / cancel {probe.cancel_bins.size} · "
        f"guard {probe.guard_bins()} bin · crest {crest_noise:.1f}/{crest_cancel:.1f} dB\n"
        f"  peak {args.amplitude:.4f} · block {block_size} · latency {args.latency} · "
        f"warmup {args.warmup_periods} + 분석 {args.repeats} 주기 "
        f"({(args.warmup_periods + args.repeats) * args.period_seconds:.0f}초 재생)"
    )

    try:
        import sounddevice as sd

        print("출력 없는 ERR/REF raw preflight 중...")
        preflight_raw, preflight_report = cw._capture_preflight(
            sd, hardware, args.input_probe_seconds
        )
        for name, item in zip(("ERR", "REF"), cw._probe_summary(preflight_report)):
            verdict = "PASS" if item["valid"] else "FAIL"
            print(
                f"[{verdict}] {name}: RMS {item['rms_dbfs']:.2f}dBFS, "
                f"peak {item['peak']:.6f}, clip {item['clip_ratio']:.3%}"
            )
        channels = preflight_report.get("channels", [])
        if len(channels) < 2 or not all(bool(c.get("valid")) for c in channels[:2]):
            print("[실패] 양 마이크 preflight 실패 — 출력 장치를 열지 않았습니다", file=sys.stderr)
            return 1

        in_dev = int(preflight_report["device"])
        output_cfg = hardware["output"]
        out_dev = resolve_alsa_portaudio_device(
            output_cfg["card"], output_cfg["pcm"], "output", 2
        )

        lead_in = fs // 2
        total_periods = int(args.warmup_periods) + int(args.repeats)
        playback = np.zeros((lead_in + total_periods * probe.period_samples, 2), np.float32)
        playback[lead_in:, 0] = np.tile(probe.noise_signal, total_periods)
        playback[lead_in:, 1] = np.tile(probe.cancel_signal, total_periods)

        recorded_raw, output_pcm, telemetry = cw._capture_measurement(
            sd,
            fs=fs,
            block_size=block_size,
            latency=str(args.latency),
            in_dev=in_dev,
            out_dev=out_dev,
            output_float=playback,
        )
    except (ImportError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"[실패] 측정 중단: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    recorded = pcm_int32_to_float32(recorded_raw)
    err = recorded[:, 0].astype(np.float64)
    measurement_report = cw.analyze_int32_input_probe(recorded_raw)

    invalid: list[str] = []
    if int(telemetry.get("xrun_count", 0)) != 0:
        invalid.append(f"xrun_{telemetry['xrun_count']}")
    if not telemetry.get("completed"):
        invalid.append("capture_incomplete")
    for index, item in enumerate(measurement_report.get("channels", [])[:2]):
        if float(item.get("clip_ratio", 1.0)) > cw.MAX_INPUT_CLIP_RATIO:
            invalid.append(f"input_clip_ch{index}_{item['clip_ratio']:.4f}")

    warp_report: dict[str, Any] = {"applied": False}
    if args.dewarp:
        # 재생 두 채널의 합이 곧 스피커가 함께 만든 음향 자극의 시간 구조다.
        # warp 는 그 합에 공통으로 걸리므로 합을 기준으로 추적하는 것이 옳다.
        mono = playback[:, 0].astype(np.float64) + playback[:, 1].astype(np.float64)
        centres, delays, peaks = track_warp(mono, err, window=int(args.track_window))
        err = dewarp_recording(err, centres, delays, peaks, min_peak=float(args.track_min_peak))
        warp_report = {
            "applied": True,
            "window": int(args.track_window),
            "points": int(delays.size),
            "kept_fraction": float(np.mean(peaks >= float(args.track_min_peak))),
            "delay_min": float(np.min(delays)),
            "delay_max": float(np.max(delays)),
            "delay_range": float(np.max(delays) - np.min(delays)),
            "peak_median": float(np.median(peaks)),
        }
        print(
            f"\nwarp 추적: 지연 {warp_report['delay_min']:.0f}~{warp_report['delay_max']:.0f} "
            f"(범위 {warp_report['delay_range']:.0f} 샘플) · 상관 중앙 "
            f"{warp_report['peak_median']:.3f} · 채택 {warp_report['kept_fraction']:.1%}"
        )

    period_starts = [
        lead_in + (int(args.warmup_periods) + k) * probe.period_samples
        for k in range(int(args.repeats))
    ]

    # 배경잡음 스펙트럼은 preflight 를 **같은 길이·같은 FFT** 로 변환해야 분모가 맞는다.
    preflight_err = pcm_int32_to_float32(preflight_raw)[:, 0].astype(np.float64)
    if preflight_err.size < probe.period_samples:
        preflight_err = np.pad(preflight_err, (0, probe.period_samples - preflight_err.size))
    noise_spectrum = np.fft.rfft(preflight_err[-probe.period_samples :])
    signal_spectrum = np.fft.rfft(err[period_starts[0] : period_starts[0] + probe.period_samples])

    results: dict[str, dict[str, Any]] = {}
    for drive, output_channel, label in (
        ("noise", "noise", "P(z) 소음→ERR"),
        ("cancel", "cancel", "S(z) 상쇄→ERR"),
    ):
        model, stack, consistency, correlations, error = analyse_channel(
            err=err,
            probe=probe,
            drive=drive,
            period_starts=period_starts,
            fir_length=int(args.fir_length),
            pre_roll=int(args.pre_roll),
            max_delay_samples=max_delay,
            max_delay_jitter_samples=max_jitter,
        )
        snr = tone_snr_db(signal_spectrum, noise_spectrum, probe.bins_for(drive))
        reasons = channel_quality(
            model=model,
            consistency=consistency,
            snr_db=snr,
            min_consistency=cw.MIN_CONSISTENCY,
        )
        if error:
            reasons.append(error)
        results[drive] = {
            "model": model,
            "irs": stack,
            "consistency": consistency,
            "correlations": correlations,
            "snr_db": snr,
            "reasons": reasons,
            "output_channel": output_channel,
        }
        delay = model.get("delay_samples")
        print(
            f"\n=== {label} ===\n"
            f"  지연 {delay if delay is not None else '미검출'} 샘플 · "
            f"반복 spread {model.get('delay_spread_samples')} (허용 {max_jitter}) · "
            f"반복 일관성 {consistency:.4f}\n"
            f"  톤 SNR 중앙 {np.median(snr):.1f} dB · 최소 {np.min(snr):.1f} dB · "
            f"{float(np.mean(snr >= MIN_TONE_SNR_DB)):.1%} 가 {MIN_TONE_SNR_DB:.0f}dB 이상"
        )
        if reasons:
            print(f"  [미달] {', '.join(reasons)}")

    valid = not invalid and not results["noise"]["reasons"] and not results["cancel"]["reasons"]

    metadata = {
        "capture_id": capture_id,
        "method": METHOD,
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sample_rate": fs,
        "block_size": block_size,
        "latency": args.latency,
        "amplitude": float(args.amplitude),
        "design_band_hz": [float(args.band[0]), float(args.band[1])],
        "required_band_hz": [need_lo, need_hi],
        "channel_band_hz": {k: list(v) for k, v in channel_band.items()},
        "tone_spacing_hz": float(probe.bin_step("noise") * resolution),
        "period_seconds": float(args.period_seconds),
        "warmup_periods": int(args.warmup_periods),
        "repeats": int(args.repeats),
        "guard_bins": probe.guard_bins(),
        "crest_db": {"noise": crest_noise, "cancel": crest_cancel},
        "warp": warp_report,
        "telemetry": telemetry,
        "preflight": preflight_report,
        "measurement": measurement_report,
        "invalid_reasons": invalid,
        "valid": valid,
        "channels": {
            drive: {
                "output_channel": item["output_channel"],
                "consistency": item["consistency"],
                "pairwise_correlations": item["correlations"],
                "delay_samples": item["model"].get("delay_samples"),
                "repeat_delay_samples": item["model"].get("repeat_delay_samples"),
                "delay_spread_samples": item["model"].get("delay_spread_samples"),
                "tone_snr_median_db": float(np.median(item["snr_db"])),
                "tone_snr_min_db": float(np.min(item["snr_db"])),
                "reasons": item["reasons"],
            }
            for drive, item in results.items()
        },
    }

    npz_path = session_dir / "raw_measurement.npz"
    with npz_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            output=playback.astype(np.float32),
            err=recorded[:, 0].astype(np.float32),
            ref=recorded[:, 1].astype(np.float32),
            input_raw_int32=recorded_raw.astype(np.int32),
            preflight_raw_int32=preflight_raw.astype(np.int32),
            noise_irs=results["noise"]["irs"].astype(np.float64),
            cancel_irs=results["cancel"]["irs"].astype(np.float64),
            noise_snr_db=results["noise"]["snr_db"].astype(np.float64),
            cancel_snr_db=results["cancel"]["snr_db"].astype(np.float64),
            metadata_json=np.asarray(
                json.dumps(cw._json_safe(metadata), ensure_ascii=False, sort_keys=True)
            ),
        )
    with (session_dir / "metadata.json").open("x", encoding="utf-8") as handle:
        json.dump(cw._json_safe(metadata), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    if not valid:
        print(f"\n[실패] 정식 모델을 저장하지 않았습니다. 진단: {session_dir}", file=sys.stderr)
        if invalid:
            print(f"  캡처 결함: {', '.join(invalid)}", file=sys.stderr)
        return 1

    for drive, out_path in (("noise", primary_out), ("cancel", secondary_out)):
        item = results[drive]
        cw.save_official_model(
            out_path,
            valid=True,
            arrays=_official_arrays(
                model=item["model"],
                fs=fs,
                consistency=item["consistency"],
                band_hz=channel_band[drive],
                amplitude=float(args.amplitude),
                block_size=block_size,
                latency=str(args.latency),
                output_channel=item["output_channel"],
                repeats=int(args.repeats),
                xrun_count=int(telemetry.get("xrun_count", 0)),
                capture_id=capture_id,
                probe=probe,
                drive=drive,
                snr_db=item["snr_db"],
                period_seconds=float(args.period_seconds),
            ),
        )

    p_delay = int(results["noise"]["model"]["delay_samples"])
    s_delay = int(results["cancel"]["model"]["delay_samples"])
    handoff = 256
    lead = max(0, s_delay + handoff - p_delay)
    print(
        f"\n[성공] P {primary_out.relative_to(REPO_ROOT)}\n"
        f"       S {secondary_out.relative_to(REPO_ROOT)}\n"
        f"       진단 {session_dir.relative_to(REPO_ROOT)}\n\n"
        f"duct.yaml 에 기입할 값:\n"
        f"  digital_reference.primary_path_npz: {primary_out.relative_to(REPO_ROOT)}\n"
        f"  digital_reference.d_noise_delay_samples: {p_delay}\n"
        f"  secondary_path.npz: {secondary_out.relative_to(REPO_ROOT)}\n"
        f"data_sim.yaml digital_reference_lead_samples: "
        f"{lead}  (= S {s_delay} + handoff {handoff} − P {p_delay})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
