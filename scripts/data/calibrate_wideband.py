#!/usr/bin/env python3
"""광대역(기본 80–8000Hz) 경로 보정 — ESS(지수 사인 스윕) 방식.

용도 [설계 교차검증 C3/C2]:
  --output-channel cancel : S(z) 광대역 재보정 → 풀밴드 학습(커리큘럼 B)의 선행 게이트
  --output-channel noise  : digital-ref 1차경로 지연 D_noise 실측
                            → configs/duct.yaml digital_reference.d_noise_delay_samples 에 기입

  python scripts/data/calibrate_wideband.py --output-channel cancel --out assets/measured/secondary_path_wb.npz
  python scripts/data/calibrate_wideband.py --output-channel noise

기존 anc_project/calibrate_s_path.py 의 대역제한(150–600Hz) 보정을 대체하는 확장판.
저장 npz 는 deep_anc.dsp.secondary_path.load_secondary_path 와 호환된다.
⚠ 스피커에서 스윕이 재생된다 — TPA3116D2 볼륨을 낮춘 상태로 시작할 것.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.audio_io import (                              # noqa: E402
    pcm_int32_to_float32,
    resolve_alsa_portaudio_device,
    rms_dbfs,
)
from deep_anc.config import REPO_ROOT, load_yaml             # noqa: E402


def ess_pair(f1: float, f2: float, seconds: float, fs: int, amp: float):
    """지수 사인 스윕과 진폭 보상 역필터 (Farina 방식)."""
    n = int(seconds * fs)
    t = np.arange(n) / fs
    R = np.log(f2 / f1)
    sweep = np.sin(2 * np.pi * f1 * seconds / R * (np.exp(t * R / seconds) - 1.0))
    fade = int(0.05 * fs)
    env = np.ones(n)
    env[:fade] = np.sin(np.linspace(0, np.pi / 2, fade)) ** 2
    env[-fade:] = env[:fade][::-1]
    sweep = (amp * sweep * env).astype(np.float32)
    # 역필터: 시간 반전 + 지수 진폭 보상
    inv = sweep[::-1].astype(np.float64) * np.exp(-t * R / seconds)
    return sweep, inv.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", default="configs/hardware_jetson.yaml")
    parser.add_argument("--output-channel", choices=["cancel", "noise"], default="cancel")
    parser.add_argument("--band", type=float, nargs=2, default=[80.0, 8000.0])
    parser.add_argument("--sweep-seconds", type=float, default=4.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--amplitude", type=float, default=0.06)
    parser.add_argument("--fir-length", type=int, default=2048)
    parser.add_argument("--pre-roll", type=int, default=32)
    parser.add_argument("--max-delay-ms", type=float, default=250.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import sounddevice as sd

    hw = load_yaml(REPO_ROOT / args.hardware)["audio"]
    fs = int(hw["sample_rate"])
    block = int(hw["block_size"])
    out_ch = 1 if args.output_channel == "cancel" else 0

    in_dev = resolve_alsa_portaudio_device(hw["input"]["card"], hw["input"]["pcm"], "input", 2)
    out_dev = resolve_alsa_portaudio_device(hw["output"]["card"], hw["output"]["pcm"], "output", 2)

    sweep, inv = ess_pair(args.band[0], args.band[1], args.sweep_seconds, fs, args.amplitude)
    gap = np.zeros(int(1.0 * fs), dtype=np.float32)
    one_shot = np.concatenate([gap, sweep, gap])
    playback = np.tile(one_shot, args.repeats)
    total = playback.size

    out_buf = np.zeros((total, 2), dtype=np.float32)
    out_buf[:, out_ch] = playback
    recorded = np.zeros((total, 2), dtype=np.float32)
    cursor = {"i": 0}

    def callback(indata, outdata, frames, _t, status):
        i = cursor["i"]
        n = min(frames, total - i)
        recorded[i : i + n] = pcm_int32_to_float32(indata[:n, :2])
        chunk = np.zeros((frames, 2), dtype=np.float32)
        chunk[:n] = out_buf[i : i + n]
        outdata[:] = np.rint(np.clip(chunk, -1, 1) * 32767).astype(np.int16)
        cursor["i"] = i + n
        if cursor["i"] >= total:
            raise sd.CallbackStop

    print(
        f"ESS 보정: {args.band[0]:.0f}–{args.band[1]:.0f}Hz ×{args.repeats}회, "
        f"출력 ch{out_ch}({args.output_channel}), 진폭 {args.amplitude}"
    )
    with sd.Stream(
        samplerate=fs, blocksize=block, device=(in_dev, out_dev),
        channels=(2, 2), dtype=("int32", "int16"), latency=("low", "low"),
        callback=callback, prime_output_buffers_using_stream_callback=True,
    ):
        while cursor["i"] < total:
            time.sleep(0.1)

    err = recorded[:, 0].astype(np.float64)
    print(f"녹음 완료: err RMS {rms_dbfs(err):.1f} dBFS")

    # ----- 반복별 IR 추출 → 평균 -----
    # 진폭 정규화: sweep*inv 자기 디컨볼루션 피크로 나눠야 IR 이 물리 단위가 된다
    # (미정규화 시 스윕 파라미터에 따라 수십 배 과대 — 리뷰 확정 결함 #12)
    ref_peak = float(np.max(np.abs(signal.fftconvolve(sweep.astype(np.float64), inv, mode="full"))))
    seg_len = one_shot.size
    max_delay = int(args.max_delay_ms / 1000.0 * fs)
    irs = []
    for r in range(args.repeats):
        seg = err[r * seg_len : (r + 1) * seg_len]
        ir_full = signal.fftconvolve(seg, inv, mode="full") / max(ref_peak, 1e-12)
        # 선형 IR 시작점 = 스윕 종료 위치 (gap 1s + sweep)
        start = gap.size + sweep.size - 1
        ir = ir_full[start : start + max_delay + args.fir_length + 4096]
        irs.append(ir)
    n_min = min(len(v) for v in irs)
    irs = np.stack([v[:n_min] for v in irs])
    ir_mean = irs.mean(axis=0)

    # 반복 일관성 (coherence 대용 지표)
    consistency = float(
        np.mean([np.corrcoef(irs[i], ir_mean)[0, 1] for i in range(args.repeats)])
    )

    # ----- 지연/FIR 분리 (anc_project 규약: first-arrival − pre_roll) -----
    peak = float(np.max(np.abs(ir_mean)))
    if peak <= 0 or not np.isfinite(peak):
        print("[실패] IR 피크 없음 — 볼륨/배선 확인", file=sys.stderr)
        return 1
    first = int(np.flatnonzero(np.abs(ir_mean) >= peak * 0.05)[0])
    delay_samples = max(0, first - args.pre_roll)
    fir = ir_mean[delay_samples : delay_samples + args.fir_length].astype(np.float32)

    print(
        f"결과: delay {delay_samples}샘플 ({1000*delay_samples/fs:.2f}ms), "
        f"FIR {fir.size}탭, 반복 일관성 {consistency:.3f}"
    )

    if args.output_channel == "noise":
        print(
            "\n→ configs/duct.yaml 의 digital_reference.d_noise_delay_samples 에 "
            f"{delay_samples} 를 기입하세요 (D_noise 실측값)."
        )
        out = args.out or "assets/measured/d_noise_measurement.npz"
    else:
        out = args.out or "assets/measured/secondary_path_wb.npz"
        if consistency < 0.9:
            print(
                "[경고] 반복 일관성이 낮습니다 (<0.9) — 배경소음/볼륨을 점검하고 재측정 권장.\n"
                "       이 상태로 풀밴드 커리큘럼(B) 학습은 비권장 [C3 게이트]."
            )

    out_path = REPO_ROOT / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        fir=fir,
        delay_samples=delay_samples,
        sample_rate=fs,
        dc_block_r=0.995,
        fit_improvement_db=float("nan"),
        coherence_median=consistency,
        excitation_band_hz=np.array(args.band, dtype=np.float64),
        calibration_block_size=block,
        calibration_latency="low",
        output_channel=args.output_channel,
        method="ess",
    )
    print(f"저장: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
