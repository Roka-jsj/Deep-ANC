#!/usr/bin/env python3
"""덕트 실측 수집 — 소음 재생(ch0) + 레퍼런스/에러 마이크 동시 녹음 (ANC OFF).

  python scripts/data/record_duct.py --program tone --frequency 300 --seconds 60
  python scripts/data/record_duct.py --program band --seconds 120
  python scripts/data/record_duct.py --program silence --seconds 30   # 암소음 측정

저장: data/recorded/<타임스탬프_프로그램>/{mics.wav(2ch PCM_32), source.wav, session.json}
시작 시 레퍼런스 마이크(ch1) 자가진단 — 과거 무신호 이력 대응 (docs/02).
상쇄 스피커(ch1 출력)는 전 구간 무음을 유지한다.
"""

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.audio_io import (                              # noqa: E402
    pcm_int32_to_float32,
    resolve_alsa_portaudio_device,
    rms_dbfs,
)
from deep_anc.config import REPO_ROOT, load_yaml             # noqa: E402
from deep_anc.realtime.noise_gen import NoiseProgram         # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", default="configs/hardware_jetson.yaml")
    parser.add_argument(
        "--program",
        default="tone",
        choices=["tone", "multitone", "white", "band", "nonlinear", "sweep", "file", "silence"],
    )
    parser.add_argument("--frequency", type=float, default=300.0)
    parser.add_argument("--amplitude", type=float, default=0.05)
    parser.add_argument("--band", type=float, nargs=2, default=[80.0, 1000.0])
    parser.add_argument("--file", default=None, help="program=file 재생 wav")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--out-root", default="data/recorded")
    parser.add_argument("--force", action="store_true", help="ref 마이크 무신호여도 진행")
    parser.add_argument("--ref-check-dbfs", type=float, default=-80.0)
    args = parser.parse_args()

    import sounddevice as sd

    hw = load_yaml(REPO_ROOT / args.hardware)["audio"]
    fs = int(hw["sample_rate"])
    block = int(hw["block_size"])

    in_dev = resolve_alsa_portaudio_device(hw["input"]["card"], hw["input"]["pcm"], "input", 2)
    out_dev = resolve_alsa_portaudio_device(hw["output"]["card"], hw["output"]["pcm"], "output", 2)

    # ----- 1) 레퍼런스 마이크 자가진단 (2초 무음 캡처) -----
    print("레퍼런스 마이크 점검 중 (2초)...")
    probe = sd.rec(int(2 * fs), samplerate=fs, channels=2, dtype="int32", device=in_dev)
    sd.wait()
    probe_f = pcm_int32_to_float32(probe)
    err_db = rms_dbfs(probe_f[:, 0])
    ref_db = rms_dbfs(probe_f[:, 1])
    print(f"  ch0(err) {err_db:7.2f} dBFS | ch1(ref) {ref_db:7.2f} dBFS")
    if ref_db < args.ref_check_dbfs and not args.force:
        print(
            f"[중단] 레퍼런스 마이크(ch1)가 무신호로 보입니다 ({ref_db:.1f} dBFS < "
            f"{args.ref_check_dbfs}). 배선 점검(docs/02_hardware_setup.md) 후 재시도하거나 "
            "--force 로 강행하세요.", file=sys.stderr,
        )
        return 1

    # ----- 2) 프로그램 준비 -----
    prog_cfg = {
        "type": args.program,
        "frequency": args.frequency,
        "amplitude": args.amplitude,
        "band": args.band,
        "file": args.file,
    }
    program = NoiseProgram(prog_cfg, fs)

    total = int(args.seconds * fs)
    source = np.zeros(total, dtype=np.float32)
    recorded = np.zeros((total, 2), dtype=np.float32)
    cursor = {"in": 0, "out": 0}

    fade = np.linspace(0.0, 1.0, int(0.1 * fs), dtype=np.float32)

    def callback(indata, outdata, frames, _time, status):
        if status:
            print(f"[xrun] {status}", file=sys.stderr)
        i = cursor["in"]
        n = min(frames, total - i)
        recorded[i : i + n] = pcm_int32_to_float32(indata[:n, :2])
        cursor["in"] = i + n

        o = cursor["out"]
        blk = program.generate(frames)
        # 시작/종료 페이드
        for k in range(frames):
            pos = o + k
            if pos < fade.size:
                blk[k] *= fade[pos]
            elif pos >= total - fade.size:
                blk[k] *= fade[max(0, total - 1 - pos)] if pos < total else 0.0
        m = min(frames, total - o)
        source[o : o + m] = blk[:m]
        out = np.zeros((frames, 2), dtype=np.float32)
        out[:, 0] = blk                             # ch0 = 소음 스피커
        # ch1(상쇄 스피커)은 무음 유지
        outdata[:] = np.rint(np.clip(out, -1, 1) * 32767).astype(np.int16)
        cursor["out"] = o + m
        if cursor["in"] >= total:
            raise sd.CallbackStop

    print(f"녹음 시작: {args.program}, {args.seconds:.0f}초 (ANC 없음, ch1 무음)")
    with sd.Stream(
        samplerate=fs,
        blocksize=block,
        device=(in_dev, out_dev),
        channels=(2, 2),
        dtype=("int32", "int16"),
        latency=("low", "low"),
        callback=callback,
        prime_output_buffers_using_stream_callback=True,
    ):
        while cursor["in"] < total:
            time.sleep(0.1)

    # ----- 3) 저장 -----
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = REPO_ROOT / args.out_root / f"{stamp}_{args.program}"
    session_dir.mkdir(parents=True, exist_ok=True)
    sf.write(session_dir / "mics.wav", recorded, fs, subtype="PCM_32")
    sf.write(session_dir / "source.wav", source, fs, subtype="FLOAT")
    meta = {
        "program": prog_cfg,
        "seconds": args.seconds,
        "sample_rate": fs,
        "block_size": block,
        "channels": {"err_mic": 0, "ref_mic": 1, "noise_out": 0, "cancel_out": 1},
        "ref_check_dbfs": {"err": err_db, "ref": ref_db},
        "timestamp": stamp,
    }
    (session_dir / "session.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"저장 완료: {session_dir}")
    print("다음: python scripts/data/make_recorded_manifest.py 로 manifest 갱신")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
