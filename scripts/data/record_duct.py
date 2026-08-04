#!/usr/bin/env python3
"""덕트 실측 수집 — 소음 재생(ch0) + 레퍼런스/에러 마이크 동시 녹음 (ANC OFF).

  .venv/bin/python scripts/data/record_duct.py --program tone --frequency 300 --seconds 60
  .venv/bin/python scripts/data/record_duct.py --program band --seconds 120
  .venv/bin/python scripts/data/record_duct.py --program silence --seconds 30   # 암소음 측정

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
from deep_anc.data.manifest import (                         # noqa: E402
    validate_group_id,
    validate_session_id,
    validate_source_family,
)
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
    parser.add_argument(
        "--source-family",
        default=None,
        help=(
            "소스 계열 ID(예: speech/music/environment). 생략 시 program 이름을 사용하며, "
            "program=file은 명시를 권장"
        ),
    )
    parser.add_argument(
        "--group-id",
        default=None,
        help=(
            "분할 누수를 막을 상관 그룹 ID(같은 화자/곡/원본/환경의 반복 세션은 같은 값). "
            "생략 시 현재 세션만의 ID 사용"
        ),
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=1.0,
        help=(
            "스트림을 연 뒤 무음으로 흘려보내고 버릴 길이. I2S 기동 트랜지언트가 "
            "약 0.5초 지속되므로 여유를 둔 1.0초가 기본값"
        ),
    )
    parser.add_argument("--force", action="store_true", help="ref 마이크 무신호여도 진행")
    parser.add_argument("--ref-check-dbfs", type=float, default=-80.0)
    args = parser.parse_args()

    try:
        source_family = validate_source_family(args.source_family or args.program)
        requested_group_id = (
            validate_group_id(args.group_id) if args.group_id is not None else None
        )
    except ValueError as exc:
        parser.error(str(exc))

    import sounddevice as sd

    hw = load_yaml(REPO_ROOT / args.hardware)["audio"]
    fs = int(hw["sample_rate"])
    block = int(hw["block_size"])

    in_dev = resolve_alsa_portaudio_device(hw["input"]["card"], hw["input"]["pcm"], "input", 2)
    out_dev = resolve_alsa_portaudio_device(hw["output"]["card"], hw["output"]["pcm"], "output", 2)

    # ----- 1) 레퍼런스 마이크 자가진단 (2초 무음 캡처) -----
    print("레퍼런스 마이크 점검 중 (2초)...")
    # 앞 1초는 기동 트랜지언트라 버린다. 이걸 포함해서 재면 무신호 마이크도 -42dBFS 로
    # 보여 "살아 있다"고 오판한다 — 이 점검의 목적을 정확히 무력화한다.
    probe_settle = int(1.0 * fs)
    probe = sd.rec(
        probe_settle + int(2 * fs), samplerate=fs, channels=2, dtype="int32", device=in_dev
    )
    sd.wait()
    probe_f = pcm_int32_to_float32(probe[probe_settle:])
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

    # I2S 입력은 스트림을 연 직후 약 0.5초 동안 큰 기동 트랜지언트를 낸다
    # (실측: 0.0-0.5초 -36.3 dBFS peak 0.062 → 0.5초 이후 -67.4 dBFS peak 0.002).
    # 이 구간을 세션에 남기면 (a) 학습 데이터 앞머리가 잡음이 되고 (b) 세션 QA 의
    # peak/RMS 통계가 트랜지언트를 재게 된다. 무음으로 흘려보내고 잘라낸다.
    # 출력과 입력을 같은 길이만큼 버리므로 정렬은 유지된다.
    settle = int(max(0.0, args.settle_seconds) * fs)
    keep = int(args.seconds * fs)
    total = keep + settle
    source = np.zeros(total, dtype=np.float32)
    recorded = np.zeros((total, 2), dtype=np.float32)
    cursor = {"in": 0, "out": 0}
    xrun_state: dict = {"count": 0, "flags": set()}

    fade = np.linspace(0.0, 1.0, int(0.1 * fs), dtype=np.float32)

    def callback(indata, outdata, frames, _time, status):
        if status:
            # 콜백 안에서 print 하면 그 자체가 다음 xrun 을 만든다. 세어만 두고 밖에서 판정한다.
            # xrun 은 source 와 mics 사이에 **영구 오프셋**을 남긴다 — 커서는 frames 만큼
            # 계속 전진하므로 드롭된 블록만큼 두 배열이 세션 끝까지 어긋난다. lead 예산이
            # 통째로 109 샘플인데 블록 1회 = 256 샘플이므로 학습 데이터로 쓸 수 없다.
            xrun_state["count"] += 1
            xrun_state["flags"].add(str(status))
        i = cursor["in"]
        n = min(frames, total - i)
        recorded[i : i + n] = pcm_int32_to_float32(indata[:n, :2])
        cursor["in"] = i + n

        o = cursor["out"]
        blk = program.generate(frames)
        # settle 구간은 무음으로 흘린다 — 프로그램 위상은 그대로 진행시켜 잘라낸 뒤에도
        # source/mics 가 같은 시각을 가리키게 한다.
        # 페이드는 settle 이 끝나는 시점을 0으로 삼는다.
        for k in range(frames):
            pos = o + k
            if pos < settle:
                blk[k] = 0.0
                continue
            local = pos - settle
            if local < fade.size:
                blk[k] *= fade[local]
            elif local >= keep - fade.size:
                blk[k] *= fade[max(0, keep - 1 - local)] if local < keep else 0.0
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
    # xrun 이 하나라도 있으면 source↔mics 정렬이 깨졌다. 전달맵은 이미 xrun 을 무효화
    # 사유로 쓰는데(measure_duct_transfer_map) 학습데이터 수집기만 기준이 느슨했다.
    if xrun_state["count"] > 0:
        print(
            f"[중단] 오디오 xrun {xrun_state['count']}회 ({', '.join(sorted(xrun_state['flags']))}) — "
            "source 와 mics 의 정렬이 깨져 학습에 쓸 수 없습니다. 세션을 저장하지 않습니다.",
            file=sys.stderr,
        )
        return 1

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = REPO_ROOT / args.out_root / f"{stamp}_{args.program}"
    session_dir.mkdir(parents=True, exist_ok=True)
    # settle 구간을 양쪽에서 동일하게 잘라낸다 (정렬 유지).
    sf.write(session_dir / "mics.wav", recorded[settle:], fs, subtype="PCM_32")
    sf.write(session_dir / "source.wav", source[settle:], fs, subtype="FLOAT")
    session_id = validate_session_id(session_dir.name)
    group_id = requested_group_id or validate_group_id(session_id)
    meta = {
        "session_id": session_id,
        "program": prog_cfg,
        "source_family": source_family,
        "group_id": group_id,
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
    print("다음: .venv/bin/python scripts/data/make_recorded_manifest.py 로 manifest 갱신")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
