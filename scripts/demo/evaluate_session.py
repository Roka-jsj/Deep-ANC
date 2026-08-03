#!/usr/bin/env python3
"""자동 실기 평가 — 시나리오별 OFF(베이스라인)→ON→OFF 프로토콜, md 리포트 생성.

  .venv/bin/python scripts/demo/evaluate_session.py --controllers fxlms dl --scenarios tone300 band
⚠ 스피커에서 소음이 재생된다. TPA3116D2 볼륨을 낮춘 상태에서, 사용자 입회 하에 실행.
"""

import argparse
import datetime
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.audio_io import capture_input_probe                         # noqa: E402
from deep_anc.config import REPO_ROOT, load_runtime_config, load_yaml     # noqa: E402
from deep_anc.dsp.secondary_path import load_secondary_path                # noqa: E402
from deep_anc.eval.metrics import (                                        # noqa: E402
    band_nmse_db,
    intersect_frequency_bands,
    nmse_db,
    octave_band_attenuation,
)
from deep_anc.realtime.run_realtime import RealtimeANC                  # noqa: E402


def run_scenario(cfg: dict, protocol: dict) -> dict:
    base_s = float(protocol.get("baseline_seconds", 10))
    on_s = float(protocol.get("on_seconds", 30))
    tail_s = float(protocol.get("tail_seconds", 5))
    total = base_s + on_s + tail_s

    anc = RealtimeANC(cfg, record_seconds=total + 2.0)
    try:
        anc.start()
        time.sleep(base_s)
        anc.state.anc_enabled = True
        time.sleep(on_s)
        anc.state.anc_enabled = False
        time.sleep(tail_s)
        stats = dict(anc.state.latest_stats)
    finally:
        anc.stop()

    if anc.state.fatal_error is not None:
        raise RuntimeError("실기 평가 오디오 콜백이 실패했습니다") from anc.state.fatal_error

    data = anc.session_data()
    fs = anc.fs
    err = data["err"]
    # 게이트 램프를 피해서 구간 절단 (경계 1초 여유)
    off_seg = err[int(1.0 * fs) : int((base_s - 1.0) * fs)]
    on_seg = err[int((base_s + 2.0) * fs) : int((base_s + on_s - 1.0) * fs)]
    on_gain = data["anc_gain"][
        int((base_s + 2.0) * fs) : int((base_s + on_s - 1.0) * fs)
    ]
    on_duty = float(np.mean(on_gain >= 0.999)) if on_gain.size else 0.0
    if on_duty < 0.95:
        raise RuntimeError(
            "ANC ON 유효 구간이 95% 미만입니다 "
            f"({100.0 * on_duty:.1f}%). 자동 mute/언더런 결과는 성능 측정으로 인정하지 않습니다."
        )
    return {
        "fs": fs,
        "off": off_seg,
        "on": on_seg,
        "on_duty": on_duty,
        "stats": stats,
        "data": data,
    }


def input_preflight(cfg: dict, seconds: float = 2.0) -> bool:
    """출력 장치를 열기 전에 ERR/REF I2S 입력의 생존 여부를 확인한다."""
    report = capture_input_probe(cfg["hardware"]["audio"], seconds=seconds)
    names = ("ERR", "REF")
    for item in report["channels"][:2]:
        index = int(item["channel"])
        verdict = "PASS" if item["valid"] else "FAIL"
        print(
            f"[{verdict}] {names[index]} ch{index}: RMS {item['rms_dbfs']:.2f}dBFS, "
            f"peak {item['peak']:.6f}, clip {item['clip_ratio']:.3%}, "
            f"unique {item['unique_codes']}, raw [{item['raw_min']}, {item['raw_max']}]"
        )

    required_channels = (0, 1) if cfg.get("reference") == "mic" else (0,)
    failed = [index for index in required_channels if not report["channels"][index]["valid"]]
    if failed:
        labels = ", ".join(f"{names[index]} ch{index}" for index in failed)
        print(
            f"[중단] 필수 입력({labels})이 무효입니다. 스피커 출력과 실기 평가를 시작하지 않습니다.",
            file=sys.stderr,
        )
        return False
    if not report["channels"][1]["valid"]:
        print(
            "[경고] REF ch1이 무효입니다. digital-reference 평가는 가능하지만 "
            "acoustic-reference 수집/평가는 금지합니다.",
            file=sys.stderr,
        )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/runtime.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--controllers", nargs="+", default=["fxlms", "dl"])
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--input-probe-seconds",
        type=float,
        default=2.0,
        help="스피커 출력 전에 수행할 무출력 마이크 사전점검 길이",
    )
    args = parser.parse_args()

    eval_cfg = load_yaml(REPO_ROOT / args.eval_config)
    scenarios = {s["name"]: s for s in eval_cfg["scenarios"]}
    chosen = []
    for name in args.scenarios or list(scenarios.keys()):
        if name in scenarios:
            chosen.append(name)
        else:
            print(f"[skip] 알 수 없는 시나리오 '{name}' — eval.yaml scenarios: {list(scenarios)}")
    protocol = eval_cfg.get("protocol", {})
    bands = eval_cfg.get("octave_bands_hz", [125, 250, 500, 1000, 2000, 4000, 8000])
    initial_cfg = load_runtime_config(args.config)
    try:
        if not input_preflight(initial_cfg, seconds=args.input_probe_seconds):
            return 2
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[중단] 입력 사전점검 실패: {exc}", file=sys.stderr)
        return 2
    fs_config = int(initial_cfg["hardware"]["audio"]["sample_rate"])
    sp = load_secondary_path(REPO_ROOT / initial_cfg["duct"]["secondary_path"]["npz"])
    if sp.sample_rate != fs_config:
        raise ValueError(
            f"S(z) sample_rate={sp.sample_rate}Hz != runtime sample_rate={fs_config}Hz"
        )
    trusted = intersect_frequency_bands(
        sp.excitation_band_hz,
        initial_cfg["duct"]["acoustics"]["realistic_target_band_hz"],
        fs_config / 2.0,
    )

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    lines = [
        f"# 실기 평가 리포트 ({stamp})",
        "",
        f"- Trusted 대역: **{trusted[0]:.0f}–{trusted[1]:.0f} Hz** "
        f"(S(z) 실측 대역 ∩ 덕트 목표 대역)",
        "",
    ]
    lines.append(
        "| 시나리오 | 컨트롤러 | Trusted NMSE(dB) | Fullband NMSE(dB) | 간극 T−F(dB) | "
        "전대역 감쇠(dB) | 밴드별(dB, *=신뢰낮음) | miss | xrun |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for name in chosen:
        for controller in args.controllers:
            cfg = load_runtime_config(args.config)
            cfg["controller"] = controller
            cfg["noise"] = dict(scenarios[name]["noise"])
            print(f"\n=== {name} × {controller} ===")
            result = run_scenario(cfg, protocol)
            if result["fs"] != sp.sample_rate:
                raise ValueError(
                    f"세션 sample_rate={result['fs']}Hz != S(z) sample_rate={sp.sample_rate}Hz"
                )
            nmse_fullband = nmse_db(result["off"], result["on"])
            nmse_trusted = band_nmse_db(
                result["off"], result["on"], result["fs"], trusted
            )
            nmse_gap = nmse_trusted - nmse_fullband
            att = -nmse_fullband
            band_att = octave_band_attenuation(
                result["off"], result["on"], result["fs"], bands, trusted
            )
            band_txt = " ".join(
                f"{b['center_hz']:.0f}:{b['attenuation_db']:+.1f}{'' if b['trusted'] else '*'}"
                for b in band_att
            )
            s = result["stats"]
            print(
                f"  Trusted NMSE {nmse_trusted:+.2f} dB | "
                f"Fullband NMSE {nmse_fullband:+.2f} dB | 간극 {nmse_gap:+.2f} dB | "
                f"전대역 감쇠 {att:+.2f} dB | "
                f"{band_txt}"
            )
            lines.append(
                f"| {name} | {controller} | {nmse_trusted:+.2f} | {nmse_fullband:+.2f} | "
                f"{nmse_gap:+.2f} | {att:+.2f} | {band_txt} | "
                f"{s.get('underruns', 0)} | {s.get('xruns', 0)} |"
            )
            # 원시 세션 저장
            report_root = REPO_ROOT / eval_cfg.get("report_dir", "results")
            sess_dir = report_root / f"session_{stamp}"
            sess_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                sess_dir / f"{name}_{controller}.npz",
                fs=result["fs"],
                trusted_band_hz=np.asarray(trusted, dtype=np.float64),
                nmse_trusted_db=nmse_trusted,
                nmse_fullband_db=nmse_fullband,
                nmse_gap_trusted_minus_fullband_db=nmse_gap,
                **result["data"],
            )

    out = (
        Path(args.out) if args.out
        else REPO_ROOT / eval_cfg.get("report_dir", "results") / f"eval_report_{stamp}.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n리포트 저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
