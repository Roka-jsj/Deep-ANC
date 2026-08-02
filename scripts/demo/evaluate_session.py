#!/usr/bin/env python3
"""자동 실기 평가 — 시나리오별 OFF(베이스라인)→ON→OFF 프로토콜, md 리포트 생성.

  python scripts/demo/evaluate_session.py --controllers fxlms dl --scenarios tone300 band
⚠ 스피커에서 소음이 재생된다. TPA3116D2 볼륨을 낮춘 상태에서, 사용자 입회 하에 실행.
"""

import argparse
import datetime
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.config import REPO_ROOT, load_runtime_config, load_yaml   # noqa: E402
from deep_anc.eval.metrics import attenuation_db, octave_band_attenuation  # noqa: E402
from deep_anc.realtime.run_realtime import RealtimeANC                  # noqa: E402


def run_scenario(cfg: dict, protocol: dict) -> dict:
    base_s = float(protocol.get("baseline_seconds", 10))
    on_s = float(protocol.get("on_seconds", 30))
    tail_s = float(protocol.get("tail_seconds", 5))
    total = base_s + on_s + tail_s

    anc = RealtimeANC(cfg, record_seconds=total + 2.0)
    anc.start()
    time.sleep(base_s)
    anc.state.anc_enabled = True
    time.sleep(on_s)
    anc.state.anc_enabled = False
    time.sleep(tail_s)
    stats = dict(anc.state.latest_stats)
    anc.stop()

    data = anc.session_data()
    fs = anc.fs
    err = data["err"]
    # 게이트 램프를 피해서 구간 절단 (경계 1초 여유)
    off_seg = err[int(1.0 * fs) : int((base_s - 1.0) * fs)]
    on_seg = err[int((base_s + 2.0) * fs) : int((base_s + on_s - 1.0) * fs)]
    return {"fs": fs, "off": off_seg, "on": on_seg, "stats": stats, "data": data}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/runtime.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--controllers", nargs="+", default=["fxlms", "dl"])
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    eval_cfg = load_yaml(REPO_ROOT / args.eval_config)
    scenarios = {s["name"]: s for s in eval_cfg["scenarios"]}
    chosen = args.scenarios or list(scenarios.keys())
    protocol = eval_cfg.get("protocol", {})
    bands = eval_cfg.get("octave_bands_hz", [125, 250, 500, 1000, 2000, 4000, 8000])
    trusted = tuple(eval_cfg.get("trusted_band_hz", [150, 600]))

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    lines = [f"# 실기 평가 리포트 ({stamp})", ""]
    lines.append("| 시나리오 | 컨트롤러 | 전대역 감쇠(dB) | 밴드별(dB, *=신뢰낮음) | miss | xrun |")
    lines.append("|---|---|---|---|---|---|")

    for name in chosen:
        for controller in args.controllers:
            cfg = load_runtime_config(args.config)
            cfg["controller"] = controller
            cfg["noise"] = dict(scenarios[name]["noise"])
            print(f"\n=== {name} × {controller} ===")
            result = run_scenario(cfg, protocol)
            att = attenuation_db(result["off"], result["on"])
            band_att = octave_band_attenuation(
                result["off"], result["on"], result["fs"], bands, trusted
            )
            band_txt = " ".join(
                f"{b['center_hz']:.0f}:{b['attenuation_db']:+.1f}{'' if b['trusted'] else '*'}"
                for b in band_att
            )
            s = result["stats"]
            print(f"  전대역 감쇠 {att:+.2f} dB | {band_txt}")
            lines.append(
                f"| {name} | {controller} | {att:+.2f} | {band_txt} | "
                f"{s.get('underruns', 0)} | {s.get('xruns', 0)} |"
            )
            # 원시 세션 저장
            sess_dir = REPO_ROOT / "results" / f"session_{stamp}"
            sess_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                sess_dir / f"{name}_{controller}.npz", fs=result["fs"], **result["data"]
            )

    out = Path(args.out) if args.out else REPO_ROOT / "results" / f"eval_report_{stamp}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n리포트 저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
