#!/usr/bin/env python3
"""이미 녹음된 세션 npz 에서 표준 산출물(CSV/WAV/summary)을 다시 만든다.

실기 세션은 스피커를 다시 울려야 얻을 수 있다. 그래서 리포트 생성이 실패했거나 산출물
형식을 바꿨을 때 **원신호로부터 재계산**할 수 있어야 한다. 이 스크립트가 그 경로다.

    .venv/bin/python scripts/eval/rebuild_session_artifacts.py \
      --npz results/session_20260804_0939/*.npz --out results/session_20260804_0939

npz 에 판정 구간(off_segment/on_segment)이 저장돼 있으면 그것을 쓰고, 없는 구형 파일이면
anc_gain 으로 OFF/ON 구간을 다시 잘라낸다. 자를 때는 게이트 램프를 피해 경계를 버린다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.eval.artifacts import (  # noqa: E402
    atomic_write_text,
    write_csv,
    write_wav_pair,
)
from deep_anc.eval.metrics import (  # noqa: E402
    band_nmse_db,
    nmse_db,
    octave_band_attenuation,
)

DEFAULT_BANDS = [125, 250, 500, 1000, 2000, 4000, 8000]
# 게이트 램프와 엔진 워밍업을 피하는 여유. 이 값을 줄이면 감쇠가 좋아 보이지만 그건
# 램프 구간의 낮은 출력을 성능으로 세는 것이다.
ON_LEAD_IN_SECONDS = 2.0
EDGE_GUARD_SECONDS = 0.5


def segments(data, fs: int) -> tuple[np.ndarray, np.ndarray]:
    if "off_segment" in data and "on_segment" in data:
        return np.asarray(data["off_segment"]), np.asarray(data["on_segment"])
    err = np.asarray(data["err"], dtype=np.float64)
    gain = np.asarray(data["anc_gain"], dtype=np.float64)
    full_on = np.flatnonzero(gain >= 0.999)
    if full_on.size == 0:
        raise ValueError("ANC 가 full-on 인 구간이 없습니다 — 성능으로 인정할 수 없습니다")
    start, stop = int(full_on[0]), int(full_on[-1])
    off = err[int(1.0 * fs) : max(int(1.0 * fs) + 1, start - int(EDGE_GUARD_SECONDS * fs))]
    on = err[start + int(ON_LEAD_IN_SECONDS * fs) : stop - int(EDGE_GUARD_SECONDS * fs)]
    if off.size < fs or on.size < fs:
        raise ValueError("OFF/ON 구간이 1초 미만입니다 — 비교하지 않습니다")
    return off, on


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default=None)
    parser.add_argument("--note", action="append", default=[])
    args = parser.parse_args(argv)

    run_dir = Path(args.out)
    (run_dir / "wav").mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for path in sorted(Path(p) for p in args.npz):
        data = np.load(path)
        fs = int(data["fs"])
        off, on = segments(data, fs)
        trusted = tuple(float(v) for v in np.asarray(data["trusted_band_hz"]).ravel()[:2])

        nmse_trusted = band_nmse_db(off, on, fs, trusted)
        nmse_fullband = nmse_db(off, on)
        band_att = octave_band_attenuation(off, on, fs, DEFAULT_BANDS, trusted)

        stem = path.stem
        scenario, _, controller = stem.rpartition("_")
        wavs = write_wav_pair(run_dir / "wav", stem, off, on, fs)

        gain = np.asarray(data["anc_gain"], dtype=np.float64) if "anc_gain" in data else None
        row = {
            "scenario": scenario or stem,
            "controller": controller or "dl",
            "sample_rate_hz": fs,
            "off_seconds": off.size / fs,
            "on_seconds": on.size / fs,
            "trusted_low_hz": trusted[0],
            "trusted_high_hz": trusted[1],
            "trusted_attenuation_db": -nmse_trusted,
            "fullband_attenuation_db": -nmse_fullband,
            "gap_trusted_minus_fullband_db": nmse_trusted - nmse_fullband,
            "off_rms": float(np.sqrt(np.mean(off**2))),
            "on_rms": float(np.sqrt(np.mean(on**2))),
            "on_duty_fraction": float(np.mean(gain >= 0.999)) if gain is not None else "",
        }
        for band in band_att:
            center = int(band["center_hz"])
            row[f"band_{center}_att_db"] = float(band["attenuation_db"])
            row[f"band_{center}_trusted"] = bool(band["trusted"])
        row["source_npz"] = path.name
        row["wav_ab"] = wavs["ab"].name
        rows.append(row)
        print(
            f"[{row['scenario']:10s}] trusted {row['trusted_attenuation_db']:+6.2f} dB · "
            f"fullband {row['fullband_attenuation_db']:+6.2f} dB · "
            f"OFF {row['off_seconds']:.1f}s / ON {row['on_seconds']:.1f}s"
        )

    if not rows:
        print("[중단] 처리된 세션이 없습니다.", file=sys.stderr)
        return 2

    csv_path = write_csv(run_dir / "metrics.csv", rows)

    lines = [f"# {args.title or '실기 세션 재계산 리포트'}", ""]
    lines += [f"- {note}" for note in args.note]
    lines += [
        f"- 원신호 {len(rows)}개에서 재계산. 기계가 읽는 단일 출처는 `metrics.csv` 다.",
        f"- OFF/ON 구간은 anc_gain 이 full-on 인 구간에서 앞 {ON_LEAD_IN_SECONDS:.0f}초·"
        f"양끝 {EDGE_GUARD_SECONDS:.1f}초를 버리고 잘랐다.",
        "",
        "| 시나리오 | trusted 감쇠(dB) | fullband 감쇠(dB) | OFF/ON 길이(s) | A/B 파일 |",
        "|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | **{row['trusted_attenuation_db']:+.2f}** | "
            f"{row['fullband_attenuation_db']:+.2f} | "
            f"{row['off_seconds']:.1f} / {row['on_seconds']:.1f} | `wav/{row['wav_ab']}` |"
        )
    lines += ["", "## 옥타브밴드 감쇠 (dB, *=trusted 대역 밖)", "",
              "| 시나리오 | " + " | ".join(f"{c}Hz" for c in DEFAULT_BANDS) + " |",
              "|---" * (len(DEFAULT_BANDS) + 1) + "|"]
    for row in rows:
        cells = [
            f"{row[f'band_{c}_att_db']:+.1f}{'' if row[f'band_{c}_trusted'] else '*'}"
            for c in DEFAULT_BANDS
        ]
        lines.append(f"| {row['scenario']} | " + " | ".join(cells) + " |")
    lines.append("")
    summary_path = atomic_write_text(run_dir / "summary.md", "\n".join(lines) + "\n")

    print(f"\n산출물:\n  {csv_path}\n  {summary_path}\n  {run_dir / 'wav'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
