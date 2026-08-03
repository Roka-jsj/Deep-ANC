#!/usr/bin/env python3
"""데이터셋 QA — 학습·실시간 추론 적합성 자동 점검 (다운로드 직후 실행).

태그별로 실제 오디오를 표본 검사한다:
  샘플레이트 분포 / 총 시간 / 읽기 실패율 / 클리핑 / 무음 비율 / RMS 분포
  / 덕트 관심대역(80–800 / 800–1633 / 1633+ Hz) 에너지 비율

  .venv/bin/python scripts/data/validate_noise_pool.py  # → data/manifests/dataset_qa.md
경고는 리포트에 표기하고 계속 진행, 치명(태그 전체 읽기 불가 등)은 종료코드 1.
판단 기준 근거: docs/03 §8 (학습 분포 ↔ 배포 분포 정합).
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.config import REPO_ROOT                      # noqa: E402
from deep_anc.data.manifest import read_manifest           # noqa: E402

BANDS = [(80.0, 800.0), (800.0, 1633.0), (1633.0, 24000.0)]


def analyze_file(path: str, max_seconds: float = 10.0) -> dict | None:
    try:
        info = sf.info(path)
        frames = min(int(info.frames), int(max_seconds * info.samplerate))
        if frames < info.samplerate // 10:
            return None
        data, sr = sf.read(path, frames=frames, dtype="float32", always_2d=True)
    except Exception:
        return None
    x = data.mean(axis=1)
    peak = float(np.max(np.abs(x)) + 1e-12)
    rms = float(np.sqrt(np.mean(x**2)) + 1e-12)
    clip_ratio = float(np.mean(np.abs(x) >= 0.99))
    freqs, psd = signal.welch(x, fs=sr, nperseg=min(4096, x.size))
    total = float(np.trapz(psd, freqs) + 1e-20)
    band_frac = []
    for lo, hi in BANDS:
        m = (freqs >= lo) & (freqs < min(hi, sr / 2))
        band_frac.append(float(np.trapz(psd[m], freqs[m])) / total if m.any() else 0.0)
    return {
        "sr": sr,
        "rms_dbfs": 20 * np.log10(rms),
        "peak_dbfs": 20 * np.log10(peak),
        "clip": clip_ratio,
        "silent": rms < 10 ** (-60 / 20),
        "band_frac": band_frac,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--samples-per-tag", type=int, default=150)
    parser.add_argument("--out", default="data/manifests/dataset_qa.md")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    mdir = REPO_ROOT / args.manifest_dir
    manifests = sorted(mdir.glob("*.jsonl")) if mdir.exists() else []
    if not manifests:
        print(f"manifest 없음: {mdir} — prepare_noise_pool.py 를 먼저 실행", file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)
    lines = [
        "# 데이터셋 QA 리포트 (학습·추론 적합성)",
        "",
        "| 태그 | 파일수 | 시간(h) | SR 분포 | 실패율 | 평균RMS(dBFS) | 클립% | 무음% | 대역에너지 80-800/800-1.6k/1.6k+ | 판정 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    hard_fail = False

    for mpath in manifests:
        tag = mpath.stem
        if tag.startswith("recorded"):
            continue
        entries = read_manifest(mpath)
        n = len(entries)
        hours = sum(e.get("duration_s", 0) for e in entries) / 3600
        pick = rng.choice(n, size=min(args.samples_per_tag, n), replace=False)
        results, fails = [], 0
        for i in pick:
            r = analyze_file(entries[int(i)]["path"])
            if r is None:
                fails += 1
            else:
                results.append(r)

        if not results:
            lines.append(f"| {tag} | {n} | {hours:.1f} | - | 100% | - | - | - | - | **치명: 전부 읽기 실패** |")
            hard_fail = True
            continue

        srs = Counter(r["sr"] for r in results)
        sr_txt = " ".join(f"{k//1000}k:{v}" for k, v in srs.most_common(3))
        fail_pct = 100 * fails / max(1, len(pick))
        rms = float(np.mean([r["rms_dbfs"] for r in results]))
        clip = 100 * float(np.mean([r["clip"] for r in results]))
        silent = 100 * float(np.mean([r["silent"] for r in results]))
        bf = np.mean([r["band_frac"] for r in results], axis=0) * 100

        warns = []
        if fail_pct > 2:
            warns.append(f"읽기실패 {fail_pct:.0f}%")
        if clip > 0.5:
            warns.append("클리핑 과다")
        if silent > 30:
            warns.append("무음 과다")
        if bf[0] + bf[1] < 10:
            warns.append("덕트 대역(<1.6kHz) 에너지 빈약")
        if all(sr < 44100 for sr in srs):
            warns.append("저샘플레이트(고주파 없음 — 저역 학습용으로만)")
        verdict = "OK" if not warns else "⚠ " + ", ".join(warns)

        lines.append(
            f"| {tag} | {n} | {hours:.1f} | {sr_txt} | {fail_pct:.0f}% | {rms:.1f} | "
            f"{clip:.2f} | {silent:.0f} | {bf[0]:.0f}/{bf[1]:.0f}/{bf[2]:.0f}% | {verdict} |"
        )
        print(f"[{tag}] {n}개 {hours:.1f}h | SR {sr_txt} | RMS {rms:.1f}dBFS | {verdict}")

    lines += [
        "",
        "판정 기준: 배포(덕트 실기) 정합 관점 — 학습 로더가 48kHz 리샘플·RMS 정규화·레벨 랜덤화",
        "(−35~−10 dBFS)를 수행하므로 원본 레벨 편차는 흡수된다. 관건은 ① 읽기 가능성",
        "② 덕트 제어대역(<1.6kHz) 에너지 존재 ③ 과도한 클리핑/무음이다 (docs/03 §8).",
    ]
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n리포트: {out}")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
