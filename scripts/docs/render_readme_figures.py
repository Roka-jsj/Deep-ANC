#!/usr/bin/env python3
"""README 에 들어가는 그림을 **실측 산출물에서** 다시 만든다.

손으로 그린 그림은 시간이 지나면 조용히 거짓말이 된다. 이 스크립트는 저장소에 있는
실제 파일만 읽어 그림을 만들고, 각 그림의 출처를 캡션에 박아 넣는다. 숫자가 바뀌면
그림도 바뀌어야 하고, 원본이 없으면 만들지 않고 실패한다.

    .venv/bin/python scripts/docs/render_readme_figures.py

덕트 도면만 예외다 — 이건 물리 치수라 ``configs/duct.yaml`` 에서 읽어 SVG 로 그린다.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from deep_anc.config import load_yaml  # noqa: E402

IMAGES = REPO / "assets" / "images"
DIAGRAMS = REPO / "assets" / "diagrams"

INK = "#1f2933"
ACCENT = "#2f6f4f"
WARN = "#b4532a"
MUTED = "#8b949e"


def _korean_font() -> str:
    """한글 글리프가 있는 폰트를 찾아 matplotlib 에 등록한다.

    기본 DejaVu Sans 에는 한글이 없어서 라벨이 통째로 두부(□)가 된다. 조용히 그렇게
    되면 그림은 만들어지고 README 에 올라간 뒤에야 발견된다 — 그래서 못 찾으면 만들지
    않고 멈춘다.
    """

    from matplotlib import font_manager

    candidates = sorted(Path("/usr/share/fonts").rglob("NotoSansCJK*.ttc"))
    candidates += sorted(Path("/usr/share/fonts").rglob("NanumGothic*.ttf"))
    for path in candidates:
        try:
            font_manager.fontManager.addfont(str(path))
            return font_manager.FontProperties(fname=str(path)).get_name()
        except (OSError, RuntimeError):
            continue
    raise SystemExit(
        "[중단] 한글 글리프를 가진 폰트를 찾지 못했습니다. "
        "그림 라벨이 깨진 채로 올라가는 것을 막기 위해 생성하지 않습니다."
    )


plt.rcParams.update(
    {
        "font.family": _korean_font(),
        "axes.unicode_minus": False,
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "font.size": 9,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)


def _require(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"[중단] 원본이 없습니다: {path.relative_to(REPO)}")
    return path


# ---------------------------------------------------------------------------
# 1. 덕트 도면 — configs/duct.yaml 이 단일 출처
# ---------------------------------------------------------------------------


def render_duct_layout() -> Path:
    duct = load_yaml(REPO / "configs" / "duct.yaml")
    pos = duct["positions_m"]
    geom = duct["duct"]
    acoustics = duct["acoustics"]

    length = float(geom["interior_length_m"])
    side = float(geom["cross_section_m"][0])
    scale = 900.0 / (length + 0.16)      # px per metre
    x0, y0 = 70.0, 120.0
    height = side * scale * 1.4          # 시각적으로 알아보게 단면을 과장한다

    def px(metres: float) -> float:
        return x0 + metres * scale

    def marker(name: str, metres: float, label: str, colour: str, above: bool) -> str:
        x = px(metres)
        y = y0 - 8 if above else y0 + height + 8
        ty = y - 10 if above else y + 22
        anchor = "baseline" if above else "hanging"
        return (
            f'  <line x1="{x:.1f}" y1="{y0 if above else y0 + height:.1f}" '
            f'x2="{x:.1f}" y2="{y:.1f}" stroke="{colour}" stroke-width="2"/>\n'
            f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{colour}"/>\n'
            f'  <text x="{x:.1f}" y="{ty:.1f}" text-anchor="middle" '
            f'dominant-baseline="{anchor}" font-size="13" fill="{colour}" '
            f'font-family="Noto Sans CJK KR, Noto Sans KR, Apple SD Gothic Neo, Malgun Gothic, sans-serif">{label}</text>\n'
            f'  <text x="{x:.1f}" y="{ty + (-16 if above else 16):.1f}" text-anchor="middle" '
            f'dominant-baseline="{anchor}" font-size="11" fill="{MUTED}" '
            f'font-family="Noto Sans CJK KR, Noto Sans KR, Apple SD Gothic Neo, Malgun Gothic, sans-serif">x = {metres:.3f} m</text>\n'
        )

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="330" '
        'viewBox="0 0 1040 330">\n',
        '  <rect width="1040" height="330" fill="white"/>\n',
        # 덕트 몸통
        f'  <rect x="{x0:.1f}" y="{y0:.1f}" width="{length * scale:.1f}" '
        f'height="{height:.1f}" fill="#eef2f5" stroke="{INK}" stroke-width="2.5"/>\n',
        # 좌측 폐단
        f'  <rect x="{x0 - 10:.1f}" y="{y0 - 6:.1f}" width="10" '
        f'height="{height + 12:.1f}" fill="{INK}"/>\n',
        f'  <text x="{x0 - 16:.1f}" y="{y0 + height / 2:.1f}" text-anchor="end" '
        f'dominant-baseline="middle" font-size="12" fill="{INK}" '
        f'font-family="Noto Sans CJK KR, Noto Sans KR, Apple SD Gothic Neo, Malgun Gothic, sans-serif">closed</text>\n',
        # 개방단
        f'  <text x="{px(length) + 16:.1f}" y="{y0 + height / 2:.1f}" '
        f'dominant-baseline="middle" font-size="12" fill="{INK}" '
        f'font-family="Noto Sans CJK KR, Noto Sans KR, Apple SD Gothic Neo, Malgun Gothic, sans-serif">open</text>\n',
    ]
    parts.append(marker("ns", float(pos["noise_speaker"]), "NS  소음 스피커", WARN, True))
    parts.append(marker("cs", float(pos["cancel_speaker"]), "CS  상쇄 스피커", ACCENT, True))
    parts.append(marker("ref", float(pos["reference_mic"]), "REF  기준 마이크", INK, False))
    parts.append(marker("err", float(pos["error_mic"]), "ERR  에러 마이크", INK, False))

    band = acoustics["realistic_target_band_hz"]
    caption = (
        f'단면 {geom["cross_section_m"][0] * 1000:.0f}×{geom["cross_section_m"][1] * 1000:.0f} mm '
        f'· 내부 길이 {length:.3f} m · closed-open '
        f'· 평면파 차단 {acoustics["plane_wave_cutoff_hz"]:.0f} Hz '
        f'· 목표 대역 {band[0]:.0f}–{band[1]:.0f} Hz '
        f'· 축방향 공진 {", ".join(str(v) for v in acoustics["axial_resonances_hz"])} Hz'
    )
    parts.append(
        f'  <text x="70" y="300" font-size="12" fill="{INK}" '
        f'font-family="Noto Sans CJK KR, Noto Sans KR, Apple SD Gothic Neo, Malgun Gothic, sans-serif">{caption}</text>\n'
    )
    parts.append(
        f'  <text x="70" y="46" font-size="16" font-weight="bold" fill="{INK}" '
        f'font-family="Noto Sans CJK KR, Noto Sans KR, Apple SD Gothic Neo, Malgun Gothic, sans-serif">아크릴 덕트 배치 (원점 = 좌측 폐단)</text>\n'
    )
    parts.append("</svg>\n")

    out = DIAGRAMS / "duct_layout.svg"
    out.write_text("".join(parts), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# 2. 실기 ANC 시연 — results/session_*/ 의 WAV 와 metrics.csv
# ---------------------------------------------------------------------------


def render_anc_demo(session: Path) -> Path:
    import soundfile as sf

    row = next(csv.DictReader((_require(session / "metrics.csv")).open(encoding="utf-8")))
    off, fs = sf.read(str(_require(session / "wav" / f"{row['scenario']}_{row['controller']}_off.wav")),
                      dtype="float64", always_2d=True)
    on, _ = sf.read(str(_require(session / "wav" / f"{row['scenario']}_{row['controller']}_on.wav")),
                    dtype="float64", always_2d=True)
    off, on = off[:, 0], on[:, 0]

    lo, hi = float(row["measure_low_hz"]), float(row["measure_high_hz"])
    att = float(row["attenuation_db"])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.5))

    # (a) 포락선 — OFF 구간과 ON 구간을 시간축에 이어 붙여 "켠 순간"을 보여준다
    def envelope(x: np.ndarray, win: int = 2048) -> np.ndarray:
        n = (x.size // win) * win
        return np.sqrt(np.mean(x[:n].reshape(-1, win) ** 2, axis=1))

    e_off, e_on = envelope(off), envelope(on)
    t_off = np.arange(e_off.size) * 2048 / fs
    t_on = t_off[-1] + 2048 / fs + np.arange(e_on.size) * 2048 / fs
    ax = axes[0]
    ax.plot(t_off, 20 * np.log10(e_off + 1e-12), color=WARN, lw=1.2, label="ANC OFF")
    ax.plot(t_on, 20 * np.log10(e_on + 1e-12), color=ACCENT, lw=1.2, label="ANC ON")
    ax.axvline(t_off[-1], color=INK, ls="--", lw=1.0)
    ax.annotate("ANC ON", xy=(t_off[-1], ax.get_ylim()[1]), xytext=(4, -12),
                textcoords="offset points", fontsize=9, color=INK)
    ax.set_xlabel("시간 [s]")
    ax.set_ylabel("에러 마이크 RMS [dBFS]")
    ax.set_title("(a) 에러 마이크 레벨", loc="left", fontsize=10)
    ax.legend(loc="lower left", framealpha=0.9)

    # (b) 스펙트럼 — 측정 대역을 음영으로 표시
    def spectrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nfft = 8192
        n = (x.size // nfft) * nfft
        frames = x[:n].reshape(-1, nfft) * np.hanning(nfft)
        power = np.mean(np.abs(np.fft.rfft(frames, axis=1)) ** 2, axis=0)
        return np.fft.rfftfreq(nfft, 1.0 / fs), 10 * np.log10(power + 1e-20)

    f, p_off = spectrum(off)
    _, p_on = spectrum(on)
    ax = axes[1]
    ax.axvspan(lo, hi, color=ACCENT, alpha=0.08)
    ax.semilogx(f[1:], p_off[1:], color=WARN, lw=1.0, label="ANC OFF")
    ax.semilogx(f[1:], p_on[1:], color=ACCENT, lw=1.0, label="ANC ON")
    ax.set_xlim(50, 8000)
    ax.set_xlabel("주파수 [Hz]")
    ax.set_ylabel("파워 [dB]")
    ax.set_title(f"(b) 스펙트럼 · {lo:.0f}–{hi:.0f} Hz 감쇠 {att:+.2f} dB",
                 loc="left", fontsize=10)
    ax.legend(loc="upper right", framealpha=0.9)

    fig.suptitle(
        f"실기 ANC — {session.name} · 사전학습 tiny(ONNX Runtime) · 실제 덕트/마이크/스피커",
        fontsize=10.5, y=1.04,
    )
    out = IMAGES / "anc_demo.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 3. 실측 데이터셋 구성 — data/manifests/*.jsonl
# ---------------------------------------------------------------------------


def render_dataset(manifest_dir: Path) -> Path:
    # 분할은 파일 이름이 아니라 **레코드 안의 split 필드**가 정한다. 파일명으로 추론하면
    # 한 파일에 세 분할이 들어 있는 현재 형식에서 전부 train 으로 보인다.
    rows: list[dict] = []
    for path in sorted(manifest_dir.glob("recorded_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"[중단] manifest 가 없습니다: {manifest_dir}")
    missing = [r for r in rows if not r.get("split")]
    if missing:
        raise SystemExit(f"[중단] split 필드가 없는 항목 {len(missing)}개")

    families = sorted({r["source_family"] for r in rows})
    splits = ["train", "val", "test"]
    colours = {"train": ACCENT, "val": "#4a7fa5", "test": WARN}

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.2))

    ax = axes[0]
    bottom = np.zeros(len(families))
    for split in splits:
        counts = np.array(
            [sum(1 for r in rows if r["source_family"] == f and r["split"] == split)
             for f in families], dtype=float
        )
        ax.bar(families, counts, bottom=bottom, color=colours[split], label=split, width=0.62)
        bottom += counts
    for index, total in enumerate(bottom):
        ax.text(index, total + 0.5, f"{int(total)}", ha="center", fontsize=9)
    ax.set_ylabel("세션 수")
    ax.set_title(
        f"(a) 소스 계열 × 분할 — 총 {len(rows)}세션 / "
        f"{len({r['group_id'] for r in rows})}그룹",
        loc="left", fontsize=10, pad=14,
    )
    ax.legend(framealpha=0.9, loc="lower right", ncol=3, fontsize=8)
    ax.margins(y=0.24)

    ax = axes[1]
    minutes = {
        f: sum(float(r["duration_s"]) for r in rows if r["source_family"] == f) / 60.0
        for f in families
    }
    ax.barh(families, [minutes[f] for f in families], color=ACCENT, height=0.6)
    for index, f in enumerate(families):
        ax.text(minutes[f] + 0.2, index, f"{minutes[f]:.1f}분", va="center", fontsize=9)
    total = sum(minutes.values())
    ax.set_xlabel("녹음 분량 [분]")
    ax.set_title(f"(b) 계열별 분량 — 합계 {total:.1f}분", loc="left", fontsize=10)
    ax.margins(x=0.16)

    fig.suptitle(
        "실측 덕트 녹음 데이터셋 — 그룹 단위 분할(같은 원본이 두 분할에 걸치지 않음)",
        fontsize=10.5, y=1.06,
    )
    out = IMAGES / "dataset_composition.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 4. 지연 물리 — duct.yaml + data_sim.yaml 이 만드는 lead
# ---------------------------------------------------------------------------


def render_timing() -> Path:
    duct = load_yaml(REPO / "configs" / "duct.yaml")
    handoff = int(duct["secondary_path"]["handoff_extra_samples"])
    secondary = np.load(REPO / duct["secondary_path"]["npz"], allow_pickle=False)
    s_delay = int(np.asarray(secondary["delay_samples"]).reshape(-1)[0])
    data = load_yaml(REPO / "configs" / "data_sim.yaml")
    lead = int(data["digital_reference_lead_samples"])
    p_delay = s_delay + handoff - lead

    fig, ax = plt.subplots(figsize=(9.5, 2.5))
    bars = [
        ("P(z)  소음 출력 → ERR 도달", p_delay, WARN, 0),
        ("S(z)  상쇄 출력 → ERR 도달", s_delay, ACCENT, 0),
        ("handoff  3-스레드 1 hop", handoff, MUTED, s_delay),
    ]
    for index, (label, width, colour, start) in enumerate(bars):
        y = len(bars) - 1 - index
        ax.barh(y, width, left=start, color=colour, height=0.55)
        ax.text(start + width / 2, y, f"{width}", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")
        ax.text(-40, y, label, ha="right", va="center", fontsize=9)

    ax.annotate(
        "", xy=(p_delay, -0.75), xytext=(s_delay + handoff, -0.75),
        arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4),
    )
    ax.text(p_delay - 60, -0.75,
            f"lead = S {s_delay} + handoff {handoff} − P {p_delay} = {lead} 샘플",
            ha="right", va="center", fontsize=10, color=INK)
    ax.set_xlim(-620, max(s_delay + handoff, p_delay) * 1.06)
    ax.set_ylim(-1.2, len(bars) - 0.3)
    ax.set_yticks([])
    ax.set_xlabel("48 kHz 샘플")
    ax.grid(axis="y", visible=False)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.set_title(
        "digital reference 의 선행량 — 상쇄음이 소음보다 늦게 도착하는 만큼 미리 본다",
        loc="left", fontsize=10.5,
    )
    out = IMAGES / "timing_budget.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> int:
    IMAGES.mkdir(parents=True, exist_ok=True)
    DIAGRAMS.mkdir(parents=True, exist_ok=True)

    sessions = sorted((REPO / "results").glob("session_*/metrics.csv"))
    if not sessions:
        raise SystemExit("[중단] results/session_*/metrics.csv 가 없습니다")

    produced = [
        render_duct_layout(),
        render_anc_demo(sessions[-1].parent),
        render_dataset(REPO / "data" / "manifests"),
        render_timing(),
    ]
    for path in produced:
        size = path.stat().st_size
        print(f"  {path.relative_to(REPO)}  ({size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
