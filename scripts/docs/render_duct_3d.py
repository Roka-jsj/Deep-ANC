#!/usr/bin/env python3
"""덕트 배치를 3D(축측투영) SVG 로 그린다.

2D 측면도는 두 가지를 못 보여준다. 둘 다 이 시스템의 물리에 직접 관계된다.

* 상쇄 스피커가 **상면 side-branch** 다 — 상쇄음이 축방향이 아니라 위에서 들어온다.
* 단면이 정사각형 105mm 다 — 이 값이 평면파 차단 1633Hz 를 결정한다.

좌표·치수는 전부 ``configs/duct.yaml`` 에서 읽는다. 그림 안에는 짧은 라벨과 치수만 넣고
설명은 README 캡션이 맡는다.

    .venv/bin/python scripts/docs/render_duct_3d.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from svgkit import Canvas, ENCODER, INK, MONO, MUTED, PHYS  # noqa: E402

from deep_anc.config import load_yaml  # noqa: E402

OUT = REPO / "assets" / "diagrams"

# 축측투영. X = 덕트 길이(화면 오른쪽), Z = 높이(화면 위), Y = 깊이(오른쪽 위로 단축).
# 단축률 0.42 는 정투영처럼 보이지 않으면서 상면이 충분히 드러나는 값이다.
DEPTH_U = 0.46
DEPTH_V = 0.34


def render() -> Path:
    duct = load_yaml(REPO / "configs" / "duct.yaml")
    geom, pos, ac = duct["duct"], duct["positions_m"], duct["acoustics"]
    length = float(geom["interior_length_m"])
    side_w, side_h = (float(v) for v in geom["cross_section_m"])

    scale = 640.0 / length          # px per metre (길이 축)
    depth_px = side_w * scale * 3.4  # 단면을 시각적으로 과장한다
    height_px = side_h * scale * 3.4
    ox, oy = 210.0, 300.0

    def proj(x: float, y: float, z: float) -> tuple[float, float]:
        """덕트 좌표 → 화면 좌표. y, z 는 단면 안에서 0..1 로 정규화한다."""
        return (
            ox + x * scale + y * depth_px * DEPTH_U,
            oy - z * height_px - y * depth_px * DEPTH_V,
        )

    c = Canvas(1120, 460)

    # x=0 면은 뒤를 향하므로 면으로 그리지 않는다. 대신 좌측 모서리를 두껍게 그려
    # 벽을 옆에서 본 것으로 표현한다.
    top = [proj(0, 0, 1), proj(length, 0, 1), proj(length, 1, 1), proj(0, 1, 1)]
    front = [proj(0, 0, 0), proj(length, 0, 0), proj(length, 0, 1), proj(0, 0, 1)]
    opening = [proj(length, 0, 0), proj(length, 1, 0),
               proj(length, 1, 1), proj(length, 0, 1)]

    c.polygon(top, fill="#e7ecf1", stroke=INK)
    c.polygon(front, fill="#f3f6f9", stroke=INK)
    # 개구 — 바깥 윤곽과 안쪽 윤곽을 겹쳐 보어 깊이를 암시한다
    c.polygon(opening, fill="#ffffff", stroke=INK)
    inner = [proj(length, 0.12, 0.12), proj(length, 0.88, 0.12),
             proj(length, 0.88, 0.88), proj(length, 0.12, 0.88)]
    c.polygon(inner, fill="#eef2f6", stroke=MUTED, width=1.2, dash="5 4")

    # 좌측 폐단 — 벽을 옆에서 본 두꺼운 모서리
    lx0, ly0 = proj(0, 0, 0)
    lx1, ly1 = proj(0, 0, 1)
    c.line(lx0, ly0, lx1, ly1, stroke="#3b4a5a", width=7)
    c.text(lx0 - 96, (ly0 + ly1) / 2 + 34, "closed", size=12.5, anchor="start",
           fill=MUTED)
    ax_open, ay_open = proj(length, 1, 1)
    c.text(ax_open + 30, ay_open + 26, "open", size=12.5, anchor="start", fill=MUTED)

    # 소음 스피커 — 외부 부착, 축방향 방사. 덕트 왼쪽에 두고 안쪽을 향한 화살표.
    nsx, nsy = proj(0, 0.5, 0.5)
    c.rect(nsx - 116, nsy - 24, 46, 48, fill="#ffffff", stroke=PHYS, radius=5)
    c.polygon([(nsx - 70, nsy - 24), (nsx - 70, nsy + 24), (nsx - 46, nsy + 12),
               (nsx - 46, nsy - 12)], fill="#ffffff", stroke=PHYS)
    c.line(nsx - 44, nsy, nsx - 8, nsy, stroke=PHYS, width=1.6, arrow=True)
    c.text(nsx - 93, nsy - 40, "NS", size=13.5, weight="700", fill=PHYS)
    c.text(nsx - 93, nsy + 44, f"x = {float(pos['noise_speaker']):.3f}",
           size=11, family=MONO, fill=MUTED)

    def wall_marker(x, label, *, dy):
        px, py = proj(x, 0.0, 0.5)
        c.circle(px, py, 7, fill=INK, stroke=INK)
        c.line(px, py, px, py + dy, stroke=INK, width=1.5)
        c.text(px, py + dy + (16 if dy > 0 else -10), label, size=13.5,
               weight="700", fill=INK)
        c.text(px, py + dy + (33 if dy > 0 else -27), f"x = {x:.3f}",
               size=11, family=MONO, fill=MUTED)

    wall_marker(float(pos["reference_mic"]), "REF", dy=74)
    wall_marker(float(pos["error_mic"]), "ERR", dy=74)

    # 상쇄 스피커 — 상면 side-branch
    cs_x = float(pos["cancel_speaker"])
    cx, cy = proj(cs_x, 0.5, 1.0)
    c.ellipse(cx, cy, 18, 18 * DEPTH_V / DEPTH_U * 0.85, fill="#ffffff",
              stroke=ENCODER, width=2.0)
    c.circle(cx, cy, 5, fill=ENCODER, stroke=ENCODER)
    c.line(cx, cy - 12, cx + 6, cy - 66, stroke=ENCODER, width=1.5)
    c.text(cx + 12, cy - 70, "CS", size=13.5, weight="700", anchor="start",
           fill=ENCODER)
    c.text(cx + 12, cy - 52, f"x = {cs_x:.3f}   Ø40", size=11, family=MONO,
           anchor="start", fill=MUTED)

    # 치수 — 길이
    dim_y = oy + 118
    ax, _ = proj(0, 0, 0)
    bx, _ = proj(length, 0, 0)
    c.line(ax, dim_y, bx, dim_y, stroke=MUTED, width=1.2, arrow=True)
    c.line(bx, dim_y, ax, dim_y, stroke=MUTED, width=1.2, arrow=True)
    c.text((ax + bx) / 2, dim_y - 13, f"{length:.3f} m", size=12.5, family=MONO,
           fill=MUTED)

    # 치수 — 단면
    c.text(ax_open + 30, ay_open + 48,
           f"{side_w * 1000:.0f} × {side_h * 1000:.0f} mm",
           size=12, family=MONO, anchor="start", fill=MUTED)

    # 좌표 삼각대
    tx, ty = 74.0, 412.0
    for dx, dy, name in ((58, 0, "x"), (0, -44, "z"),
                         (32 * DEPTH_U * 1.7, -32 * DEPTH_V * 1.7, "y")):
        c.line(tx, ty, tx + dx, ty + dy, stroke=MUTED, width=1.4, arrow=True)
        c.text(tx + dx * 1.2 + 5, ty + dy * 1.2, name, size=12, family=MONO,
               fill=MUTED)

    # 음향 상수 — 숫자만
    c.text(
        1096, 440,
        f"f_cut {ac['plane_wave_cutoff_hz']:.0f} Hz   ·   "
        f"target {ac['realistic_target_band_hz'][0]:.0f}–"
        f"{ac['realistic_target_band_hz'][1]:.0f} Hz   ·   "
        f"axial {', '.join(str(v) for v in ac['axial_resonances_hz'])} Hz",
        size=11.5, family=MONO, anchor="end", fill=MUTED,
    )

    path = OUT / "fig0_duct_3d.svg"
    path.write_text(c.render(), encoding="utf-8")
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    path = render()
    print(f"  {path.relative_to(REPO)}  ({path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
