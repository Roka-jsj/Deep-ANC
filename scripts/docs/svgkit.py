"""논문 그림용 최소 SVG 빌더.

matplotlib 로 블록 다이어그램을 그리면 좌표 계산이 축 변환에 묶여 읽기 어려워지고,
글꼴 대체가 조용히 일어난다. 여기서는 픽셀 좌표를 직접 쓰고 글꼴은 뷰어에 맡긴다
(GitHub 는 SVG 안의 텍스트를 브라우저 글꼴로 렌더링하므로 한글이 깨지지 않는다).

색은 의미에 고정한다 — 같은 색이 다른 그림에서 다른 뜻을 가지면 그림 여러 장이
한 시스템으로 읽히지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 한글 글리프가 있는 폰트를 **앞에** 둔다. 브라우저는 글리프 단위로 대체하지만
# cairo(로컬 검증에 쓰는 래스터라이저)는 목록의 첫 항목만 보고 없는 글리프를 두부로
# 그린다. 순서를 바꾸면 검증이 실제 렌더링과 같아진다.
SANS = "'Noto Sans CJK KR', 'Noto Sans KR', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif"
MONO = "'Noto Sans Mono CJK KR', 'DejaVu Sans Mono', Menlo, Consolas, monospace"

# 의미 고정 팔레트
INK = "#16202c"          # 본문·테두리
MUTED = "#7b8794"        # 보조 설명
LINE = "#9aa5b1"         # 연결선
PANEL = "#f4f6f8"        # 중립 블록
ENCODER = "#2f6f4f"      # 학습형 encoder/decoder (Conv-TasNet 계열)
TCN = "#2b5d8a"          # dilated causal TCN (WaveNet 계열)
RECURR = "#7a4b8f"       # 순환 상태 (GLSTM, GCRN 계열)
ATTN = "#b4532a"         # attention (base 전용)
PHYS = "#8a6d1f"         # 물리 경로 P(z)/S(z)
DANGER = "#a8322d"       # 제약·리미터


def esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


@dataclass
class Canvas:
    width: float
    height: float
    parts: list[str] = field(default_factory=list)
    defs: list[str] = field(default_factory=list)

    # -- 기본 도형 -------------------------------------------------------
    def rect(
        self, x: float, y: float, w: float, h: float, *,
        fill: str = PANEL, stroke: str = INK, width: float = 1.6,
        radius: float = 8.0, dash: str | None = None, opacity: float = 1.0,
    ) -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{radius:.1f}" fill="{fill}" fill-opacity="{opacity}" '
            f'stroke="{stroke}" stroke-width="{width}"{d}/>'
        )

    def text(
        self, x: float, y: float, content: str, *,
        size: float = 13, fill: str = INK, anchor: str = "middle",
        family: str = SANS, weight: str = "normal", baseline: str = "middle",
        opacity: float = 1.0,
    ) -> None:
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" fill-opacity="{opacity}" '
            f'text-anchor="{anchor}" dominant-baseline="{baseline}">{esc(content)}</text>'
        )

    def lines(self, x: float, y: float, rows: list[str], *, gap: float = 15, **kw) -> None:
        for index, row in enumerate(rows):
            self.text(x, y + index * gap, row, **kw)

    def line(
        self, x1: float, y1: float, x2: float, y2: float, *,
        stroke: str = LINE, width: float = 1.6, dash: str | None = None,
        arrow: bool = False, opacity: float = 1.0,
    ) -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        self.parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{width}" stroke-opacity="{opacity}" '
            f'stroke-linecap="round"{d}{marker}/>'
        )

    def path(
        self, d: str, *, stroke: str = LINE, width: float = 1.6,
        fill: str = "none", dash: str | None = None, arrow: bool = False,
        opacity: float = 1.0,
    ) -> None:
        dd = f' stroke-dasharray="{dash}"' if dash else ""
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        self.parts.append(
            f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" '
            f'stroke-opacity="{opacity}" stroke-linecap="round" '
            f'stroke-linejoin="round"{dd}{marker}/>'
        )

    def circle(self, cx: float, cy: float, r: float, *,
               fill: str = "#ffffff", stroke: str = INK, width: float = 1.6) -> None:
        self.parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{width}"/>'
        )

    # -- 합성 도형 -------------------------------------------------------
    def block(
        self, x: float, y: float, w: float, h: float, title: str, *,
        subtitle: list[str] | None = None, colour: str = INK,
        fill: str | None = None, title_size: float = 14, dash: str | None = None,
    ) -> tuple[float, float]:
        """제목 + 여러 줄 부제를 담은 블록. 반환은 (중심 x, 중심 y)."""

        self.rect(x, y, w, h, fill=fill or "#ffffff", stroke=colour, dash=dash)
        self.rect(x, y, w, 4, fill=colour, stroke=colour, radius=2, width=0)
        cx = x + w / 2
        rows = subtitle or []
        top = y + h / 2 - (len(rows) * 14) / 2 + 2
        self.text(cx, top, title, size=title_size, weight="600", fill=colour)
        for index, row in enumerate(rows):
            self.text(cx, top + 17 + index * 14, row, size=11, fill=MUTED, family=MONO)
        return cx, y + h / 2

    def sum_node(self, cx: float, cy: float, r: float = 15) -> None:
        self.circle(cx, cy, r)
        self.line(cx - r * 0.5, cy, cx + r * 0.5, cy, stroke=INK, width=1.8)
        self.line(cx, cy - r * 0.5, cx, cy + r * 0.5, stroke=INK, width=1.8)

    def brace(self, x: float, y1: float, y2: float, *, depth: float = 10,
              stroke: str = MUTED) -> None:
        mid = (y1 + y2) / 2
        self.path(
            f"M {x:.1f} {y1:.1f} q {depth:.1f} 0 {depth:.1f} {(mid - y1) / 2:.1f} "
            f"q 0 {(mid - y1) / 2:.1f} {depth:.1f} {(mid - y1) / 2:.1f} "
            f"q -{depth:.1f} 0 -{depth:.1f} {(y2 - mid) / 2:.1f} "
            f"q 0 {(y2 - mid) / 2:.1f} -{depth:.1f} {(y2 - mid) / 2:.1f}",
            stroke=stroke, width=1.3,
        )

    def caption(self, x: float, y: float, label: str, body: str, *,
                size: float = 12.5, width_hint: float = 0) -> None:
        self.text(x, y, label, size=size, weight="700", anchor="start", fill=INK)
        self.text(x + width_hint, y, body, size=size, anchor="start", fill=MUTED)

    def render(self, *, background: str = "#ffffff") -> str:
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width:.0f}" '
            f'height="{self.height:.0f}" viewBox="0 0 {self.width:.0f} {self.height:.0f}">'
        )
        defs = (
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{LINE}"/></marker>'
            + "".join(self.defs)
            + "</defs>"
        )
        body = "\n  ".join(self.parts)
        return (
            f"{head}\n  {defs}\n"
            f'  <rect width="{self.width:.0f}" height="{self.height:.0f}" fill="{background}"/>\n'
            f"  {body}\n</svg>\n"
        )
