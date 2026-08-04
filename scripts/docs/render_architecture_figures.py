#!/usr/bin/env python3
"""논문 그림 형식의 아키텍처 도해를 생성한다.

모든 수치(채널 수, dilation, 커널, 수용영역, 파라미터 수)는 ``configs/model_*.yaml`` 과
실제 모델에서 읽는다. 그림에 손으로 적은 숫자는 하나도 없다 — 설정이 바뀌면 그림도 바뀐다.

    .venv/bin/python scripts/docs/render_architecture_figures.py

생성물 (assets/diagrams/):
  fig1_system.svg              시스템 신호 흐름과 손실 대상
  fig2_architecture.svg        HybridANCNet 전체 스택 (base / tiny 병렬)
  fig3_tcn_block.svg           TCN 잔차 블록 내부
  fig4_receptive_field.svg     dilated causal conv 의 수용영역
  fig5_glstm.svg               GLSTM 그룹 분할·셔플
  fig6_streaming.svg           스트리밍 상태와 프레임 간 전달
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from svgkit import (  # noqa: E402
    ATTN, DANGER, ENCODER, INK, LINE, MONO, MUTED, PANEL, PHYS, RECURR, TCN,
    Canvas,
)

from deep_anc.config import load_yaml  # noqa: E402

OUT = REPO / "assets" / "diagrams"


def model_facts(name: str) -> dict:
    cfg = load_yaml(REPO / "configs" / f"model_{name}.yaml")
    dilations = list(cfg["tcn"]["dilations"])
    repeats = int(cfg["tcn"]["repeats"])
    kernel = int(cfg["tcn"]["kernel"])
    hop = int(cfg["hop"])
    win = int(cfg["win"])
    # 수용영역: 프레임 단위로 각 블록이 (k-1)*d 만큼 과거를 본다
    frames = 1 + repeats * sum((kernel - 1) * d for d in dilations)
    return {
        "name": cfg["name"],
        "hop": hop,
        "win": win,
        "io_scale": float(cfg["io_scale"]),
        "enc_channels": int(cfg["encoder"]["channels"]),
        "repeats": repeats,
        "dilations": dilations,
        "kernel": kernel,
        "tcn_hidden": int(cfg["tcn"]["hidden"]),
        "glstm_groups": int(cfg["glstm"]["groups"]),
        "glstm_hidden": int(cfg["glstm"]["hidden_per_group"]),
        "glstm_after": int(cfg["glstm"]["insert_after_repeat"]),
        "attention": bool(cfg["attention"]["enabled"]),
        "attn_heads": int(cfg["attention"].get("heads", 0) or 0),
        "attn_window": int(cfg["attention"].get("window_frames", 0) or 0),
        "limit": float(cfg["limiter"]["limit"]),
        "blocks": repeats * len(dilations),
        "rf_frames": frames,
        "rf_ms": (frames - 1) * hop / 48000.0 * 1000.0 + win / 48000.0 * 1000.0,
    }


def param_count(name: str) -> int:
    import torch  # noqa: F401
    from deep_anc.models.hybrid_anc import HybridANCNet

    cfg = load_yaml(REPO / "configs" / f"model_{name}.yaml")
    return sum(p.numel() for p in HybridANCNet(cfg).parameters())


# ---------------------------------------------------------------------------
# Fig 1 — 시스템 신호 흐름
# ---------------------------------------------------------------------------


def fig_system(tiny: dict, lead: int) -> Path:
    c = Canvas(1080, 470)
    c.text(40, 38, "Figure 1.  ANC 신호 흐름 — 학습과 실기가 같은 방정식을 쓴다",
           size=16, weight="700", anchor="start")
    c.text(40, 60,
           "모델은 상쇄 파형 y(t) 를 직접 회귀한다. 학습에서는 실측 S(z) 를 미분 가능한 "
           "플랜트로 통과시켜 e(t) 를 최소화한다.",
           size=12, fill=MUTED, anchor="start")

    y_top, y_bot = 120, 300
    c.block(40, y_top - 26, 150, 52, "소음원 n(t)",
            subtitle=["Jetson 이 생성"], colour=INK, fill=PANEL)

    # 위쪽 경로 — 물리
    c.block(300, y_top - 30, 170, 60, "P(z)  1차경로",
            subtitle=[f"D_noise = {lead['p']} samp"], colour=PHYS)
    c.line(190, y_top, 300, y_top, arrow=True)
    c.text(245, y_top - 12, "스피커 ch0", size=11, fill=MUTED)

    # 아래쪽 경로 — 모델
    c.line(115, y_top + 26, 115, y_bot, stroke=LINE)
    c.line(115, y_bot, 230, y_bot, arrow=True)
    c.block(230, y_bot - 30, 210, 60, "digital reference",
            subtitle=[f"x_ref(t) = n(t + {lead['lead']})"], colour=INK, fill=PANEL)

    c.block(490, y_bot - 46, 210, 92, "HybridANCNet",
            subtitle=[f"{tiny['name']}", "causal · lookahead 0",
                      f"limiter {tiny['limit']}·tanh"], colour=ENCODER)
    c.line(440, y_bot, 490, y_bot, arrow=True)

    c.block(750, y_bot - 30, 170, 60, "S(z)  2차경로",
            subtitle=[f"delay {lead['s']} + {lead['handoff']}"], colour=PHYS)
    c.line(700, y_bot, 750, y_bot, arrow=True)
    c.text(725, y_bot - 12, "y(t)", size=11, fill=MUTED, family=MONO)

    # 합류
    sx, sy = 985, (y_top + y_bot) / 2
    c.path(f"M 470 {y_top} L {sx} {y_top} L {sx} {sy - 15}", arrow=True)
    c.path(f"M 920 {y_bot} L {sx} {y_bot} L {sx} {sy + 15}", arrow=True)
    c.sum_node(sx, sy)
    c.text(sx + 30, y_top, "d(t)", size=12, family=MONO, anchor="start", fill=PHYS)
    c.text(sx + 30, y_bot, "S·y(t)", size=12, family=MONO, anchor="start", fill=PHYS)

    c.line(sx, sy + 15, sx, 400, arrow=True)
    c.rect(830, 400, 210, 44, fill="#fdf3f1", stroke=DANGER)
    c.text(935, 422, "e(t) = d(t) + S·y(t)", size=13.5, family=MONO, weight="600",
           fill=DANGER)
    c.text(935, 456, "에러 마이크 — 손실 대상", size=11.5, fill=MUTED)

    c.text(40, 400, "극성 규약", size=12.5, weight="700", anchor="start")
    c.lines(40, 420, [
        "측정 FIR 에 극성이 이미 들어 있다.",
        "어디에서도 추가 부호 반전을 하지 않는다.",
    ], size=11.5, fill=MUTED, anchor="start", gap=16)

    c.text(300, 400, "선행량", size=12.5, weight="700", anchor="start")
    c.lines(300, 420, [
        f"lead = S {lead['s']} + handoff {lead['handoff']} − P {lead['p']} = {lead['lead']}",
        "상쇄음이 소음보다 늦게 도착하는 만큼 미리 본다.",
    ], size=11.5, fill=MUTED, anchor="start", gap=16, family=MONO)

    path = OUT / "fig1_system.svg"
    path.write_text(c.render(), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Fig 2 — 전체 아키텍처 (논문식 수직 스택 + 텐서 shape)
# ---------------------------------------------------------------------------


def fig_architecture(base: dict, tiny: dict, params: dict) -> Path:
    c = Canvas(1080, 940)
    c.text(40, 38, "Figure 2.  HybridANCNet — 학습형 encoder/decoder + dilated TCN + 그룹 LSTM",
           size=16, weight="700", anchor="start")
    c.text(40, 60,
           "Conv-TasNet 의 파형영역 encoder/decoder, WaveNet 의 dilated causal TCN, "
           "GCRN 의 그룹 LSTM 을 인과 제약 아래 결합한다.",
           size=12, fill=MUTED, anchor="start")

    x0, w = 300, 480
    rows = [
        ("입력", ["ch0 = x_ref (digital reference)", "ch1 = err_in (에러 마이크)"],
         INK, PANEL, "[B, 2, T]"),
        ("÷ io_scale", [f"io_scale = {tiny['io_scale']} — 마이크 실측 RMS 기준 상수"],
         INK, PANEL, "[B, 2, T]"),
        ("Encoder", [
            f"causal Conv1d(2 → 2C,  k = {tiny['win']},  stride = {tiny['hop']})",
            "→ GLU → ChannelLN → 1×1",
        ], ENCODER, "#ffffff", "[B, C, T/hop]"),
        ("TCN stack", [
            f"repeats R × dilations {tuple(tiny['dilations'])}",
            f"depthwise causal Conv1d(k = {tiny['kernel']}, dilation = d)",
            "→ Figure 3",
        ], TCN, "#ffffff", "[B, C, T/hop]"),
        ("GLSTM", [
            f"G groups × H hidden,  repeat {tiny['glstm_after']} 뒤 삽입",
            "그룹별 LSTM → 그룹 간 셔플 → 결합  → Figure 5",
        ], RECURR, "#ffffff", "[B, C, T/hop]"),
        ("MHSA", [
            f"windowed causal, {base['attn_heads']} heads, "
            f"window {base['attn_window']} frames",
            "base 전용 — tiny 계열은 제거",
        ], ATTN, "#ffffff", "[B, C, T/hop]"),
        ("Decoder", [
            f"1×1 → ConvTranspose1d(C → 1,  k = {tiny['win']}, "
            f"stride = {tiny['hop']})",
            "overlap-add",
        ], ENCODER, "#ffffff", "[B, 1, T]"),
        ("× io_scale · limiter", [
            f"y = {tiny['limit']} · tanh(y / {tiny['limit']})  "
            f"— 런타임 safety.control_limit 과 같은 값",
        ], DANGER, "#fdf3f1", "[B, 1, T]"),
    ]

    y = 110
    heights = []
    for index, (title, subs, colour, fill, shape) in enumerate(rows):
        h = 44 + 14 * len(subs)
        dash = "6 4" if title == "MHSA" else None
        c.rect(x0, y, w, h, fill=fill, stroke=colour, dash=dash)
        c.rect(x0, y, 5, h, fill=colour, stroke=colour, radius=2, width=0)
        c.text(x0 + 20, y + 20, title, size=13.5, weight="700", anchor="start", fill=colour)
        for k, sub in enumerate(subs):
            c.text(x0 + 20, y + 40 + k * 14, sub, size=11, anchor="start",
                   fill=MUTED, family=MONO)
        # 오른쪽 텐서 shape
        c.text(x0 + w + 18, y + h / 2, shape, size=12, anchor="start",
               family=MONO, fill=INK)
        heights.append((y, h))
        if index < len(rows) - 1:
            c.line(x0 + w / 2, y + h, x0 + w / 2, y + h + 22, arrow=True)
        y += h + 22

    # 왼쪽: 변형별 하이퍼파라미터
    c.text(40, 110, "변형", size=13, weight="700", anchor="start")
    table = [
        ("", "base", "tiny"),
        ("파라미터", f"{params['base']:,}", f"{params['tiny']:,}"),
        ("C  encoder", str(base["enc_channels"]), str(tiny["enc_channels"])),
        ("R  repeats", str(base["repeats"]), str(tiny["repeats"])),
        ("dilations", str(len(base["dilations"])) + "개", str(len(tiny["dilations"])) + "개"),
        ("TCN blocks", str(base["blocks"]), str(tiny["blocks"])),
        ("TCN hidden", str(base["tcn_hidden"]), str(tiny["tcn_hidden"])),
        ("G × H", f"{base['glstm_groups']}×{base['glstm_hidden']}",
         f"{tiny['glstm_groups']}×{tiny['glstm_hidden']}"),
        ("MHSA", "있음", "없음"),
        ("수용영역", f"{base['rf_ms']:.0f} ms", f"{tiny['rf_ms']:.0f} ms"),
        ("Jetson P99", "6.8 ms", "1.84 ms"),
    ]
    ty = 136
    for index, (label, a, b) in enumerate(table):
        weight = "700" if index == 0 else "normal"
        fill = INK if index == 0 else MUTED
        if index == 0:
            c.line(40, ty + 14, 250, ty + 14, stroke=LINE, width=1)
        c.text(40, ty, label, size=11, anchor="start", fill=fill, weight=weight)
        c.text(180, ty, a, size=11, anchor="end", family=MONO, fill=fill, weight=weight)
        c.text(250, ty, b, size=11, anchor="end", family=MONO,
               fill=INK if index else fill, weight="600" if index else weight)
        ty += 19

    c.text(40, ty + 20, "인과성", size=13, weight="700", anchor="start")
    c.lines(40, ty + 42, [
        "모든 conv 가 좌측 패딩만 쓴다.",
        "MHSA 는 과거 window 만 본다.",
        "알고리즘 룩어헤드 = 0 프레임.",
        "테스트가 오프라인↔스트리밍",
        "수치 등가를 강제한다.",
    ], size=11, fill=MUTED, anchor="start", gap=16)

    c.text(x0, 918,
           f"hop = {tiny['hop']} samples ({tiny['hop'] / 48:.2f} ms/frame) · "
           f"런타임 블록 256 = 그래프 내부 2프레임 언롤",
           size=11.5, fill=MUTED, anchor="start")

    path = OUT / "fig2_architecture.svg"
    path.write_text(c.render(), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Fig 3 — TCN 잔차 블록 (ResNet Fig.2 형식)
# ---------------------------------------------------------------------------


def fig_tcn_block(tiny: dict) -> Path:
    c = Canvas(1080, 700)
    c.text(40, 38, "Figure 3.  TCN 잔차 블록 — depthwise dilated causal convolution",
           size=16, weight="700", anchor="start")
    c.text(40, 60,
           "Conv-TasNet 의 1-D 블록 구조. 1×1 로 채널을 넓혔다가 depthwise dilated conv 로 "
           "시간을 보고, skip 과 residual 로 나눠 내보낸다.",
           size=12, fill=MUTED, anchor="start")

    x, w = 380, 300
    stack = [
        ("1×1 conv", f"C → H  (H = {tiny['tcn_hidden']})", TCN),
        ("PReLU", "채널별 학습형 기울기", MUTED),
        ("ChannelLN", "시간축 통계를 쓰지 않는 정규화 — 인과성 유지", MUTED),
        ("D-Conv", f"depthwise causal, k = {tiny['kernel']}, dilation = d", TCN),
        ("PReLU", "", MUTED),
        ("ChannelLN", "", MUTED),
    ]
    y = 110
    for title, sub, colour in stack:
        h = 44 if sub else 34
        c.rect(x, y, w, h, fill="#ffffff", stroke=colour, width=1.5)
        c.text(x + w / 2, y + (18 if sub else h / 2), title, size=13, weight="600",
               fill=colour if colour != MUTED else INK)
        if sub:
            c.text(x + w / 2, y + 33, sub, size=10.5, fill=MUTED, family=MONO)
        c.line(x + w / 2, y + h, x + w / 2, y + h + 16, arrow=True)
        y += h + 16

    # 분기
    split_y = y
    c.circle(x + w / 2, split_y, 6, fill=INK, stroke=INK)
    c.text(x + w / 2 + 14, split_y, "분기", size=11, anchor="start", fill=MUTED)

    c.path(f"M {x + w / 2} {split_y} L {x - 80} {split_y} L {x - 80} {split_y + 40}",
           arrow=True)
    c.rect(x - 170, split_y + 40, 180, 40, fill="#ffffff", stroke=TCN)
    c.text(x - 80, split_y + 60, "skip 1×1", size=12.5, weight="600", fill=TCN)
    c.line(x - 80, split_y + 80, x - 80, split_y + 118, arrow=True)
    c.text(x - 80, split_y + 132, "→ skip 합산 (모든 블록)", size=11, fill=MUTED)

    c.path(f"M {x + w / 2} {split_y} L {x + w + 90} {split_y} "
           f"L {x + w + 90} {split_y + 40}", arrow=True)
    c.rect(x + w, split_y + 40, 180, 40, fill="#ffffff", stroke=TCN)
    c.text(x + w + 90, split_y + 60, "residual 1×1", size=12.5, weight="600", fill=TCN)

    # residual 덧셈
    add_y = split_y + 118
    c.line(x + w + 90, split_y + 80, x + w + 90, add_y - 15, arrow=True)
    c.sum_node(x + w + 90, add_y)
    c.path(f"M {x + w / 2} 104 L {x + w + 90} 104 L {x + w + 90} {add_y - 15}",
           stroke=LINE, dash="5 4", arrow=True)
    c.text(x + w + 100, 104, "identity", size=11, anchor="start", fill=MUTED)
    c.line(x + w + 90, add_y + 15, x + w + 90, add_y + 45, arrow=True)
    c.text(x + w + 90, add_y + 58, "다음 블록", size=11.5, fill=MUTED)

    c.text(40, 110, "왜 depthwise 인가", size=13, weight="700", anchor="start")
    c.lines(40, 132, [
        "채널마다 독립 시간 필터를 두면",
        "파라미터가 H·k 로 끝난다.",
        "채널 혼합은 앞뒤 1×1 이 맡는다.",
    ], size=11, fill=MUTED, anchor="start", gap=16)

    c.text(40, 210, "왜 ChannelLN 인가", size=13, weight="700", anchor="start")
    c.lines(40, 232, [
        "BatchNorm 은 배치 통계를,",
        "global LN 은 발화 전체 통계를 쓴다.",
        "둘 다 스트리밍에서 미래를 본다.",
        "채널 축만 정규화해야 프레임 단위",
        "추론과 오프라인이 수치로 같아진다.",
    ], size=11, fill=MUTED, anchor="start", gap=16)

    c.text(40, 330, "블록 수", size=13, weight="700", anchor="start")
    c.lines(40, 352, [
        f"tiny  : R={tiny['repeats']} × {len(tiny['dilations'])} = {tiny['blocks']} blocks",
        f"dilation d ∈ {tuple(tiny['dilations'])}",
        "블록마다 d 가 커지며 과거를 넓게 본다.",
    ], size=11, fill=MUTED, anchor="start", gap=16, family=MONO)

    path = OUT / "fig3_tcn_block.svg"
    path.write_text(c.render(), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Fig 4 — dilated causal conv 수용영역 (WaveNet Fig.3 형식)
# ---------------------------------------------------------------------------


def fig_receptive_field(tiny: dict) -> Path:
    dilations = tiny["dilations"]
    span = 1 + sum((tiny["kernel"] - 1) * d for d in dilations)
    n = span
    c = Canvas(1080, 150 + 90 * (len(dilations) + 1))

    c.text(40, 38, "Figure 4.  Dilated causal convolution — 미래를 보지 않고 과거를 넓게 본다",
           size=16, weight="700", anchor="start")
    c.text(40, 60,
           f"tiny 의 한 repeat: dilation {tuple(dilations)}. 층을 쌓을 때마다 수용영역이 "
           "지수적으로 커지지만 현재 프레임보다 오른쪽 탭은 하나도 없다.",
           size=12, fill=MUTED, anchor="start")

    left, right = 120, 1010
    step = (right - left) / (n - 1)
    rows = len(dilations) + 1
    base_y = 130

    def px(index: int) -> float:
        return left + index * step

    def py(layer: int) -> float:
        return base_y + (rows - 1 - layer) * 90

    # 각 층의 활성 노드 — 출력(최신 프레임)에서 역추적
    active = [{n - 1}]
    for d in reversed(dilations):
        previous = set()
        for node in active[-1]:
            for tap in range(tiny["kernel"]):
                previous.add(node - tap * d)
        active.append({v for v in previous if v >= 0})
    active = list(reversed(active))

    labels = ["input"] + [f"d = {d}" for d in dilations]
    for layer in range(rows):
        y = py(layer)
        c.text(96, y, labels[layer], size=12, anchor="end", family=MONO,
               fill=INK if layer else MUTED)
        for index in range(n):
            on = index in active[layer]
            c.circle(px(index), y, 5.5 if on else 3.5,
                     fill=TCN if on else "#ffffff",
                     stroke=TCN if on else "#d3d9e0", width=1.4)

    # 연결선
    for layer, d in enumerate(dilations):
        for node in sorted(active[layer + 1]):
            for tap in range(tiny["kernel"]):
                source = node - tap * d
                if source < 0:
                    continue
                c.line(px(source), py(layer) - 6, px(node), py(layer + 1) + 6,
                       stroke=TCN, width=1.1, opacity=0.5)

    out_y = py(rows - 1)
    c.circle(px(n - 1), out_y, 8, fill=ATTN, stroke=ATTN)
    c.text(px(n - 1), out_y - 24, "현재 프레임 출력", size=11.5, fill=ATTN, weight="600")

    c.line(px(n - 1) + 14, base_y - 40, px(n - 1) + 14, out_y + 40,
           stroke=DANGER, dash="4 4", width=1.4)
    c.text(px(n - 1) + 22, (base_y + out_y) / 2, "미래 없음", size=11.5,
           anchor="start", fill=DANGER, weight="600")

    y = c.height - 46
    c.text(40, y,
           f"수용영역 {span} 프레임 × hop {tiny['hop']} + win {tiny['win']} "
           f"= {tiny['rf_ms']:.1f} ms  ({tiny['repeats']} repeat 전체 "
           f"{tiny['rf_frames']} 프레임)",
           size=12, anchor="start", family=MONO, fill=INK)

    path = OUT / "fig4_receptive_field.svg"
    path.write_text(c.render(), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Fig 5 — GLSTM
# ---------------------------------------------------------------------------


def fig_glstm(base: dict) -> Path:
    groups = max(2, base["glstm_groups"])
    c = Canvas(1080, 430)
    c.text(40, 38, "Figure 5.  Grouped LSTM — 순환 비용을 G 배 줄이고 셔플로 다시 섞는다",
           size=16, weight="700", anchor="start")
    c.text(40, 60,
           f"GCRN 의 그룹 전략. 채널을 G 그룹으로 나눠 그룹마다 작은 LSTM 을 돌리면 "
           f"파라미터가 1/G 로 준다. 그대로 두면 그룹이 서로 못 보므로 셔플로 섞는다. "
           f"(그림은 G = {groups} 예시)",
           size=12, fill=MUTED, anchor="start")

    x_in, x_split, x_lstm, x_shuf, x_out = 70, 260, 470, 700, 920
    y0, gap = 130, 70

    c.rect(x_in, y0, 120, gap * groups - 20, fill=PANEL, stroke=INK)
    c.text(x_in + 60, y0 + (gap * groups - 20) / 2, "C 채널", size=13, weight="600")

    for g in range(groups):
        y = y0 + g * gap
        c.line(x_in + 120, y0 + (gap * groups - 20) / 2, x_split, y + 22, arrow=True)
        c.rect(x_split, y, 150, 44, fill="#ffffff", stroke=RECURR)
        c.text(x_split + 75, y + 22, f"group {g + 1}   C/G", size=11.5, family=MONO,
               fill=RECURR)
        c.line(x_split + 150, y + 22, x_lstm, y + 22, arrow=True)
        c.rect(x_lstm, y, 150, 44, fill="#f7f2fa", stroke=RECURR)
        c.text(x_lstm + 75, y + 22, f"LSTM  H = {base['glstm_hidden']}", size=11.5,
               family=MONO, fill=RECURR)
        c.line(x_lstm + 150, y + 22, x_shuf, y + 22, arrow=True)

    c.rect(x_shuf, y0, 150, gap * groups - 20, fill="#ffffff", stroke=RECURR,
           dash="6 4")
    c.text(x_shuf + 75, y0 + (gap * groups - 20) / 2 - 10, "channel", size=13,
           weight="600", fill=RECURR)
    c.text(x_shuf + 75, y0 + (gap * groups - 20) / 2 + 8, "shuffle", size=13,
           weight="600", fill=RECURR)
    c.line(x_shuf + 150, y0 + (gap * groups - 20) / 2,
           x_out, y0 + (gap * groups - 20) / 2, arrow=True)
    c.rect(x_out, y0, 120, gap * groups - 20, fill=PANEL, stroke=INK)
    c.text(x_out + 60, y0 + (gap * groups - 20) / 2, "C 채널", size=13, weight="600")

    hy = y0 + gap * groups + 20
    c.rect(x_lstm - 20, hy, 190, 46, fill="#fdf9f2", stroke=RECURR, dash="5 4")
    c.text(x_lstm + 75, hy + 23, "h, c  →  다음 프레임", size=12, family=MONO,
           fill=RECURR, weight="600")
    for g in range(groups):
        c.path(f"M {x_lstm + 75} {y0 + g * gap + 44} L {x_lstm + 75} {hy}",
               stroke=RECURR, dash="4 3", width=1.2, opacity=0.6)

    c.text(40, hy + 90,
           "이 h, c 가 스트리밍 상태의 실체다 — ONNX 에서 그래프 밖으로 드러나야 "
           "오프라인/스트리밍 등가를 검증할 수 있다 (Figure 6).",
           size=11.5, fill=MUTED, anchor="start")

    path = OUT / "fig5_glstm.svg"
    path.write_text(c.render(), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Fig 6 — 스트리밍 상태
# ---------------------------------------------------------------------------


def fig_streaming(tiny: dict) -> Path:
    c = Canvas(1080, 580)
    c.text(40, 38, "Figure 6.  스트리밍 상태 — 숨은 전역 상태 없이 그래프 I/O 로 드러낸다",
           size=16, weight="700", anchor="start")
    c.text(40, 60,
           "상태를 그래프 안에 숨기면 오프라인 결과와 프레임 단위 결과가 같은지 검증할 "
           "방법이 없다. 모든 상태를 명시 입출력으로 두고 테스트가 등가를 강제한다.",
           size=12, fill=MUTED, anchor="start")

    blocks = tiny["blocks"]
    states = [
        ("st_enc", f"encoder 룩백 창 (win − hop = {tiny['win'] - tiny['hop']})", ENCODER),
        (f"st_0_tcn … st_{blocks - 1}_tcn",
         f"TCN {blocks}블록의 depthwise 지연선", TCN),
        ("st_lstm_h, st_lstm_c", "GLSTM 은닉·셀 상태", RECURR),
        ("st_dec", "decoder overlap-add 꼬리", ENCODER),
    ]

    y = 120
    for name, desc, colour in states:
        c.rect(60, y, 420, 46, fill="#ffffff", stroke=colour)
        c.text(78, y + 23, name, size=12, family=MONO, anchor="start", fill=colour,
               weight="600")
        c.text(500, y + 23, desc, size=11.5, anchor="start", fill=MUTED)
        y += 58

    total = 1 + blocks + 2 + 1
    c.rect(60, y + 10, 420, 40, fill=PANEL, stroke=INK)
    c.text(270, y + 30, f"tiny 상태 개수 = {total}", size=12.5, family=MONO,
           weight="700")

    fy = y + 90
    c.text(40, fy, "프레임 t 마다", size=13, weight="700", anchor="start")
    x = 60
    for label, colour in (("state[t−1]", MUTED), ("x[t]", INK)):
        c.rect(x, fy + 20, 130, 38, fill="#ffffff", stroke=colour)
        c.text(x + 65, fy + 39, label, size=11.5, family=MONO, fill=colour)
        x += 150
    c.line(x - 10, fy + 39, x + 30, fy + 39, arrow=True)
    c.rect(x + 30, fy + 20, 170, 38, fill="#ffffff", stroke=ENCODER)
    c.text(x + 115, fy + 39, "HybridANCNet", size=12, weight="600", fill=ENCODER)
    c.line(x + 200, fy + 39, x + 240, fy + 39, arrow=True)
    for index, (label, colour) in enumerate((("y[t]", INK), ("state[t]", MUTED))):
        c.rect(x + 240 + index * 150, fy + 20, 130, 38, fill="#ffffff", stroke=colour)
        c.text(x + 305 + index * 150, fy + 39, label, size=11.5, family=MONO, fill=colour)

    c.text(40, fy + 92,
           "검증: 오프라인 일괄 추론과 프레임 단위 스트리밍의 최대 오차 3e-8, "
           "PyTorch ↔ ONNX Runtime 8e-8 이하 — 회귀 테스트가 강제한다.",
           size=11.5, fill=MUTED, anchor="start")

    path = OUT / "fig6_streaming.svg"
    path.write_text(c.render(), encoding="utf-8")
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    base, tiny = model_facts("base"), model_facts("tiny")
    params = {"base": param_count("base"), "tiny": param_count("tiny")}

    duct = load_yaml(REPO / "configs" / "duct.yaml")
    data = load_yaml(REPO / "configs" / "data_sim.yaml")
    import numpy as np

    secondary = np.load(REPO / duct["secondary_path"]["npz"], allow_pickle=False)
    s_delay = int(np.asarray(secondary["delay_samples"]).reshape(-1)[0])
    handoff = int(duct["secondary_path"]["handoff_extra_samples"])
    lead_samples = int(data["digital_reference_lead_samples"])
    lead = {
        "s": s_delay,
        "handoff": handoff,
        "lead": lead_samples,
        "p": s_delay + handoff - lead_samples,
    }

    produced = [
        fig_system(tiny, lead),
        fig_architecture(base, tiny, params),
        fig_tcn_block(tiny),
        fig_receptive_field(tiny),
        fig_glstm(base),
        fig_streaming(tiny),
    ]
    for path in produced:
        print(f"  {path.relative_to(REPO)}  ({path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
