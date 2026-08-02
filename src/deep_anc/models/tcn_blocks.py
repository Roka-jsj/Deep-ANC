"""dilated causal depthwise TCN 블록 (WaveNet 계열 + GLU 게이팅).

프레임 단위(hop 128)로 dilation을 적용해 0.5초급 수용영역을 확보한다.
모든 conv는 좌측 패딩만 사용(인과) — 미래 프레임을 보지 않는다.
스트리밍은 (kernel-1)·dilation 프레임의 좌측 히스토리를 상태로 유지한다.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ChannelLayerNorm(nn.Module):
    """[B, C, F] 텐서의 채널축 LayerNorm (프레임 독립 — 인과 안전)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class TCNBlock(nn.Module):
    """1×1 확장 → PReLU → LN → dw causal conv ×2(주경로·게이트) → 1×1 축소 + residual.

    파라미터 수(base, C=256, H=512): ≈266k — 설계 문서 §4 스펙.
    """

    def __init__(self, channels: int, hidden: int, kernel: int = 3, dilation: int = 1) -> None:
        super().__init__()
        self.channels = channels
        self.hidden = hidden
        self.kernel = kernel
        self.dilation = dilation
        self.context = (kernel - 1) * dilation  # 좌측 히스토리 프레임 수

        self.expand = nn.Conv1d(channels, hidden, 1)
        self.act = nn.PReLU()
        self.norm = ChannelLayerNorm(hidden)
        self.dw_main = nn.Conv1d(hidden, hidden, kernel, dilation=dilation, groups=hidden)
        self.dw_gate = nn.Conv1d(hidden, hidden, kernel, dilation=dilation, groups=hidden)
        self.project = nn.Conv1d(hidden, channels, 1)

    def _inner(self, y_padded: torch.Tensor) -> torch.Tensor:
        u = self.dw_main(y_padded)
        g = torch.sigmoid(self.dw_gate(y_padded))
        return self.project(u * g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """오프라인(전체 시퀀스) 경로. x: [B, C, F]."""
        y = self.norm(self.act(self.expand(x)))
        y = F.pad(y, (self.context, 0))
        return x + self._inner(y)

    def init_state(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch, self.hidden, self.context, device=device)

    def streaming_forward(
        self, x: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """스트리밍 경로. x: [B, C, F_blk], state: [B, H, context]."""
        y = self.norm(self.act(self.expand(x)))
        y_cat = torch.cat([state, y], dim=-1)
        out = x + self._inner(y_cat)
        new_state = y_cat[..., y_cat.shape[-1] - self.context :]
        return out, new_state
