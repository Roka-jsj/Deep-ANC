"""GLSTM — 그룹 LSTM 병목 (GCRN 계열).

학습(오프라인)에서는 cuDNN 융합 nn.LSTM 을 사용하고, 스트리밍/ONNX export 에서는
동일 가중치로 수동 셀 언롤을 사용한다 [설계 교차검증 H1].
두 경로의 등가성은 tests/test_model_shapes.py 에서 검증한다.
"""

from __future__ import annotations

import torch
from torch import nn


def lstm_cell_manual(
    x_t: torch.Tensor,
    h: torch.Tensor,
    c: torch.Tensor,
    w_ih: torch.Tensor,
    w_hh: torch.Tensor,
    b_ih: torch.Tensor,
    b_hh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """torch nn.LSTM 과 동일한 게이트 순서(i, f, g, o)의 셀 1스텝.

    x_t: [B, in], h/c: [B, hid] → (h', c')
    ONNX 호환: MatMul/Add/Sigmoid/Tanh/Split 만 사용.
    """
    gates = x_t @ w_ih.t() + b_ih + h @ w_hh.t() + b_hh
    i, f, g, o = gates.chunk(4, dim=-1)
    c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
    h_new = torch.sigmoid(o) * torch.tanh(c_new)
    return h_new, c_new


class GLSTM(nn.Module):
    """입력 [B, C, F]를 G그룹으로 나눠 그룹별 LSTM → concat → 1×1 proj → LN → residual."""

    def __init__(self, channels: int, groups: int, hidden_per_group: int) -> None:
        super().__init__()
        if channels % groups != 0:
            raise ValueError("channels 는 groups 로 나누어떨어져야 합니다")
        self.channels = channels
        self.groups = groups
        self.hidden_per_group = hidden_per_group
        self.in_per_group = channels // groups

        self.lstms = nn.ModuleList(
            nn.LSTM(self.in_per_group, hidden_per_group, num_layers=1, batch_first=True)
            for _ in range(groups)
        )
        self.proj = nn.Linear(groups * hidden_per_group, channels)
        self.norm = nn.LayerNorm(channels)

    # ---------- 오프라인 (nn.LSTM, cuDNN) ----------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.transpose(1, 2)                      # [B, F, C]
        chunks = seq.chunk(self.groups, dim=-1)
        outs = [lstm(chunk)[0] for lstm, chunk in zip(self.lstms, chunks)]
        y = self.norm(self.proj(torch.cat(outs, dim=-1)))
        return x + y.transpose(1, 2)

    # ---------- 스트리밍/Export (수동 셀) ----------

    def init_state(self, batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """h, c: [B, G·hid] — 명시적 상태 텐서 (정적 shape)."""
        size = self.groups * self.hidden_per_group
        zeros = torch.zeros(batch, size, device=device)
        return zeros, zeros.clone()

    def streaming_forward(
        self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: [B, C, F_blk] (F_blk 는 작은 정수 — export 시 정적 언롤됨)."""
        seq = x.transpose(1, 2)                      # [B, F_blk, C]
        frames = seq.shape[1]
        h_groups = list(h.chunk(self.groups, dim=-1))
        c_groups = list(c.chunk(self.groups, dim=-1))
        in_groups = seq.chunk(self.groups, dim=-1)   # 각 [B, F_blk, in_per_group]

        outs: list[torch.Tensor] = []
        for t in range(frames):
            frame_outs = []
            for gi, lstm in enumerate(self.lstms):
                h_new, c_new = lstm_cell_manual(
                    in_groups[gi][:, t, :],
                    h_groups[gi],
                    c_groups[gi],
                    lstm.weight_ih_l0,
                    lstm.weight_hh_l0,
                    lstm.bias_ih_l0,
                    lstm.bias_hh_l0,
                )
                h_groups[gi], c_groups[gi] = h_new, c_new
                frame_outs.append(h_new)
            outs.append(torch.cat(frame_outs, dim=-1))  # [B, G·hid]

        stacked = torch.stack(outs, dim=1)               # [B, F_blk, G·hid]
        y = self.norm(self.proj(stacked))
        out = x + y.transpose(1, 2)
        return out, torch.cat(h_groups, dim=-1), torch.cat(c_groups, dim=-1)
