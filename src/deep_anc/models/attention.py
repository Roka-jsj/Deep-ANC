"""Windowed causal MHSA — 주기성(회전기계 등) 재조회용 1층 attention.

과거 window_frames(기본 64프레임 = 170ms) 안의 KV만 참조한다.
스트리밍은 KV 링버퍼를 concat+slice 로 갱신(정적 shape, ONNX 호환) —
인과성이 구조적으로 보장되므로 마스크가 필요 없다.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class WindowedCausalMHSA(nn.Module):
    def __init__(self, channels: int, heads: int, head_dim: int, window_frames: int) -> None:
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.head_dim = head_dim
        self.window = window_frames
        inner = heads * head_dim

        self.norm = nn.LayerNorm(channels)           # pre-LN
        self.qkv = nn.Linear(channels, 3 * inner)
        self.out = nn.Linear(inner, channels)
        # 상대위치 bias: 거리 0(현재)~window-1(가장 과거)
        self.rel_bias = nn.Parameter(torch.zeros(heads, window_frames))
        self.scale = head_dim ** -0.5

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, f, _ = x.shape
        return x.view(b, f, self.heads, self.head_dim).transpose(1, 2)  # [B,H,F,D]

    # ---------- 오프라인 (전체 시퀀스, 밴드 인과 마스크) ----------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, F]."""
        b, _, frames = x.shape
        seq = self.norm(x.transpose(1, 2))
        q, k, v = self.qkv(seq).chunk(3, dim=-1)
        q, k, v = map(self._split_heads, (q, k, v))   # [B,H,F,D]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,F,F]

        idx = torch.arange(frames, device=x.device)
        dist = idx.view(-1, 1) - idx.view(1, -1)      # i - j
        in_window = (dist >= 0) & (dist < self.window)
        clamped = dist.clamp(0, self.window - 1)
        bias = self.rel_bias[:, clamped]              # [H,F,F]
        scores = scores + bias.unsqueeze(0)
        scores = scores.masked_fill(~in_window.view(1, 1, frames, frames), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v)                    # [B,H,F,D]
        ctx = ctx.transpose(1, 2).reshape(b, frames, self.heads * self.head_dim)
        return x + self.out(ctx).transpose(1, 2)

    # ---------- 스트리밍 (KV 캐시) ----------

    def init_state(
        self, batch: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(k_cache, v_cache, mask_cache) — [B,H,W,D] ×2, [B,1,1,W].

        mask_cache 는 캐시 슬롯 유효성: 빈 슬롯 -1e4 (softmax에서 제외),
        채워진 슬롯 0. 워밍업 구간에서도 오프라인 forward 와 수치 등가가 되도록 한다.
        """
        shape = (batch, self.heads, self.window, self.head_dim)
        k = torch.zeros(shape, device=device)
        v = torch.zeros(shape, device=device)
        mask = torch.full((batch, 1, 1, self.window), -1.0e4, device=device)
        return k, v, mask

    def streaming_forward(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        mask_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: [B, C, F_blk] — 프레임별 순차 처리 (export 시 정적 언롤).

        캐시 갱신을 먼저 하고 현재 프레임을 포함한 최근 window 개에 attend 하므로
        오프라인 밴드 마스크(dist∈[0, window))와 동일한 참조 범위를 갖는다.
        """
        b, _, frames = x.shape
        seq = self.norm(x.transpose(1, 2))
        q_all, k_all, v_all = self.qkv(seq).chunk(3, dim=-1)
        q_all, k_all, v_all = map(self._split_heads, (q_all, k_all, v_all))  # [B,H,F_blk,D]

        zero_slot = torch.zeros_like(mask_cache[..., :1])
        outs: list[torch.Tensor] = []
        for t in range(frames):
            k_t = k_all[:, :, t : t + 1, :]
            v_t = v_all[:, :, t : t + 1, :]
            k_cache = torch.cat([k_cache, k_t], dim=2)[:, :, 1:, :]
            v_cache = torch.cat([v_cache, v_t], dim=2)[:, :, 1:, :]
            mask_cache = torch.cat([mask_cache, zero_slot], dim=-1)[..., 1:]

            q_t = q_all[:, :, t : t + 1, :]                                  # [B,H,1,D]
            scores = torch.matmul(q_t, k_cache.transpose(-2, -1)) * self.scale  # [B,H,1,W]
            # 캐시는 [가장 과거 ... 현재] 순서 → 거리 = window-1 ... 0
            bias = torch.flip(self.rel_bias, dims=[-1]).view(1, self.heads, 1, self.window)
            attn = torch.softmax(scores + bias + mask_cache, dim=-1)
            outs.append(torch.matmul(attn, v_cache))                          # [B,H,1,D]

        ctx = torch.cat(outs, dim=2).transpose(1, 2).reshape(b, frames, self.heads * self.head_dim)
        out = x + self.out(ctx).transpose(1, 2)
        return out, k_cache, v_cache, mask_cache
