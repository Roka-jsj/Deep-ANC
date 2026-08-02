"""스트리밍 래퍼 — 실시간 엔진과 ONNX export 의 공용 진입점.

ONNX export 규약 (docs/06_deployment_jetson.md):
  opset 17, 배치 1, 모든 shape 정적, 상태는 전부 명시적 입출력 텐서,
  블록 256샘플 = 모델 hop 128 × 2프레임을 그래프 내부에서 정적 언롤,
  LSTM 은 수동 셀(MatMul/Sigmoid/Tanh), DFT/If/Loop 미사용.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .attention import WindowedCausalMHSA
from .glstm import GLSTM
from .hybrid_anc import HybridANCNet


def flatten_states(states: list) -> list[torch.Tensor]:
    flat: list[torch.Tensor] = []
    for s in states:
        if isinstance(s, tuple):
            flat.extend(s)
        else:
            flat.append(s)
    return flat


def unflatten_states(model: HybridANCNet, flat: list[torch.Tensor]) -> list:
    """flatten_states 의 역변환 — 모델 블록 구조에 맞춰 튜플 복원."""
    states: list = []
    it = iter(flat)
    states.append(next(it))                       # enc_hist
    for block in model.blocks:
        if isinstance(block, GLSTM):
            states.append((next(it), next(it)))
        elif isinstance(block, WindowedCausalMHSA):
            states.append((next(it), next(it), next(it)))
        else:
            states.append(next(it))
    states.append(next(it))                       # dec_tail
    return states


def state_names(model: HybridANCNet) -> list[str]:
    """export 입출력 텐서 이름 (순서 = flatten_states)."""
    names = ["st_enc"]
    for i, block in enumerate(model.blocks):
        if isinstance(block, GLSTM):
            names += [f"st_{i}_lstm_h", f"st_{i}_lstm_c"]
        elif isinstance(block, WindowedCausalMHSA):
            names += [f"st_{i}_attn_k", f"st_{i}_attn_v", f"st_{i}_attn_m"]
        else:
            names.append(f"st_{i}_tcn")
    names.append("st_dec")
    return names


class StreamingHybridANC:
    """PyTorch eager 스트리밍 실행기 (개발/torch 엔진용)."""

    def __init__(self, model: HybridANCNet, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        self.states = self.model.init_states(batch=1, device=self.device)

    @torch.no_grad()
    def step(self, block: np.ndarray) -> np.ndarray:
        """block: [in_ch, N] float32 → y [N] float32."""
        x = torch.from_numpy(np.ascontiguousarray(block, dtype=np.float32))
        x = x.unsqueeze(0).to(self.device)
        y, self.states = self.model.streaming_step(x, self.states)
        return y.squeeze(0).squeeze(0).cpu().numpy()


class ExportWrapper(nn.Module):
    """ONNX export 대상 — forward(x, *flat_states) -> (y, *new_flat_states)."""

    def __init__(self, model: HybridANCNet, block_samples: int = 256) -> None:
        super().__init__()
        self.model = model
        self.block_samples = int(block_samples)

    def forward(self, x: torch.Tensor, *flat_states: torch.Tensor):
        states = unflatten_states(self.model, list(flat_states))
        y, new_states = self.model.streaming_step(x, states)
        return (y, *flatten_states(new_states))
