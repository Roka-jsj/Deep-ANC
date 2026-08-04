"""스피커/앰프 비선형 모델 — 학습 증강용 (numpy + torch 양쪽 구현).

Deep ANC(Zhang & Wang, 2021) 프로토콜의 SEF(Scaled Error Function)를
η·tanh(x/η) 근사로 사용한다. η가 작을수록 포화가 심하고 η=10 이면 사실상 선형.
미분가능해야 학습 그래프(losses/anc_loss.py)에 포함할 수 있다.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:  # Jetson 초기 설치 전에도 numpy 경로는 동작해야 한다
    torch = None  # type: ignore[assignment]


def sef_np(x: np.ndarray, eta: float) -> np.ndarray:
    """η·tanh(x/η) — 스피커 포화 근사."""
    eta = float(eta)
    if eta >= 10.0:
        return np.asarray(x, dtype=np.float32)
    return (eta * np.tanh(np.asarray(x, dtype=np.float64) / eta)).astype(np.float32)


def drive_np(x: np.ndarray, gain: float) -> np.ndarray:
    """tanh(g·x)/g — 게인 과구동(앰프 클리핑 근방) 근사."""
    g = float(gain)
    return (np.tanh(g * np.asarray(x, dtype=np.float64)) / g).astype(np.float32)


if torch is not None:

    def sef_torch(x: "torch.Tensor", eta: "torch.Tensor | float") -> "torch.Tensor":
        """배치별 η 지원: eta 는 스칼라 또는 [B,1,1] 텐서."""
        if not isinstance(eta, torch.Tensor):
            eta = torch.tensor(float(eta), dtype=x.dtype, device=x.device)
        # η>=10 은 선형 취급 (tanh 포화 회피)
        linear = eta >= 10.0
        safe_eta = torch.clamp(eta, min=1.0e-3)
        curved = safe_eta * torch.tanh(x / safe_eta)
        return torch.where(linear.expand_as(curved) if linear.ndim else linear, x, curved)

    def drive_torch(x: "torch.Tensor", gain: "torch.Tensor | float") -> "torch.Tensor":
        if not isinstance(gain, torch.Tensor):
            gain = torch.tensor(float(gain), dtype=x.dtype, device=x.device)
        gain = torch.clamp(gain, min=1.0e-3)
        return torch.tanh(gain * x) / gain


class RandomNonlinear:
    """학습용 랜덤 비선형 샘플러 — 배치마다 (SEF η, drive g)를 추첨."""

    def __init__(
        self,
        eta_choices: list[float],
        drive_range: tuple[float, float],
        hardclip_prob: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.eta_choices = [float(v) for v in eta_choices]
        self.drive_range = (float(drive_range[0]), float(drive_range[1]))
        self.hardclip_prob = float(hardclip_prob)
        self.rng = np.random.default_rng(seed)

    def sample(self, batch: int) -> dict:
        eta = self.rng.choice(self.eta_choices, size=batch)
        g = self.rng.uniform(*self.drive_range, size=batch)
        clip = self.rng.random(batch) < self.hardclip_prob
        return {"eta": eta.astype(np.float32), "drive": g.astype(np.float32), "hardclip": clip}

    def apply_torch(self, y: "torch.Tensor", params: dict) -> "torch.Tensor":
        """y: [B, 1, T]. params: sample() 결과."""
        assert torch is not None
        eta = torch.as_tensor(params["eta"], dtype=y.dtype, device=y.device).view(-1, 1, 1)
        g = torch.as_tensor(params["drive"], dtype=y.dtype, device=y.device).view(-1, 1, 1)
        out = drive_torch(y, g)
        out = sef_torch(out, eta)
        if params["hardclip"].any():
            mask = torch.as_tensor(
                params["hardclip"].astype(np.float32), dtype=y.dtype, device=y.device
            ).view(-1, 1, 1)
            clipped = torch.clamp(out, -0.95, 0.95)
            out = mask * clipped + (1.0 - mask) * out
        return out
