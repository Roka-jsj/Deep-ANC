"""ANC 학습 손실 — 미분가능 플랜트를 통과한 에러신호를 직접 최소화.

    y = model(x)  →  y_nl = G_nl(y)  →  e = d + S(y_nl)
    L = L_nmse(dB) + λ1·L_mrstft×W(f) + λ2·L_pow + λ3·L_clip

- L_nmse: 평가지표(감쇠 dB)를 그대로 최소화 (10·log10 Σe²/Σd²)
- L_mrstft: 다중해상도 STFT 크기 손실, 주파수 가중 W(f)
  · curriculum_a: 평면파 대역(80–1000Hz) ×3 — 컷오프(1633Hz) 이상은 0.25
    [설계 교차검증 C3 — 광대역 S(z) 재보정 전에는 fullband 커리큘럼 금지]
- L_pow / L_clip: 과출력·클리핑 억제 (런타임 control limit 0.2 마진 0.18)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..dsp.nonlinear import RandomNonlinear
from ..dsp.secondary_path import DifferentiableSecondaryPath

_EPS = 1.0e-10


def band_weights(
    fft_size: int,
    sample_rate: int,
    scheme: str,
    cutoff_hz: float = 1633.0,
) -> torch.Tensor:
    """rfft 빈별 가중 벡터."""
    freqs = torch.fft.rfftfreq(fft_size) * sample_rate
    w = torch.ones_like(freqs)
    if scheme == "curriculum_a":
        w = torch.where((freqs >= 80.0) & (freqs <= 1000.0), torch.full_like(w, 3.0), w)
        w = torch.where(freqs > cutoff_hz, torch.full_like(w, 0.25), w)
        w = torch.where(freqs < 40.0, torch.full_like(w, 0.1), w)
    elif scheme == "fullband":
        w = torch.where(freqs < 20.0, torch.full_like(w, 0.1), w)
    else:
        raise ValueError(f"알 수 없는 band_weight: {scheme}")
    return w


def _stft_mag(x: torch.Tensor, fft_size: int, hop: int, window: torch.Tensor) -> torch.Tensor:
    """x: [B, T] → |STFT| [B, F, frames]."""
    spec = torch.stft(
        x,
        n_fft=fft_size,
        hop_length=hop,
        win_length=fft_size,
        window=window,
        center=True,
        return_complex=True,
    )
    return spec.abs()


class ANCLoss(nn.Module):
    def __init__(
        self,
        plant: DifferentiableSecondaryPath,
        loss_cfg: dict,
        sample_rate: int,
        nonlinear: RandomNonlinear | None = None,
        cutoff_hz: float = 1633.0,
    ) -> None:
        super().__init__()
        self.plant = plant
        self.nonlinear = nonlinear
        self.sample_rate = sample_rate
        self.ffts = [int(v) for v in loss_cfg.get("mrstft_ffts", [256, 512, 1024, 2048])]
        self.lambda_mrstft = float(loss_cfg.get("lambda_mrstft", 1.0))
        self.lambda_pow = float(loss_cfg.get("lambda_pow", 1.0e-3))
        self.lambda_clip = float(loss_cfg.get("lambda_clip", 1.0))
        self.clip_margin = float(loss_cfg.get("clip_margin", 0.18))
        scheme = str(loss_cfg.get("band_weight", "curriculum_a"))
        for fft_size in self.ffts:
            self.register_buffer(
                f"w_{fft_size}",
                band_weights(fft_size, sample_rate, scheme, cutoff_hz),
                persistent=False,
            )
            self.register_buffer(
                f"win_{fft_size}", torch.hann_window(fft_size), persistent=False
            )

    def forward(self, y: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """y, d: [B, 1, T] (물리 스케일). 반환: (total_loss, metrics)."""
        batch = y.shape[0]

        y_nl = y
        if self.nonlinear is not None:
            params = self.nonlinear.sample(batch)
            y_nl = self.nonlinear.apply_torch(y, params)

        perturb = self.plant.sample_perturbation() if self.training else {"jitter": 0}
        e = d + self.plant(y_nl, perturb)

        e_flat = e.squeeze(1)
        d_flat = d.squeeze(1)

        # NMSE (dB) — 감쇠량 지표 직접 최소화
        e_pow = e_flat.pow(2).sum(dim=-1)
        d_pow = d_flat.pow(2).sum(dim=-1)
        nmse_db = 10.0 * torch.log10((e_pow + _EPS) / (d_pow + _EPS))
        l_nmse = nmse_db.mean()

        # 다중해상도 STFT (주파수 가중) — e 의 스펙트럼 에너지를 d 대비로 정규화
        l_mrstft = y.new_zeros(())
        for fft_size in self.ffts:
            w = getattr(self, f"w_{fft_size}").view(1, -1, 1)
            win = getattr(self, f"win_{fft_size}")
            E = _stft_mag(e_flat, fft_size, fft_size // 4, win) * w
            D = _stft_mag(d_flat, fft_size, fft_size // 4, win) * w
            sc = torch.linalg.norm(E) / (torch.linalg.norm(D) + _EPS)
            l1 = E.sum() / (D.sum() + _EPS)
            l_mrstft = l_mrstft + sc + l1
        l_mrstft = l_mrstft / len(self.ffts)

        l_pow = y.pow(2).mean()
        l_clip = F.relu(y.abs() - self.clip_margin).pow(2).mean()

        total = (
            l_nmse
            + self.lambda_mrstft * l_mrstft
            + self.lambda_pow * l_pow
            + self.lambda_clip * l_clip
        )
        metrics = {
            "loss": float(total.detach()),
            "nmse_db": float(l_nmse.detach()),
            "mrstft": float(l_mrstft.detach()),
            "out_pow": float(l_pow.detach()),
            "clip": float(l_clip.detach()),
        }
        return total, metrics
