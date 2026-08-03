"""ANC 학습 손실 — 미분가능 플랜트를 통과한 에러신호를 직접 최소화.

    y = model(x)  →  y_nl = G_nl(y)  →  e = d + S(y_nl)
    L = L_nmse,trusted(dB) + λ1·L_mrstft×W(f) + λ2·L_pow + λ3·L_clip

- L_nmse,trusted: 실측 S(z) 유효대역 ∩ 덕트 목표대역의 NMSE를 최소화.
  전대역 NMSE는 do-no-harm 관측용 지표로 동시에 남긴다.
- L_mrstft: 다중해상도 STFT 크기 손실, 주파수 가중 W(f)
  · curriculum_a: 평면파 대역(80–1000Hz) ×3 — 컷오프(1633Hz) 이상은 0.25
    [설계 교차검증 C3 — 광대역 S(z) 재보정 전에는 fullband 커리큘럼 금지]
- L_pow / L_clip: 과출력·클리핑 억제 (런타임 control limit 0.2 마진 0.18)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..dsp.nonlinear import RandomNonlinear
from ..dsp.secondary_path import DifferentiableSecondaryPath

_EPS = 1.0e-10


def intersect_frequency_bands(
    first: tuple[float, float],
    second: tuple[float, float],
    nyquist_hz: float,
) -> tuple[float, float]:
    """두 주파수 대역의 교집을 반환한다.

    학습 NMSE에 제어 불가능하거나 S(z) 실측이 신뢰되지 않는 대역이
    조용히 섞이지 않도록, 잘못된 범위와 빈 교집은 fail-fast 한다.
    """
    a_lo, a_hi = (float(v) for v in first)
    b_lo, b_hi = (float(v) for v in second)
    nyquist = float(nyquist_hz)
    values = (a_lo, a_hi, b_lo, b_hi, nyquist)
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"유한하지 않은 주파수 대역: {first}, {second}, Nyquist={nyquist}")
    if nyquist <= 0.0:
        raise ValueError(f"Nyquist 주파수는 양수여야 합니다: {nyquist}")
    if not (0.0 <= a_lo < a_hi <= nyquist):
        raise ValueError(f"잘못된 주파수 대역: {first} (Nyquist={nyquist})")
    if not (0.0 <= b_lo < b_hi <= nyquist):
        raise ValueError(f"잘못된 주파수 대역: {second} (Nyquist={nyquist})")

    lo, hi = max(a_lo, b_lo), min(a_hi, b_hi)
    if lo >= hi:
        raise ValueError(
            f"주파수 대역 교집이 비어 있습니다: {first} ∩ {second}"
        )
    return lo, hi


def band_weights(
    fft_size: int,
    sample_rate: int,
    scheme: str,
    cutoff_hz: float = 1633.0,
    target_band_hz: tuple[float, float] = (80.0, 1000.0),
) -> torch.Tensor:
    """rfft 빈별 가중 벡터. 목표 대역은 duct.yaml acoustics.realistic_target_band_hz
    가 단일 출처다 (감사 L9 — trainer 가 주입)."""
    freqs = torch.fft.rfftfreq(fft_size) * sample_rate
    w = torch.ones_like(freqs)
    lo, hi = float(target_band_hz[0]), float(target_band_hz[1])
    if scheme == "curriculum_a":
        w = torch.where((freqs >= lo) & (freqs <= hi), torch.full_like(w, 3.0), w)
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
        target_band_hz: tuple[float, float] = (80.0, 1000.0),
        trusted_band_hz: tuple[float, float] | None = None,
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
        # 직접 ANCLoss를 생성하는 기존 코드는 trusted band를 넘기지 않으므로
        # 기존 fullband 동작을 유지한다. Trainer는 항상 실측∩목표 대역을 주입한다.
        default_objective = "trusted_band" if trusted_band_hz is not None else "fullband"
        self.nmse_objective = str(loss_cfg.get("nmse_objective", default_objective))
        if self.nmse_objective not in {"trusted_band", "fullband"}:
            raise ValueError(
                f"지원하지 않는 loss.nmse_objective: {self.nmse_objective}"
            )
        if trusted_band_hz is None:
            if self.nmse_objective == "trusted_band":
                raise ValueError("nmse_objective=trusted_band 이면 trusted_band_hz가 필요합니다")
            self.trusted_band_hz = None
        else:
            self.trusted_band_hz = intersect_frequency_bands(
                trusted_band_hz,
                trusted_band_hz,
                sample_rate / 2.0,
            )
        scheme = str(loss_cfg.get("band_weight", "curriculum_a"))
        for fft_size in self.ffts:
            self.register_buffer(
                f"w_{fft_size}",
                band_weights(fft_size, sample_rate, scheme, cutoff_hz, target_band_hz),
                persistent=False,
            )
            self.register_buffer(
                f"win_{fft_size}", torch.hann_window(fft_size), persistent=False
            )

    def _band_nmse_db(
        self,
        e: torch.Tensor,
        d: torch.Tensor,
        band_hz: tuple[float, float],
    ) -> torch.Tensor:
        """e/d [B,T]의 주어진 대역 NMSE [B]를 미분가능하게 계산."""
        samples = e.shape[-1]
        if samples < 2:
            raise ValueError(f"NMSE FFT에 필요한 샘플이 부족합니다: {samples}")
        lo, hi = band_hz
        lo_bin = max(0, int(math.ceil(lo * samples / self.sample_rate)))
        hi_bin = min(samples // 2, int(math.floor(hi * samples / self.sample_rate)))
        if lo_bin > hi_bin:
            raise ValueError(
                f"세그먼트 {samples}샘플 FFT에 trusted band {band_hz} bin이 없습니다"
            )

        E = torch.fft.rfft(e, dim=-1, norm="ortho")[..., lo_bin : hi_bin + 1]
        D = torch.fft.rfft(d, dim=-1, norm="ortho")[..., lo_bin : hi_bin + 1]
        e_pow = E.real.square() + E.imag.square()
        d_pow = D.real.square() + D.imag.square()

        # one-sided FFT Parseval 가중치. DC/Nyquist 외 bin은 음수 주파수와
        # 짝이 있으므로 2배한다. 대역 비율의 물리적 에너지를 유지한다.
        weights = torch.full(
            (hi_bin - lo_bin + 1,),
            2.0,
            dtype=e_pow.dtype,
            device=e.device,
        )
        if lo_bin == 0:
            weights[0] = 1.0
        if samples % 2 == 0 and hi_bin == samples // 2:
            weights[-1] = 1.0
        e_band_pow = (e_pow * weights).sum(dim=-1)
        d_band_pow = (d_pow * weights).sum(dim=-1)
        return 10.0 * torch.log10((e_band_pow + _EPS) / (d_band_pow + _EPS))

    def forward(
        self,
        y: torch.Tensor,
        d: torch.Tensor,
        loss_start_sample: int = 0,
        perturb: dict | None = None,
        nl_params: dict | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """y, d: [B, 1, T] (물리 스케일). 반환: (total_loss, metrics).

        - loss_start_sample: 플랜트를 전체 길이에 적용한 **뒤** NMSE/MR-STFT 에서 제외할
          앞구간(폐루프 워밍업). 잘린 y 에 플랜트를 걸면 경계 뒤 ~(지연+FIR) 구간의
          S·y 기여가 사라지므로 반드시 여기서 잘라야 한다 (리뷰 확정 결함 #2/#5).
        - perturb/nl_params: 폐루프 학습에서 되먹임 경로와 동일한 플랜트/비선형을
          쓰기 위한 외부 주입 (리뷰 확정 결함 #6). None 이면 내부 샘플링.
        - FFT(플랜트·STFT)는 bf16 미지원 → 손실 전체 FP32 고정.
        """
        if y.is_cuda:
            autocast_off = torch.autocast("cuda", enabled=False)
        else:
            import contextlib

            autocast_off = contextlib.nullcontext()
        with autocast_off:
            return self._forward_fp32(
                y.float(), d.float(), loss_start_sample, perturb, nl_params
            )

    def _forward_fp32(
        self,
        y: torch.Tensor,
        d: torch.Tensor,
        loss_start_sample: int = 0,
        perturb: dict | None = None,
        nl_params: dict | None = None,
    ) -> tuple[torch.Tensor, dict]:
        batch = y.shape[0]

        # 평가(eval) 모드에서는 비선형/섭동 없이 결정적으로 계산 — val 지표 일관성
        y_nl = y
        if self.training and self.nonlinear is not None:
            if nl_params is None:
                nl_params = self.nonlinear.sample(batch)
            y_nl = self.nonlinear.apply_torch(y, nl_params)

        if perturb is None:
            perturb = self.plant.sample_perturbation() if self.training else {"jitter": 0}
        e = d + self.plant(y_nl, perturb)

        skip = int(loss_start_sample)
        e_flat = e.squeeze(1)[..., skip:]
        d_flat = d.squeeze(1)[..., skip:]

        # 전대역 NMSE (dB) — do-no-harm 관측용
        e_pow = e_flat.pow(2).sum(dim=-1)
        d_pow = d_flat.pow(2).sum(dim=-1)
        nmse_fullband_db = 10.0 * torch.log10((e_pow + _EPS) / (d_pow + _EPS))

        nmse_trusted_db: torch.Tensor | None = None
        if self.trusted_band_hz is not None:
            nmse_trusted_db = self._band_nmse_db(e_flat, d_flat, self.trusted_band_hz)

        if self.nmse_objective == "trusted_band":
            assert nmse_trusted_db is not None
            l_nmse = nmse_trusted_db.mean()
        else:
            l_nmse = nmse_fullband_db.mean()

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
            "nmse_fullband_db": float(nmse_fullband_db.mean().detach()),
            "mrstft": float(l_mrstft.detach()),
            "out_pow": float(l_pow.detach()),
            "clip": float(l_clip.detach()),
        }
        if nmse_trusted_db is not None:
            metrics["nmse_trusted_db"] = float(nmse_trusted_db.mean().detach())
        return total, metrics
