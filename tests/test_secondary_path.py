"""미분가능 S(z)가 scipy(lfilter+지연) 기준 구현과 일치하는지 검증."""

import numpy as np
import pytest
import torch
from scipy import signal

from deep_anc.config import REPO_ROOT
from deep_anc.dsp.secondary_path import (
    DifferentiableSecondaryPath,
    load_secondary_path,
)

NPZ = REPO_ROOT / "assets" / "measured" / "secondary_path_4s.npz"


@pytest.fixture(scope="module")
def sp_data():
    return load_secondary_path(NPZ)


def test_load_measured_npz(sp_data):
    assert sp_data.fir.size == 2048
    assert sp_data.delay_samples == 1342          # secondary_path_4s.npz 실측값
    assert sp_data.sample_rate == 48000
    assert sp_data.excitation_band_hz == (150.0, 600.0)


def test_torch_matches_scipy(sp_data):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(8000).astype(np.float32) * 0.1

    handoff = 256
    plant = DifferentiableSecondaryPath(sp_data, handoff_extra_samples=handoff)
    with torch.no_grad():
        y_torch = plant(torch.from_numpy(x).view(1, 1, -1)).numpy().reshape(-1)

    delayed = np.zeros_like(x)
    total = sp_data.delay_samples + handoff
    delayed[total:] = x[: x.size - total]
    y_ref = signal.lfilter(sp_data.fir, [1.0], delayed)

    assert np.max(np.abs(y_torch - y_ref)) < 1e-4


def test_gradient_flows(sp_data):
    plant = DifferentiableSecondaryPath(sp_data)
    y = (torch.randn(2, 1, 4096) * 0.05).requires_grad_()
    out = plant(y)
    out.pow(2).mean().backward()
    assert y.grad is not None
    assert torch.isfinite(y.grad).all()


def test_perturbation_sampling(sp_data):
    plant = DifferentiableSecondaryPath(
        sp_data,
        delay_jitter_range=(0, 512),
        gain_db_range=(-3, 3),
        tilt_db_per_octave_range=(-2, 2),
        allpass_perturb=True,
        seed=7,
    )
    p = plant.sample_perturbation()
    assert 0 <= p["jitter"] <= 512                # 비대칭 지터 [C1]
    y = torch.randn(1, 1, 4096) * 0.05
    with torch.no_grad():
        out = plant(y, p)
    assert torch.isfinite(out).all()
