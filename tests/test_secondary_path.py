"""미분가능 S(z)가 scipy(lfilter+지연) 기준 구현과 일치하는지 검증."""

import numpy as np
import pytest
import torch
from scipy import signal

from deep_anc.config import REPO_ROOT, load_yaml
from deep_anc.dsp.secondary_path import (
    DifferentiableSecondaryPath,
    load_secondary_path,
)

# 파일명을 박지 않는다. duct.yaml 이 S(z) 의 단일 출처이므로 여기서 읽어야
# 자산을 교체했을 때 테스트가 **현재 쓰는 것**을 검증한다. 예전에는 파일명을 박아둬서
# S(z) 를 교체한 뒤에도 폐기된 자산만 계속 검사하고 있었다.
NPZ = REPO_ROOT / load_yaml(REPO_ROOT / "configs/duct.yaml")["secondary_path"]["npz"]


@pytest.fixture(scope="module")
def sp_data():
    return load_secondary_path(NPZ)


def test_load_measured_npz(sp_data):
    """현재 채택 S(z) 가 학습/런타임이 기대하는 형태인지 확인한다."""

    assert sp_data.fir.size == 2048
    assert sp_data.sample_rate == 48000
    assert sp_data.delay_samples > 0
    low, high = sp_data.excitation_band_hz
    assert 0.0 < low < high < 24_000.0
    # trusted band(150-600Hz)를 덮지 못하면 손실이 신뢰할 수 없는 대역을 최적화한다.
    assert low <= 150.0 and high >= 600.0


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


# ---------------------------------------------------------------------------
# 구동 대역 ≠ 신뢰 대역
# ---------------------------------------------------------------------------


def test_trusted_band_prefers_verified_consistency_band(tmp_path):
    """톤을 쏜 대역이 아니라 재현이 검증된 대역을 써야 한다.

    동시 인터리브 측정은 64-1648Hz 를 구동하지만 반복 일관성이 0.90 을 넘는 구간은
    150-600Hz 뿐이다. 구동 대역을 신뢰 대역으로 쓰면 손실이 재현되지 않는 대역까지
    최적화하고, 그 잘못된 위상이 gradient 를 지배해 신뢰 구간의 성능까지 잃는다.
    """

    path = tmp_path / "s.npz"
    np.savez(
        path,
        fir=np.asarray([0.5, -0.1], dtype=np.float32),
        delay_samples=np.asarray(100, dtype=np.int64),
        sample_rate=np.asarray(48_000, dtype=np.int64),
        excitation_band_hz=np.asarray([64.0, 1648.0]),
        consistency_band_hz=np.asarray([150.0, 600.0]),
    )
    sp = load_secondary_path(path)
    assert sp.excitation_band_hz == (64.0, 1648.0)
    assert sp.consistency_band_hz == (150.0, 600.0)
    assert sp.trusted_band_hz() == (150.0, 600.0)


def test_trusted_band_falls_back_to_excitation_band(tmp_path):
    """구버전 아티팩트에는 검증 대역이 없다 — 그때는 구동 대역이 최선의 정보다."""

    path = tmp_path / "s.npz"
    np.savez(
        path,
        fir=np.asarray([0.5, -0.1], dtype=np.float32),
        delay_samples=np.asarray(100, dtype=np.int64),
        sample_rate=np.asarray(48_000, dtype=np.int64),
        excitation_band_hz=np.asarray([150.0, 600.0]),
    )
    sp = load_secondary_path(path)
    assert sp.consistency_band_hz is None
    assert sp.trusted_band_hz() == (150.0, 600.0)


def test_current_secondary_path_declares_a_verified_band(sp_data):
    """현재 채택 S(z) 는 검증 대역을 갖고 있어야 한다 — 없으면 구버전 자산이다."""

    assert sp_data.consistency_band_hz is not None, (
        "consistency_band_hz 가 없는 S(z) 는 어느 대역에서 믿을 수 있는지 말하지 못한다"
    )
    low, high = sp_data.trusted_band_hz()
    assert low <= 150.0 and high >= 600.0
