"""덕트 시뮬 검증 — closed–open 이론 공진(70/210/350Hz)이 재현되는지."""

import numpy as np
import pytest
import yaml

from deep_anc.config import REPO_ROOT
from deep_anc.dsp.duct_sim import (
    duct_paths,
    effective_length,
    find_resonances,
    image_source_ir,
)

FS = 48000


@pytest.fixture(scope="module")
def duct_cfg():
    with open(REPO_ROOT / "configs" / "duct.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_effective_length(duct_cfg):
    L = effective_length(duct_cfg)
    # 1.190 + 0.61×0.0592 ≈ 1.226 m (docs/09 문서 계산)
    assert 1.21 < L < 1.24


def test_ir_causal_direct_delay(duct_cfg):
    L = effective_length(duct_cfg)
    ir = image_source_ir(0.0, 1.100, L, FS)
    direct = int(np.floor(1.100 / 343.0 * FS))
    # 저역통과 필터 전이를 감안해 직접음 도달 이전 구간의 에너지가 0이어야 한다
    assert np.max(np.abs(ir[: direct - 8])) == pytest.approx(0.0, abs=1e-12)


def test_resonances_match_theory(duct_cfg):
    paths = duct_paths(duct_cfg, FS, ir_len=32768)
    found = find_resonances(paths["p_err"], FS, fmax=750.0)
    # closed–open 이론값 (문서): 70, 210, 350, 489, 629 Hz
    for target in (70.0, 210.0, 350.0):
        nearest = float(found[np.argmin(np.abs(found - target))]) if found.size else 0.0
        assert abs(nearest - target) < 15.0, f"{target}Hz 공진 미재현 (탐지: {found[:8]})"


def test_rir_bank_shapes(duct_cfg):
    from deep_anc.dsp.duct_sim import build_rir_bank

    bank = build_rir_bank(duct_cfg, FS, n_variants=8, ir_len=4096)
    for key in ("p_ref", "p_err", "f_fb"):
        assert bank[key].shape == (8, 4096)
        assert np.all(np.isfinite(bank[key]))
