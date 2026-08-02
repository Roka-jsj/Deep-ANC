"""데이터 파이프라인 검증 — shape/NaN/분할 누수/지연 물리."""

import numpy as np
import pytest
import torch
import yaml

from deep_anc.config import REPO_ROOT, default_d_noise_delay
from deep_anc.data.manifest import assign_splits
from deep_anc.data.synth_dataset import SynthANCDataset
from deep_anc.data.synthetic_signals import KINDS, SyntheticNoise
from deep_anc.dsp.duct_sim import build_rir_bank


@pytest.fixture(scope="module")
def cfgs():
    with open(REPO_ROOT / "configs" / "data_sim.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    with open(REPO_ROOT / "configs" / "duct.yaml", encoding="utf-8") as f:
        duct = yaml.safe_load(f)
    data = dict(data)
    data["segment_seconds"] = 0.5                 # 테스트 고속화
    data["source_mix_ratio"] = {"synthetic": 1.0}
    return data, duct


@pytest.fixture(scope="module")
def rir_bank(cfgs):
    _, duct = cfgs
    return build_rir_bank(duct, 48000, n_variants=12, ir_len=4096)


def test_synthetic_kinds():
    synth = SyntheticNoise(48000, seed=0)
    for kind in KINDS:
        x = synth.generate(4800, kind)
        assert x.shape == (4800,)
        assert np.all(np.isfinite(x))
        assert 0.5 < np.sqrt(np.mean(x**2)) < 2.0  # RMS 정규화


@pytest.mark.parametrize("mode", ["digital", "acoustic"])
def test_dataset_items(cfgs, rir_bank, mode):
    data, duct = cfgs
    data = dict(data)
    data["reference_mode"] = mode
    ds = SynthANCDataset(data, duct, split="train", seed=1, rir_bank=rir_bank)
    assert ds.segment % 256 == 0                  # 런타임 블록 배수 요건
    it = iter(ds)
    for _ in range(3):
        item = next(it)
        assert item["x"].shape == (2, ds.segment)
        assert item["d"].shape == (1, ds.segment)
        assert torch.isfinite(item["x"]).all()
        assert torch.isfinite(item["d"]).all()


def test_rir_split_no_leak(cfgs, rir_bank):
    data, duct = cfgs
    splits = {}
    for split in ("train", "val", "test"):
        ds = SynthANCDataset(data, duct, split=split, seed=1, rir_bank=rir_bank)
        splits[split] = set(ds.rir_indices.tolist())
    assert not (splits["train"] & splits["val"])
    assert not (splits["train"] & splits["test"])
    assert not (splits["val"] & splits["test"])


def test_manifest_split_assignment():
    entries = [{"path": f"f{i}.wav", "duration_s": 1.0} for i in range(100)]
    out = assign_splits(entries, {"train": 0.9, "val": 0.05}, seed=1)
    counts = {"train": 0, "val": 0, "test": 0}
    for e in out:
        counts[e["split"]] += 1
    assert counts["train"] == 90 and counts["val"] == 5 and counts["test"] == 5


def test_d_noise_default_geometry(cfgs):
    """digital-ref 기본 지연 = s_delay − t(CS→ERR) + t(NS→ERR) [C2]."""
    _, duct = cfgs
    fs = 48000
    d = default_d_noise_delay(duct, fs, s_path_delay=1342)
    # CS(1.050)→ERR(1.100)=7샘플, NS(0)→ERR(1.100)=154샘플 → 1342-7+154=1489
    assert d == 1342 - 7 + 154


def test_d_noise_no_double_count(cfgs, rir_bank):
    """리뷰 결함 #1 회귀: RIR 에 t(NS→ERR) 온셋이 포함되므로 dataset 추가 지연은
    총지연 − t(NS→ERR) 이어야 하고, 결과 d 의 온셋은 총지연(≈1489)과 일치해야 한다."""
    from deep_anc.config import duct_distance_samples
    from deep_anc.data.synth_dataset import _delay_np
    from deep_anc.dsp.filters import fft_filter

    data, duct = cfgs
    fs = 48000
    ds = SynthANCDataset(dict(data), duct, split="train", seed=1, rir_bank=rir_bank)
    total = default_d_noise_delay(duct, fs, s_path_delay=1342)
    t_ns_err = duct_distance_samples(duct, "noise_speaker", "error_mic", fs)
    assert ds.d_noise_total == total
    assert ds.d_noise_delay == total - t_ns_err

    imp = np.zeros(ds.segment, dtype=np.float32)
    imp[0] = 1.0
    d = _delay_np(fft_filter(imp, rir_bank["p_err"][0]), ds.d_noise_delay)
    onset = int(np.flatnonzero(np.abs(d) > np.max(np.abs(d)) * 0.05)[0])
    # RIR 위치 지터(±1cm)·저역통과 전이 여유 포함
    assert abs(onset - total) <= 24, f"d 온셋 {onset} vs 총지연 {total}"
