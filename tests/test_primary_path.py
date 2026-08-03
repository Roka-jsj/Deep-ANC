"""digital-reference 1차경로의 단위/지연 authority 회귀 테스트."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import yaml

from deep_anc.config import REPO_ROOT, default_d_noise_delay
from deep_anc.data.primary_path import resolve_digital_primary_path
from deep_anc.data.synth_dataset import SynthANCDataset
from deep_anc.dsp.secondary_path import load_secondary_path
from deep_anc.train.trainer import (
    resolve_run_until_step,
    validate_resume_physics,
    validate_training_physics,
)


@pytest.fixture()
def configs():
    data = yaml.safe_load((REPO_ROOT / "configs/data_sim.yaml").read_text(encoding="utf-8"))
    duct = yaml.safe_load((REPO_ROOT / "configs/duct.yaml").read_text(encoding="utf-8"))
    return data, duct


def _write_path(path, *, delay=123, sample_rate=48_000):
    np.savez(
        path,
        fir=np.array([0.25, -0.125, 0.0625], dtype=np.float32),
        delay_samples=delay,
        sample_rate=sample_rate,
        fit_improvement_db=1.0,
        coherence_median=0.95,
        excitation_band_hz=np.array([80.0, 1600.0]),
    )


def _dataset_config(data: dict, mode: str, reference_mode: str = "digital") -> dict:
    out = deepcopy(data)
    out.update(
        {
            "segment_seconds": 256 / 48_000,
            "reference_mode": reference_mode,
            "digital_primary_path_mode": mode,
            "digital_reference_lead_samples": 0,
            "source_mix_ratio": {"synthetic": 1.0},
            "level_dbfs": [0.0, 0.0],
            "snr_mic_noise_db": [300.0, 300.0],
            "dc_hum_prob": 0.0,
            "closed_loop": {"feedback_delay_samples": [1, 2]},
        }
    )
    return out


def _rir_bank(
    p_ref: tuple[float, ...] = (1.0,),
    p_err: tuple[float, ...] = (1.0,),
) -> dict[str, np.ndarray]:
    length = max(len(p_ref), len(p_err))
    bank = {
        key: np.zeros((12, length), dtype=np.float32)
        for key in ("p_ref", "p_err", "f_fb")
    }
    bank["p_ref"][:, : len(p_ref)] = np.asarray(p_ref, dtype=np.float32)
    bank["p_err"][:, : len(p_err)] = np.asarray(p_err, dtype=np.float32)
    return bank


class _DeterministicItemRng:
    def uniform(self, low, _high):
        return float(low)

    def choice(self, values, **_kwargs):
        return np.asarray(values).reshape(-1)[0]

    def integers(self, low, *_args, **_kwargs):
        return int(low)

    def standard_normal(self, size):
        return np.zeros(size, dtype=np.float32)

    def random(self):
        return 0.5


def _impulse_item(dataset: SynthANCDataset) -> dict:
    def impulse_source(_rng, _synth, n_samples=None):
        out = np.zeros(int(n_samples), dtype=np.float32)
        out[0] = 1.0
        return out

    dataset._sample_source = impulse_source
    return dataset._make_item(_DeterministicItemRng(), None)


def _expected_impulse_path(segment: int, fir: np.ndarray, delay: int) -> np.ndarray:
    expected = np.zeros(segment, dtype=np.float32)
    usable = min(len(fir), max(0, segment - delay))
    if usable:
        expected[delay : delay + usable] = np.asarray(fir[:usable], dtype=np.float32)
    return expected


def test_measured_primary_path_requires_npz(configs):
    data, duct = deepcopy(configs)
    data["digital_primary_path_mode"] = "measured"
    duct["digital_reference"]["primary_path_npz"] = None
    secondary = load_secondary_path(REPO_ROOT / duct["secondary_path"]["npz"])
    with pytest.raises(ValueError, match="primary_path_npz"):
        resolve_digital_primary_path(data, duct, 48_000, secondary)


def test_measured_primary_path_is_delay_authority(tmp_path, configs):
    data, duct = deepcopy(configs)
    path = tmp_path / "primary.npz"
    _write_path(path, delay=777)
    data["digital_primary_path_mode"] = "measured"
    duct["digital_reference"]["primary_path_npz"] = str(path)
    duct["digital_reference"]["d_noise_delay_samples"] = 777
    secondary = load_secondary_path(REPO_ROOT / duct["secondary_path"]["npz"])

    primary, total_delay = resolve_digital_primary_path(data, duct, 48_000, secondary)

    assert primary is not None and not primary.is_surrogate
    assert primary.delay_samples == total_delay == 777
    np.testing.assert_allclose(primary.fir, [0.25, -0.125, 0.0625])


def test_measured_primary_rejects_conflicting_delay(tmp_path, configs):
    data, duct = deepcopy(configs)
    path = tmp_path / "primary.npz"
    _write_path(path, delay=777)
    data["digital_primary_path_mode"] = "measured"
    duct["digital_reference"]["primary_path_npz"] = str(path)
    duct["digital_reference"]["d_noise_delay_samples"] = 776
    secondary = load_secondary_path(REPO_ROOT / duct["secondary_path"]["npz"])
    with pytest.raises(ValueError, match="delay"):
        resolve_digital_primary_path(data, duct, 48_000, secondary)


def test_secondary_surrogate_reuses_s_gain_with_d_noise_delay(configs):
    data, duct = deepcopy(configs)
    data["digital_primary_path_mode"] = "secondary_surrogate"
    secondary = load_secondary_path(REPO_ROOT / duct["secondary_path"]["npz"])

    primary, total_delay = resolve_digital_primary_path(data, duct, 48_000, secondary)

    expected = default_d_noise_delay(duct, 48_000, secondary.delay_samples)
    assert primary is not None and primary.is_surrogate
    assert primary.mode == "secondary_surrogate"
    assert primary.delay_samples == total_delay == expected
    np.testing.assert_array_equal(primary.fir, secondary.fir)


def test_configs_explicitly_select_primary_path_policy(configs):
    data, duct = configs
    assert data["digital_primary_path_mode"] == "secondary_surrogate"
    assert "primary_path_npz" in duct["digital_reference"]
    assert duct["digital_reference"]["primary_path_npz"] is None


def test_finetune_requires_measured_primary_path(configs):
    data, duct = deepcopy(configs)
    cfg = {
        "data": data,
        "duct": duct,
        "require_measured_primary_path": True,
    }
    with pytest.raises(ValueError, match="실측 P\\(z\\)"):
        validate_training_physics(cfg)

    cfg["data"]["digital_primary_path_mode"] = "measured"
    assert validate_training_physics(cfg) == "measured_primary_path"


def test_surrogate_training_is_labeled_representation_only(configs):
    data, duct = deepcopy(configs)
    cfg = {"data": data, "duct": duct}
    assert (
        validate_training_physics(cfg)
        == "secondary_surrogate_representation_pretrain"
    )


def test_resume_requires_matching_lead_and_physics_mode():
    cfg = {
        "data": {
            "reference_mode": "digital",
            "digital_primary_path_mode": "measured",
            "digital_reference_lead_samples": 109,
        },
        "require_measured_primary_path": True,
    }
    matching = {
        "cfg": {
            "data": dict(cfg["data"]),
            "physics_status": "measured_primary_path",
            "digital_reference_lead_samples": 109,
        }
    }
    validate_resume_physics(matching, cfg)

    wrong_lead = {"cfg": dict(matching["cfg"], digital_reference_lead_samples=0)}
    with pytest.raises(ValueError, match="lead 불일치"):
        validate_resume_physics(wrong_lead, cfg)

    wrong_mode = {
        "cfg": dict(
            matching["cfg"],
            physics_status="secondary_surrogate_representation_pretrain",
        )
    }
    with pytest.raises(ValueError, match="physics mode 불일치"):
        validate_resume_physics(wrong_mode, cfg)


def test_resume_rejects_legacy_checkpoint_without_physics_metadata():
    cfg = {
        "data": {
            "reference_mode": "digital",
            "digital_primary_path_mode": "secondary_surrogate",
            "digital_reference_lead_samples": 0,
        }
    }
    with pytest.raises(ValueError, match="legacy artifact"):
        validate_resume_physics({"cfg": {"model": {}}}, cfg)


def test_run_until_step_preserves_long_schedule_and_validates_bounds():
    assert resolve_run_until_step({}, 100_000) == 100_000
    assert resolve_run_until_step({"run_until_step": 20_000}, 100_000) == 20_000
    with pytest.raises(ValueError, match="run_until_step"):
        resolve_run_until_step({"run_until_step": 100_001}, 100_000)


def test_dataset_measured_primary_applies_fir_and_delay_once(tmp_path, configs):
    data, duct = deepcopy(configs)
    primary_path = tmp_path / "primary.npz"
    _write_path(primary_path, delay=7)
    data = _dataset_config(data, "measured")
    duct["digital_reference"].update(
        {"primary_path_npz": str(primary_path), "d_noise_delay_samples": 7}
    )
    ds = SynthANCDataset(
        data,
        duct,
        split="train",
        seed=1,
        rir_bank=_rir_bank(p_err=(9.0, 8.0, 7.0)),
    )

    item = _impulse_item(ds)
    expected = _expected_impulse_path(
        ds.segment, np.array([0.25, -0.125, 0.0625], dtype=np.float32), 7
    )

    assert ds.digital_primary_path is not None
    assert ds.digital_primary_path.mode == "measured"
    assert ds.d_noise_total == ds.d_noise_delay == 7
    np.testing.assert_allclose(item["d"][0].numpy(), expected, atol=1e-6)


def test_dataset_secondary_surrogate_applies_s_fir_with_d_noise_once(
    tmp_path, configs
):
    data, duct = deepcopy(configs)
    secondary_path = tmp_path / "secondary.npz"
    _write_path(secondary_path, delay=123)
    data = _dataset_config(data, "secondary_surrogate")
    duct["secondary_path"]["npz"] = str(secondary_path)
    duct["digital_reference"]["d_noise_delay_samples"] = 11
    ds = SynthANCDataset(
        data,
        duct,
        split="train",
        seed=1,
        rir_bank=_rir_bank(p_err=(9.0, 8.0, 7.0)),
    )

    item = _impulse_item(ds)
    expected = _expected_impulse_path(
        ds.segment, np.array([0.25, -0.125, 0.0625], dtype=np.float32), 11
    )

    assert ds.digital_primary_path is not None
    assert ds.digital_primary_path.mode == "secondary_surrogate"
    assert ds.d_noise_total == ds.d_noise_delay == 11
    np.testing.assert_allclose(item["d"][0].numpy(), expected, atol=1e-6)


def test_dataset_rir_surrogate_keeps_legacy_rir_and_added_delay(configs):
    data, duct = deepcopy(configs)
    data = _dataset_config(data, "rir_surrogate")
    duct["positions_m"]["noise_speaker"] = duct["positions_m"]["error_mic"]
    duct["digital_reference"].update(
        {
            "primary_path_npz": "/unused/primary.npz",
            "d_noise_delay_samples": 5,
        }
    )
    p_err = np.array([0.0, 0.0, 0.75, -0.25], dtype=np.float32)
    ds = SynthANCDataset(
        data,
        duct,
        split="train",
        seed=1,
        rir_bank=_rir_bank(p_err=tuple(p_err)),
    )

    item = _impulse_item(ds)
    expected = _expected_impulse_path(ds.segment, p_err, 5)

    assert ds.digital_primary_path is None
    assert ds.d_noise_total == ds.d_noise_delay == 5
    np.testing.assert_allclose(item["d"][0].numpy(), expected, atol=1e-6)


def test_dataset_acoustic_mode_ignores_digital_primary_policy(configs):
    data, duct = deepcopy(configs)
    data = _dataset_config(data, "measured", reference_mode="acoustic")
    duct["digital_reference"]["primary_path_npz"] = None
    p_ref = np.array([0.5, 0.25], dtype=np.float32)
    p_err = np.array([-0.75, 0.125], dtype=np.float32)
    ds = SynthANCDataset(
        data,
        duct,
        split="train",
        seed=1,
        rir_bank=_rir_bank(p_ref=tuple(p_ref), p_err=tuple(p_err)),
    )

    item = _impulse_item(ds)

    assert ds.digital_primary_path is None
    np.testing.assert_allclose(
        item["x"][0].numpy(), _expected_impulse_path(ds.segment, p_ref, 0), atol=1e-6
    )
    np.testing.assert_allclose(
        item["d"][0].numpy(), _expected_impulse_path(ds.segment, p_err, 0), atol=1e-6
    )
