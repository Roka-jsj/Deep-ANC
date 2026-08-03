import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from deep_anc.data.manifest import write_manifest
from deep_anc.dsp.secondary_path import DifferentiableSecondaryPath, SecondaryPathData
from deep_anc.eval.recorded import (
    RecordedEvalContext,
    RecordedSegment,
    deterministic_segment_starts,
    evaluate_recorded_segments,
    iter_recorded_segments,
    load_and_audit_recorded_manifest,
    load_recorded_eval_context,
    resolve_warmup_samples,
    validate_resolved_checkpoint,
    write_recorded_metrics,
)


FS = 8_000


def _path_data(path: Path, *, delay: int) -> None:
    np.savez_compressed(
        path,
        fir=np.asarray([1.0], dtype=np.float32),
        delay_samples=np.asarray(delay, dtype=np.int64),
        sample_rate=np.asarray(FS, dtype=np.int64),
        fit_improvement_db=np.asarray(20.0),
        coherence_median=np.asarray(0.99),
        excitation_band_hz=np.asarray([100.0, 1_000.0]),
    )


def _session(
    root: Path,
    name: str,
    *,
    group: str,
    family: str,
    samples: int = 64,
) -> Path:
    path = root / name
    path.mkdir()
    source = np.arange(samples, dtype=np.float32) / samples
    mics = np.stack([source, np.zeros_like(source)], axis=1)
    sf.write(path / "mics.wav", mics, FS, subtype="FLOAT")
    sf.write(path / "source.wav", source, FS, subtype="FLOAT")
    (path / "session.json").write_text(
        json.dumps(
            {
                "group_id": group,
                "source_family": family,
                "sample_rate": FS,
            }
        ),
        encoding="utf-8",
    )
    return path


def _entry(path: Path, split: str, session: str, group: str, family: str) -> dict:
    return {
        "path": str(path),
        "split": split,
        "duration_s": 1.0,
        "session_id": session,
        "group_id": group,
        "source_family": family,
    }


def _secondary(delay: int = 0) -> SecondaryPathData:
    return SecondaryPathData(
        fir=np.asarray([1.0], dtype=np.float32),
        delay_samples=delay,
        sample_rate=FS,
        fit_improvement_db=20.0,
        coherence_median=0.99,
        excitation_band_hz=(100.0, 1_000.0),
        source_path="test_secondary.npz",
    )


class AdvanceCancelModel(torch.nn.Module):
    def forward(self, x):
        return -x[:, :1]


def test_resolved_checkpoint_rejects_surrogate_and_lead_alias_mismatch():
    cfg = {
        "model": {},
        "data": {
            "reference_mode": "digital",
            "digital_primary_path_mode": "secondary_surrogate",
            "digital_reference_lead_samples": 3,
        },
        "duct": {},
        "physics_status": "secondary_surrogate_representation_pretrain",
        "trusted_band_hz": [100, 1_000],
        "digital_reference_lead_samples": 3,
    }
    state = {"cfg": cfg, "model": {}}

    with pytest.raises(ValueError, match="measured_primary_path"):
        validate_resolved_checkpoint(state)
    assert validate_resolved_checkpoint(state, allow_surrogate=True)[1] == 3

    cfg["digital_reference_lead_samples"] = 4
    with pytest.raises(ValueError, match="alias 불일치"):
        validate_resolved_checkpoint(state, allow_surrogate=True)


def test_context_rejects_lead_inconsistent_with_measured_primary(tmp_path):
    secondary = tmp_path / "secondary.npz"
    primary = tmp_path / "primary.npz"
    checkpoint = tmp_path / "model.pt"
    _path_data(secondary, delay=3)
    _path_data(primary, delay=4)
    cfg = {
        "model": {},
        "data": {
            "sample_rate": FS,
            "reference_mode": "digital",
            "digital_primary_path_mode": "measured",
            # expected = (S 3 + handoff 2) - P 4 = 1
            "digital_reference_lead_samples": 2,
        },
        "duct": {
            "secondary_path": {
                "npz": str(secondary),
                "handoff_extra_samples": 2,
            },
            "digital_reference": {
                "primary_path_npz": str(primary),
                "d_noise_delay_samples": 4,
            },
            "acoustics": {"realistic_target_band_hz": [100, 1_000]},
        },
        "physics_status": "measured_primary_path",
        "trusted_band_hz": [100, 1_000],
        "digital_reference_lead_samples": 2,
    }
    torch.save({"cfg": cfg, "model": {}}, checkpoint)

    with pytest.raises(ValueError, match="expected=1"):
        load_recorded_eval_context(checkpoint, device="cpu")


def test_manifest_audit_rejects_group_leakage_and_duplicate_paths(tmp_path):
    session = _session(tmp_path, "s1", group="g1", family="speech")
    leaking = tmp_path / "leaking.jsonl"
    leaking.write_text(
        "\n".join(
            json.dumps(entry)
            for entry in (
                _entry(session, "train", "s1", "g1", "speech"),
                _entry(session, "test", "s2", "g1", "speech"),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="여러 split"):
        load_and_audit_recorded_manifest(leaking, "test")

    duplicate = tmp_path / "duplicate.jsonl"
    write_manifest(
        [
            _entry(session, "test", "s1", "g1", "speech"),
            _entry(session, "test", "s1", "g1", "speech"),
        ],
        duplicate,
    )
    with pytest.raises(ValueError, match="중복 session path"):
        load_and_audit_recorded_manifest(duplicate, "test")


def test_recorded_segments_are_finite_deterministic_and_apply_lead(tmp_path):
    session = _session(tmp_path, "s1", group="g1", family="speech")
    entry = _entry(session, "test", "s1", "g1", "speech")
    data = {
        "sample_rate": FS,
        "segment_seconds": 8 / FS,
        "reference_mode": "digital",
        "digital_reference_lead_samples": 2,
        "closed_loop": {"feedback_delay_samples": [1, 1]},
    }

    first = list(
        iter_recorded_segments(
            [entry],
            data,
            model_hop=4,
            max_segments_per_session=2,
            edge_trim_seconds=0.0,
        )
    )
    second = list(
        iter_recorded_segments(
            [entry],
            data,
            model_hop=4,
            max_segments_per_session=2,
            edge_trim_seconds=0.0,
        )
    )

    assert [segment.start_sample for segment in first] == [0, 48]
    assert [segment.start_sample for segment in second] == [0, 48]
    np.testing.assert_array_equal(first[0].x, second[0].x)
    np.testing.assert_allclose(first[0].x[0], first[0].d + 2 / 64)
    np.testing.assert_allclose(first[0].x[1], np.r_[0.0, first[0].d[:-1]])
    assert deterministic_segment_starts(62, 8, 2) == [0, 48]
    assert deterministic_segment_starts(62, 8, 2, edge_trim_samples=8) == [8, 40]
    assert resolve_warmup_samples(data, FS) == 2_000


def test_recorded_segments_default_edge_trim_skips_session_boundaries(tmp_path):
    session = _session(
        tmp_path, "long", group="g1", family="environment", samples=5_000
    )
    data = {
        "sample_rate": FS,
        "segment_seconds": 8 / FS,
        "reference_mode": "digital",
        "digital_reference_lead_samples": 0,
        "closed_loop": {"feedback_delay_samples": [1, 1]},
    }

    segments = list(
        iter_recorded_segments(
            [_entry(session, "test", "long", "g1", "environment")],
            data,
            model_hop=4,
            max_segments_per_session=1,
        )
    )

    # 기본 0.25 s = 2,000 samples를 session 시작과 끝에서 모두 제외한다.
    assert [segment.start_sample for segment in segments] == [2_000]


def test_evaluation_applies_secondary_before_warmup_cut():
    samples = 256
    time = np.arange(samples, dtype=np.float32) / FS
    d = np.cos(2 * np.pi * 500 * time).astype(np.float32)
    # S(z)의 1 sample handoff 뒤 y(t-1)=-d(t)가 되도록 reference를 선행시킨다.
    reference = np.r_[d[1:], 0.0].astype(np.float32)
    segment = RecordedSegment(
        x=np.stack([reference, np.zeros_like(reference)]),
        d=d,
        session_id="s1",
        group_id="g1",
        source_family="speech",
        start_sample=0,
    )
    plant = DifferentiableSecondaryPath(
        _secondary(delay=0), handoff_extra_samples=1
    )

    result = evaluate_recorded_segments(
        AdvanceCancelModel(),
        plant,
        [segment],
        sample_rate=FS,
        trusted_band_hz=(100.0, 1_000.0),
        octave_bands_hz=[500.0],
        batch_size=1,
        warmup_samples=1,
    )

    assert result["fullband"]["mean_db"] < -100.0
    assert result["trusted"]["mean_db"] < -100.0
    assert result["source_rows"][0]["source_family"] == "speech"
    assert result["octave_rows"][0]["attenuation_mean_db"] > 100.0

    # warmup을 쓰지 않으면 S(z) 지연 때문에 첫 샘플의 e=d가 남는다. 위 결과가
    # 이보다 훨씬 좋아야 플랜트를 먼저 적용하고 나중에 자른 순서가 보장된다.
    untrimmed = evaluate_recorded_segments(
        AdvanceCancelModel(),
        plant,
        [segment],
        sample_rate=FS,
        trusted_band_hz=(100.0, 1_000.0),
        octave_bands_hz=[500.0],
        batch_size=1,
        warmup_samples=0,
    )
    assert untrimmed["fullband"]["mean_db"] > -30.0


def test_metrics_markdown_and_npz_include_source_octave_and_worst10(tmp_path):
    samples = 256
    time = np.arange(samples, dtype=np.float32) / FS
    d = np.sin(2 * np.pi * 500 * time).astype(np.float32)
    segments = [
        RecordedSegment(
            x=np.stack([d, np.zeros_like(d)]),
            d=d,
            session_id="speech_session",
            group_id="speech_group",
            source_family="speech",
            start_sample=0,
        ),
        RecordedSegment(
            x=np.stack([0.5 * d, np.zeros_like(d)]),
            d=d,
            session_id="music_session",
            group_id="music_group",
            source_family="music",
            start_sample=0,
        ),
    ]
    secondary = _secondary()
    plant = DifferentiableSecondaryPath(secondary)
    model = AdvanceCancelModel()
    result = evaluate_recorded_segments(
        model,
        plant,
        segments,
        sample_rate=FS,
        trusted_band_hz=(100.0, 1_000.0),
        octave_bands_hz=[500.0, 2_000.0],
        batch_size=2,
    )
    context = RecordedEvalContext(
        model=model,
        plant=plant,
        cfg={"model": {"hop": 4}, "data": {}, "duct": {}},
        device=torch.device("cpu"),
        sample_rate=FS,
        trusted_band_hz=(100.0, 1_000.0),
        physics_status="measured_primary_path",
        reference_mode="digital",
        digital_reference_lead_samples=1,
        expected_digital_reference_lead_samples=1,
        primary_delay_samples=0,
        secondary_path=secondary,
        secondary_handoff_samples=0,
    )

    markdown, archive = write_recorded_metrics(
        result,
        tmp_path / "out",
        checkpoint=tmp_path / "best.pt",
        manifest=tmp_path / "recorded.jsonl",
        split="test",
        context=context,
        feedback_delay_samples=1,
        allow_surrogate=False,
        edge_trim_samples=2_000,
        warmup_samples=0,
    )

    report = markdown.read_text(encoding="utf-8")
    assert "최악 10%" in report
    assert "speech" in report and "music" in report
    assert "500 Hz" in report
    assert "G4 종합: PASS" in report
    assert "S(z) 적용 후 절단" in report
    with np.load(archive, allow_pickle=False) as saved:
        assert saved["per_segment_trusted_db"].shape == (2,)
        assert set(saved["source_family"].tolist()) == {"music", "speech"}
        assert saved["octave_center_hz"].tolist() == [500.0, 2_000.0]
        assert str(saved["physics_status"]) == "measured_primary_path"
        assert bool(saved["g4_pass"])
        assert int(saved["edge_trim_samples"]) == 2_000
        assert int(saved["warmup_samples"]) == 0
