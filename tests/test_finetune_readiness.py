"""실측 fine-tune 진입/완료 실패-폐쇄 게이트 회귀 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from deep_anc.data.manifest import write_manifest
from deep_anc.train.finetune_readiness import (
    audit_finetune_completion,
    audit_finetune_readiness,
    audit_official_path_model,
    require_finetune_readiness,
    sha256_file,
)


FS = 8_000
FAMILIES = ("speech", "music", "environment", "machine")


def _official_path(
    path: Path,
    *,
    channel: str,
    delay: int,
    consistency: float = 0.97,
    amplitude: float = 0.005,
) -> None:
    np.savez(
        path,
        fir=np.asarray([0.5, -0.1, 0.02], dtype=np.float32),
        delay_samples=np.asarray(delay, dtype=np.int64),
        sample_rate=np.asarray(FS, dtype=np.int64),
        fit_improvement_db=np.asarray(np.nan),
        coherence_median=np.asarray(consistency),
        excitation_band_hz=np.asarray([100.0, 1_000.0]),
        calibration_block_size=np.asarray(256, dtype=np.int64),
        calibration_latency=np.asarray("high"),
        output_channel=np.asarray(channel),
        method=np.asarray("ess"),
        repeats=np.asarray(3, dtype=np.int64),
        amplitude=np.asarray(amplitude),
        xrun_count=np.asarray(0, dtype=np.int64),
        delay_spread_samples=np.asarray(1, dtype=np.int64),
        max_delay_jitter_samples=np.asarray(8, dtype=np.int64),
    )


def _checkpoint(
    path: Path,
    *,
    cfg: dict,
    step: int,
    best_metric: float = -3.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": {"weight": torch.ones(1)},
            "cfg": cfg,
            "step": step,
            "best_metric": best_metric,
        },
        path,
    )


def _recorded_manifest(root: Path, *, frames: int = 512) -> Path:
    manifest = root / "manifests" / "recorded.jsonl"
    entries = []
    t = np.arange(frames, dtype=np.float64) / FS
    for family_index, family in enumerate(FAMILIES):
        for split_index, split in enumerate(("train", "val", "test")):
            session_id = f"{family}-{split}"
            session = root / "recorded" / session_id
            session.mkdir(parents=True)
            source = (0.05 * np.sin(2 * np.pi * (250 + family_index * 40) * t)).astype(
                np.float32
            )
            mics = np.stack([0.7 * source, 0.4 * source], axis=1)
            sf.write(session / "mics.wav", mics, FS, subtype="FLOAT")
            sf.write(session / "source.wav", source, FS, subtype="FLOAT")
            group = f"group-{family}-{split_index}"
            (session / "session.json").write_text(
                json.dumps(
                    {
                        "session_id": session_id,
                        "group_id": group,
                        "source_family": family,
                        "sample_rate": FS,
                        "seconds": frames / FS,
                        "program": {"type": "file"},
                    }
                ),
                encoding="utf-8",
            )
            entries.append(
                {
                    "path": str(session),
                    "duration_s": frames / FS,
                    "sample_rate": FS,
                    "channels": 2,
                    "tag": "recorded",
                    "session_id": session_id,
                    "group_id": group,
                    "source_family": family,
                    "split": split,
                }
            )
    write_manifest(entries, manifest)
    return manifest


def _ready_config(tmp_path: Path) -> dict:
    primary = tmp_path / "primary.npz"
    secondary = tmp_path / "secondary.npz"
    _official_path(primary, channel="noise", delay=4)
    _official_path(secondary, channel="cancel", delay=5)
    manifest = _recorded_manifest(tmp_path / "data")
    model_cfg = {"name": "test-model", "hop": 4}
    pretrain_cfg = {
        "model": model_cfg,
        "data": {"digital_reference_lead_samples": 3},
        "digital_reference_lead_samples": 3,
        "physics_status": "secondary_surrogate_representation_pretrain",
        "schedule": {"total_steps": 10},
    }
    init_best = tmp_path / "pretrain" / "ckpt" / "best.pt"
    _checkpoint(init_best, cfg=pretrain_cfg, step=8)
    _checkpoint(init_best.parent / "last.pt", cfg=pretrain_cfg, step=10)

    return {
        "stage": "open_loop",
        "model": model_cfg,
        "data": {
            "sample_rate": FS,
            "segment_seconds": 0.01,
            "reference_mode": "digital",
            "digital_primary_path_mode": "measured",
            "digital_reference_lead_samples": 3,
            "closed_loop": {
                "feedback_delay_samples": [4, 8],
                "warmup_seconds": 0.0,
            },
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
        },
        "require_measured_primary_path": True,
        "require_init_checkpoint": True,
        "require_recorded_manifest": True,
        "init_ckpt": str(init_best),
        "recorded_manifest": str(manifest),
        "recorded_ratio": 0.7,
        "schedule": {"total_steps": 6},
        "ckpt_dir": str(tmp_path / "finetune"),
        "readiness": {
            "required_path_band_hz": [100, 1_000],
            "min_path_consistency": 0.9,
            "required_recorded_ratio": 0.7,
            "min_recorded_sessions": 12,
            "min_recorded_duration_seconds": 0.1,
            "required_source_families": list(FAMILIES),
            "require_completed_init_checkpoint": True,
            "max_init_best_metric_db": 0.0,
        },
    }


def _g4_metrics(path: Path, *, split: str, checkpoint: Path, manifest: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        split=np.asarray(split),
        physics_status=np.asarray("measured_primary_path"),
        allow_surrogate=np.asarray(False),
        checkpoint_sha256=np.asarray(sha256_file(checkpoint)),
        manifest_sha256=np.asarray(sha256_file(manifest)),
        g4_trusted_pass=np.asarray(True),
        g4_fullband_pass=np.asarray(True),
        g4_pass=np.asarray(True),
        source_family=np.asarray(FAMILIES),
        n_sessions=np.asarray(4, dtype=np.int64),
        n_segments=np.asarray(16, dtype=np.int64),
    )


def test_official_path_gate_rejects_wrong_channel_and_low_consistency(tmp_path):
    wrong = tmp_path / "wrong.npz"
    _official_path(wrong, channel="cancel", delay=4, consistency=0.8)

    try:
        audit_official_path_model(
            wrong,
            expected_output_channel="noise",
            sample_rate=FS,
            required_band_hz=(100, 1_000),
        )
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover - 명시적 실패 메시지를 보기 위한 방어
        raise AssertionError("invalid official path가 통과했습니다")

    assert "output_channel" in message
    assert "일관성" in message


def test_readiness_passes_only_with_official_paths_completed_init_and_full_recorded_qa(
    tmp_path,
):
    cfg = _ready_config(tmp_path)
    report = audit_finetune_readiness(cfg)

    assert report["ok"], report
    assert {item["id"] for item in report["checks"]} == {
        "config_fail_closed_flags",
        "measured_primary_mode",
        "recorded_mix_ratio",
        "official_secondary_path",
        "official_primary_path",
        "matched_path_measurement_conditions",
        "path_delay_and_lead",
        "completed_init_checkpoint",
        "recorded_dataset_qa",
    }
    assert require_finetune_readiness(cfg)["ok"]

    cfg["duct"]["digital_reference"]["d_noise_delay_samples"] = 5
    failed = audit_finetune_readiness(cfg)
    assert not failed["ok"]
    assert not next(
        item for item in failed["checks"] if item["id"] == "path_delay_and_lead"
    )["ok"]


def test_readiness_rejects_timing_invalid_or_legacy_path_metadata(tmp_path):
    cfg = _ready_config(tmp_path)
    legacy = tmp_path / "legacy_secondary.npz"
    np.savez(
        legacy,
        fir=np.ones(4, dtype=np.float32),
        delay_samples=5,
        sample_rate=FS,
        coherence_median=0.4,
        excitation_band_hz=np.asarray([150.0, 600.0]),
    )
    cfg["duct"]["secondary_path"]["npz"] = str(legacy)

    report = audit_finetune_readiness(cfg)
    path_check = next(
        item for item in report["checks"] if item["id"] == "official_secondary_path"
    )

    assert not report["ok"]
    assert not path_check["ok"]
    assert "official ESS 품질 메타데이터" in path_check["message"]


def test_completion_requires_same_checkpoint_and_manifest_sha_for_val_and_test(tmp_path):
    cfg = _ready_config(tmp_path)
    run = tmp_path / "finetune"
    best = run / "ckpt" / "best.pt"
    saved_cfg = {
        **cfg,
        "physics_status": "measured_primary_path",
        "digital_reference_lead_samples": 3,
    }
    _checkpoint(best, cfg=saved_cfg, step=4)
    _checkpoint(best.parent / "last.pt", cfg=saved_cfg, step=6)
    manifest = Path(cfg["recorded_manifest"])
    val_metrics = run / "eval_recorded_val" / "metrics.npz"
    test_metrics = run / "eval_recorded_test" / "metrics.npz"
    _g4_metrics(val_metrics, split="val", checkpoint=best, manifest=manifest)
    _g4_metrics(test_metrics, split="test", checkpoint=best, manifest=manifest)

    passed = audit_finetune_completion(
        cfg,
        checkpoint=best,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )
    assert passed["ok"], passed
    assert passed["fine_tuning_complete"]

    with np.load(test_metrics, allow_pickle=False) as current:
        arrays = {key: current[key] for key in current.files}
    arrays["checkpoint_sha256"] = np.asarray("0" * 64)
    np.savez_compressed(test_metrics, **arrays)
    failed = audit_finetune_completion(
        cfg,
        checkpoint=best,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )
    assert not failed["ok"]
    assert not failed["fine_tuning_complete"]
    assert "SHA-256" in next(
        item for item in failed["checks"] if item["id"] == "recorded_test_g4"
    )["message"]
