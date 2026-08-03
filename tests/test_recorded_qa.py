"""실측 세션 스트리밍 QA 게이트 회귀 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from deep_anc.data.manifest import MANIFEST_PATH_BASE, read_manifest, write_manifest
from deep_anc.data.recorded_qa import (
    RecordedQASettings,
    render_recorded_qa_markdown,
    validate_recorded_sessions,
)
from scripts.data.validate_recorded_sessions import main as qa_main


FS = 48_000


def _settings(*, minimum_segment: int = 512, lead: int = 7) -> RecordedQASettings:
    return RecordedQASettings(
        sample_rate=FS,
        segment_samples=minimum_segment,
        digital_reference_lead_samples=lead,
        block_frames=127,
        required_splits=("train", "val", "test"),
    )


def _signals(frames: int, *, source_channels: int = 1) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(frames, dtype=np.float64) / FS
    mics = np.stack(
        [0.08 * np.sin(2 * np.pi * 310 * t), 0.04 * np.sin(2 * np.pi * 430 * t)],
        axis=1,
    ).astype(np.float32)
    source = (0.05 * np.sin(2 * np.pi * 310 * t)).astype(np.float32)
    if source_channels == 2:
        source = np.stack([source, source], axis=1)
    return mics, source


def _write_session(
    session: Path,
    *,
    session_id: str,
    group_id: str,
    family: str = "speech",
    frames: int = 1200,
    mics: np.ndarray | None = None,
    source: np.ndarray | None = None,
    mics_sr: int = FS,
    source_sr: int = FS,
    metadata: dict | None = None,
) -> dict:
    session.mkdir(parents=True)
    default_mics, default_source = _signals(frames)
    mics = default_mics if mics is None else mics
    source = default_source if source is None else source
    sf.write(session / "mics.wav", mics, mics_sr, subtype="FLOAT")
    sf.write(session / "source.wav", source, source_sr, subtype="FLOAT")
    if metadata is None:
        metadata = {
            "session_id": session_id,
            "group_id": group_id,
            "source_family": family,
            "sample_rate": FS,
            "seconds": frames / FS,
            "program": {"type": "tone"},
        }
    (session / "session.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "path": str(session),
        "duration_s": frames / FS,
        "sample_rate": FS,
        "channels": 2,
        "tag": "recorded",
        "session_id": session_id,
        "group_id": group_id,
        "source_family": family,
    }


def test_recorded_qa_consumes_resolved_manifest_paths_and_streams_all_blocks(tmp_path):
    manifest = tmp_path / "data" / "manifests" / "recorded.jsonl"
    sessions_root = tmp_path / "data" / "recorded"
    entries = []
    for split, suffix in (("train", "a"), ("val", "b"), ("test", "c")):
        session_id = f"session-{suffix}"
        entry = _write_session(
            sessions_root / session_id,
            session_id=session_id,
            group_id=f"speaker-{suffix}",
        )
        entry.update(
            {
                "path": f"../recorded/{session_id}",
                "path_base": MANIFEST_PATH_BASE,
                "split": split,
            }
        )
        entries.append(entry)
    write_manifest(entries, manifest)

    resolved = read_manifest(manifest)
    report = validate_recorded_sessions(
        resolved, _settings(), manifest_path=str(manifest)
    )

    assert report["ok"]
    assert report["summary"]["sessions"] == 3
    assert report["summary"]["groups"] == 3
    assert all(Path(entry["path"]).is_absolute() for entry in resolved)
    assert all(
        session["audio"]["mics"]["blocks_read"] > 1
        and session["audio"]["source"]["blocks_read"] > 1
        for session in report["sessions"]
    )
    json.dumps(report, ensure_ascii=False)
    markdown = render_recorded_qa_markdown(report)
    assert "판정: **PASS**" in markdown
    assert "Source-family 커버리지" in markdown


def test_group_split_leak_is_a_fatal_global_error(tmp_path):
    first = _write_session(
        tmp_path / "session-a",
        session_id="session-a",
        group_id="same-speaker",
    )
    second = _write_session(
        tmp_path / "session-b",
        session_id="session-b",
        group_id="same-speaker",
    )
    first["split"] = "train"
    second["split"] = "test"

    report = validate_recorded_sessions(
        [first, second],
        RecordedQASettings(
            sample_rate=FS,
            segment_samples=512,
            digital_reference_lead_samples=7,
            block_frames=256,
            required_splits=("train", "test"),
        ),
    )

    assert not report["ok"]
    assert any("여러 split" in message for message in report["errors"])


def test_family_must_cover_all_required_splits_unless_diagnostic_override(tmp_path):
    entries = []
    for split, family in (("train", "speech"), ("val", "speech"), ("test", "music")):
        session_id = f"{split}-{family}"
        entry = _write_session(
            tmp_path / session_id,
            session_id=session_id,
            group_id=f"group-{session_id}",
            family=family,
        )
        entry["split"] = split
        entries.append(entry)

    strict = validate_recorded_sessions(entries, _settings())
    diagnostic = validate_recorded_sessions(
        entries,
        RecordedQASettings(
            sample_rate=FS,
            segment_samples=512,
            digital_reference_lead_samples=7,
            block_frames=127,
            required_splits=("train", "val", "test"),
            allow_incomplete_family_coverage=True,
        ),
    )

    assert not strict["ok"]
    assert any("source_family=" in message for message in strict["errors"])
    assert diagnostic["ok"]
    assert diagnostic["warnings"]


def test_acoustic_reference_does_not_require_or_read_source_wav(tmp_path):
    entries = []
    for split in ("train", "val", "test"):
        session_id = f"acoustic-{split}"
        entry = _write_session(
            tmp_path / session_id,
            session_id=session_id,
            group_id=f"room-{split}",
            family="machine",
        )
        (tmp_path / session_id / "source.wav").unlink()
        entry["split"] = split
        entries.append(entry)

    report = validate_recorded_sessions(
        entries,
        RecordedQASettings(
            sample_rate=FS,
            segment_samples=512,
            digital_reference_lead_samples=0,
            reference_mode="acoustic",
            block_frames=127,
            required_splits=("train", "val", "test"),
        ),
    )

    assert report["ok"]
    assert all("source" not in session["audio"] for session in report["sessions"])


def test_audio_shape_rate_and_lead_aware_minimum_length_fail(tmp_path):
    frames = 518  # segment 512 + lead 7 + 1보다 2샘플 부족
    mono_mics = np.full(frames, 0.05, dtype=np.float32)
    _, stereo_source = _signals(frames, source_channels=2)
    entry = _write_session(
        tmp_path / "bad-shape",
        session_id="bad-shape",
        group_id="group-shape",
        frames=frames,
        mics=mono_mics,
        source=stereo_source,
        source_sr=44_100,
    )
    entry["split"] = "train"

    report = validate_recorded_sessions(
        [entry],
        RecordedQASettings(
            sample_rate=FS,
            segment_samples=512,
            digital_reference_lead_samples=7,
            block_frames=128,
            required_splits=("train",),
        ),
    )
    errors = "\n".join(report["sessions"][0]["errors"])

    assert not report["ok"]
    assert "mics.wav는 정확히 2채널" in errors
    assert "source.wav는 mono" in errors
    assert "source.wav sample rate" in errors
    assert "최소길이 미달" in errors


def test_nonfinite_rms_clip_and_metadata_mismatch_fail(tmp_path):
    frames = 900
    mics, source = _signals(frames)
    mics[:, 1] = 0.0
    source[:] = 1.0
    source[0] = np.nan
    entry = _write_session(
        tmp_path / "bad-values",
        session_id="bad-values",
        group_id="group-values",
        family="music",
        frames=frames,
        mics=mics,
        source=source,
        metadata={
            "session_id": "different-session",
            "group_id": "different-group",
            "source_family": "speech",
            "sample_rate": 16_000,
            "seconds": 9.0,
            "program": {"type": "file"},
        },
    )
    entry["split"] = "train"

    report = validate_recorded_sessions(
        [entry],
        RecordedQASettings(
            sample_rate=FS,
            segment_samples=512,
            digital_reference_lead_samples=7,
            block_frames=128,
            required_splits=("train",),
        ),
    )
    errors = "\n".join(report["sessions"][0]["errors"])

    assert not report["ok"]
    assert "비유한 샘플" in errors
    assert "clip ratio" in errors
    assert "ch1 RMS" in errors
    assert "source_family" in errors
    assert "group_id" in errors
    assert "session_id" in errors
    assert "sample_rate" in errors
    assert "seconds" in errors


def test_missing_required_session_files_fail(tmp_path):
    session = tmp_path / "missing-files"
    session.mkdir()
    entry = {
        "path": str(session),
        "duration_s": 1.0,
        "sample_rate": FS,
        "channels": 2,
        "tag": "recorded",
        "session_id": "missing-files",
        "group_id": "group-missing",
        "source_family": "environment",
        "split": "train",
    }

    report = validate_recorded_sessions(
        [entry],
        RecordedQASettings(
            sample_rate=FS,
            segment_samples=512,
            digital_reference_lead_samples=7,
            required_splits=("train",),
        ),
    )
    errors = "\n".join(report["sessions"][0]["errors"])

    assert not report["ok"]
    assert "mics.wav" in errors
    assert "source.wav" in errors
    assert "session.json" in errors


def test_cli_writes_failure_reports_and_returns_nonzero_for_leaky_manifest(tmp_path):
    manifest = tmp_path / "leaky.jsonl"
    raw_entries = [
        {
            "path": "session-a",
            "split": "train",
            "group_id": "same",
            "source_family": "speech",
            "session_id": "session-a",
        },
        {
            "path": "session-b",
            "split": "test",
            "group_id": "same",
            "source_family": "speech",
            "session_id": "session-b",
        },
    ]
    manifest.write_text(
        "".join(json.dumps(entry) + "\n" for entry in raw_entries), encoding="utf-8"
    )
    data_config = tmp_path / "data.yaml"
    data_config.write_text(
        yaml.safe_dump(
            {
                "sample_rate": FS,
                "segment_seconds": 0.02,
                "reference_mode": "digital",
                "digital_reference_lead_samples": 7,
            }
        ),
        encoding="utf-8",
    )
    out_md = tmp_path / "qa.md"
    out_json = tmp_path / "qa.json"

    code = qa_main(
        [
            "--manifest",
            str(manifest),
            "--data-config",
            str(data_config),
            "--out-md",
            str(out_md),
            "--out-json",
            str(out_json),
        ]
    )

    assert code == 1
    assert out_md.is_file() and out_json.is_file()
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert not report["ok"]
    assert any("여러 split" in message for message in report["errors"])
    assert "판정: **FAIL**" in out_md.read_text(encoding="utf-8")
