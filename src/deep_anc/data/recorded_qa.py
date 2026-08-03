"""실측 ANC 세션의 스트리밍 무결성·메타데이터 QA.

이 모듈은 오디오 파일을 읽기만 하며 재생 장치나 ``sounddevice``를 열지 않는다.
긴 녹음도 ``block_frames`` 단위로 순회해 전체 파형을 메모리에 올리지 않는다.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

from .manifest import VALID_SPLITS, validate_group_id, validate_source_family


@dataclass(frozen=True)
class RecordedQASettings:
    """실측 세션 QA 판정값."""

    sample_rate: int
    segment_samples: int
    digital_reference_lead_samples: int
    reference_mode: str = "digital"
    block_frames: int = 262_144
    clip_threshold: float = 0.99
    max_clip_ratio: float = 0.005
    min_mic_rms_dbfs: float = -80.0
    min_source_rms_dbfs: float = -80.0
    required_splits: tuple[str, ...] = VALID_SPLITS
    allow_incomplete_family_coverage: bool = False

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate는 양수여야 합니다")
        if self.segment_samples <= 0:
            raise ValueError("segment_samples는 양수여야 합니다")
        if self.digital_reference_lead_samples < 0:
            raise ValueError("digital_reference_lead_samples는 0 이상이어야 합니다")
        if self.reference_mode not in {"digital", "acoustic"}:
            raise ValueError(f"지원하지 않는 reference_mode: {self.reference_mode!r}")
        if self.reference_mode != "digital" and self.digital_reference_lead_samples:
            raise ValueError("acoustic reference QA에는 digital lead를 적용할 수 없습니다")
        if self.block_frames <= 0:
            raise ValueError("block_frames는 양수여야 합니다")
        if not 0.0 < self.clip_threshold <= 1.0:
            raise ValueError("clip_threshold는 0보다 크고 1 이하여야 합니다")
        if not 0.0 <= self.max_clip_ratio <= 1.0:
            raise ValueError("max_clip_ratio는 0 이상 1 이하여야 합니다")
        for value, name in (
            (self.min_mic_rms_dbfs, "min_mic_rms_dbfs"),
            (self.min_source_rms_dbfs, "min_source_rms_dbfs"),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name}는 유한값이어야 합니다")
        invalid = [split for split in self.required_splits if split not in VALID_SPLITS]
        if invalid:
            raise ValueError(f"지원하지 않는 required split: {invalid}")

    @property
    def effective_lead_samples(self) -> int:
        return (
            self.digital_reference_lead_samples
            if self.reference_mode == "digital"
            else 0
        )

    @property
    def minimum_frames(self) -> int:
        # RecordedANCDataset은 start 상한을 만들기 위해 segment+lead보다 최소
        # 1샘플 더 긴 세션을 요구한다.
        return self.segment_samples + self.effective_lead_samples + 1


def settings_from_data_config(
    data_cfg: dict,
    *,
    block_frames: int = 262_144,
    clip_threshold: float = 0.99,
    max_clip_ratio: float = 0.005,
    min_mic_rms_dbfs: float = -80.0,
    min_source_rms_dbfs: float = -80.0,
    required_splits: Iterable[str] = VALID_SPLITS,
    allow_incomplete_family_coverage: bool = False,
) -> RecordedQASettings:
    """학습 데이터 설정과 동일한 세그먼트/lead 최소 길이를 해석한다."""

    sample_rate = int(data_cfg["sample_rate"])
    raw_segment = int(round(float(data_cfg["segment_seconds"]) * sample_rate))
    segment_samples = max(256, (raw_segment // 256) * 256)
    return RecordedQASettings(
        sample_rate=sample_rate,
        segment_samples=segment_samples,
        digital_reference_lead_samples=int(
            data_cfg.get("digital_reference_lead_samples", 0)
        ),
        reference_mode=str(data_cfg.get("reference_mode", "digital")),
        block_frames=int(block_frames),
        clip_threshold=float(clip_threshold),
        max_clip_ratio=float(max_clip_ratio),
        min_mic_rms_dbfs=float(min_mic_rms_dbfs),
        min_source_rms_dbfs=float(min_source_rms_dbfs),
        required_splits=tuple(required_splits),
        allow_incomplete_family_coverage=bool(allow_incomplete_family_coverage),
    )


def _identifier(field: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}는 비어 있지 않은 문자열이어야 합니다")
    if value != value.strip() or len(value) > 128:
        raise ValueError(f"{field} 형식이 올바르지 않습니다: {value!r}")
    if any(ch in value for ch in ("/", "\\", "\0", "\n", "\r")):
        raise ValueError(f"{field}에 경로 구분자/제어문자를 사용할 수 없습니다")
    return value


def _dbfs_from_sum_squares(sum_squares: np.ndarray, frames: int) -> list[float]:
    rms = np.sqrt(sum_squares / max(1, frames))
    return [20.0 * math.log10(max(float(value), 1.0e-12)) for value in rms]


def _stream_audio(path: Path, settings: RecordedQASettings) -> dict:
    """오디오 하나를 블록 단위로 전수 검사한다."""

    with sf.SoundFile(str(path), mode="r") as audio:
        channels = int(audio.channels)
        declared_frames = int(audio.frames)
        sum_squares = np.zeros(channels, dtype=np.float64)
        clipped = np.zeros(channels, dtype=np.int64)
        nonfinite = np.zeros(channels, dtype=np.int64)
        peak = np.zeros(channels, dtype=np.float64)
        frames = 0
        blocks_read = 0

        while True:
            block = audio.read(
                frames=settings.block_frames,
                dtype="float32",
                always_2d=True,
            )
            if block.shape[0] == 0:
                break
            blocks_read += 1
            frames += int(block.shape[0])
            finite = np.isfinite(block)
            nonfinite += np.count_nonzero(~finite, axis=0)
            safe = np.where(finite, block, 0.0).astype(np.float64, copy=False)
            sum_squares += np.einsum("ij,ij->j", safe, safe)
            magnitude = np.abs(safe)
            clipped += np.count_nonzero(
                magnitude >= settings.clip_threshold, axis=0
            )
            peak = np.maximum(peak, np.max(magnitude, axis=0))

        return {
            "path": str(path),
            "sample_rate": int(audio.samplerate),
            "channels": channels,
            "frames": frames,
            "declared_frames": declared_frames,
            "duration_s": frames / float(audio.samplerate),
            "format": str(audio.format),
            "subtype": str(audio.subtype),
            "blocks_read": blocks_read,
            "rms_dbfs": _dbfs_from_sum_squares(sum_squares, frames),
            "peak": [float(value) for value in peak],
            "clip_ratio": [
                float(value) / max(1, frames) for value in clipped
            ],
            "nonfinite_samples": [int(value) for value in nonfinite],
        }


def _read_session_json(path: Path) -> tuple[dict | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"session.json을 읽을 수 없습니다: {exc}"
    if not isinstance(value, dict):
        return None, "session.json 최상위 값은 JSON 객체여야 합니다"
    return value, None


def _append_error(result: dict, message: str) -> None:
    result["errors"].append(message)


def _validate_manifest_metadata(entry: dict, session_path: Path, result: dict) -> None:
    for key in ("split", "source_family", "group_id", "session_id"):
        if key not in entry:
            _append_error(result, f"manifest 필수 필드 누락: {key}")

    split = entry.get("split")
    if split not in VALID_SPLITS:
        _append_error(result, f"잘못된 split: {split!r}")

    try:
        result["source_family"] = validate_source_family(entry.get("source_family"))
    except ValueError as exc:
        _append_error(result, str(exc))
    try:
        result["group_id"] = validate_group_id(entry.get("group_id"))
    except ValueError as exc:
        _append_error(result, str(exc))
    try:
        result["session_id"] = _identifier("session_id", entry.get("session_id"))
    except ValueError as exc:
        _append_error(result, str(exc))

    session_id = result.get("session_id")
    if session_id is not None and session_id != session_path.name:
        _append_error(
            result,
            f"manifest session_id({session_id!r})와 디렉터리명({session_path.name!r}) 불일치",
        )
    if entry.get("tag") not in (None, "recorded"):
        _append_error(result, f"recorded manifest의 tag가 아닙니다: {entry.get('tag')!r}")


def _validate_session_metadata(
    metadata: dict,
    entry: dict,
    result: dict,
    settings: RecordedQASettings,
) -> None:
    for key, validator in (
        ("source_family", validate_source_family),
        ("group_id", validate_group_id),
    ):
        if key not in metadata:
            _append_error(result, f"session.json 필수 필드 누락: {key}")
            continue
        try:
            value = validator(metadata[key])
        except ValueError as exc:
            _append_error(result, f"session.json: {exc}")
            continue
        if entry.get(key) != value:
            _append_error(
                result,
                f"session.json {key}({value!r})와 manifest({entry.get(key)!r}) 불일치",
            )

    # 현재 수집 포맷은 session_id를 디렉터리/manifest가 소유한다. 향후 JSON에도
    # 기록되면 그 값까지 엄격히 교차검증한다.
    if "session_id" in metadata:
        try:
            metadata_session_id = _identifier("session_id", metadata["session_id"])
        except ValueError as exc:
            _append_error(result, f"session.json: {exc}")
        else:
            if metadata_session_id != entry.get("session_id"):
                _append_error(
                    result,
                    "session.json session_id와 manifest session_id가 다릅니다: "
                    f"{metadata_session_id!r} != {entry.get('session_id')!r}",
                )

    if "sample_rate" in metadata:
        try:
            metadata_sr = int(metadata["sample_rate"])
        except (TypeError, ValueError):
            _append_error(result, "session.json sample_rate가 정수가 아닙니다")
        else:
            if metadata_sr != settings.sample_rate:
                _append_error(
                    result,
                    f"session.json sample_rate {metadata_sr} != {settings.sample_rate}",
                )


def _validate_audio(
    entry: dict,
    metadata: dict | None,
    mics: dict,
    source: dict | None,
    result: dict,
    settings: RecordedQASettings,
) -> None:
    if mics["channels"] != 2:
        _append_error(result, f"mics.wav는 정확히 2채널이어야 합니다: {mics['channels']}")
    audio_items = [("mics.wav", mics)]
    if source is not None:
        if source["channels"] != 1:
            _append_error(result, f"source.wav는 mono여야 합니다: {source['channels']}채널")
        audio_items.append(("source.wav", source))
    for label, stats in audio_items:
        if stats["sample_rate"] != settings.sample_rate:
            _append_error(
                result,
                f"{label} sample rate {stats['sample_rate']} != {settings.sample_rate}",
            )
        if stats["frames"] != stats["declared_frames"]:
            _append_error(
                result,
                f"{label} 헤더 frames({stats['declared_frames']})와 읽은 frames"
                f"({stats['frames']}) 불일치",
            )
        if sum(stats["nonfinite_samples"]) > 0:
            _append_error(
                result,
                f"{label}에 비유한 샘플 {sum(stats['nonfinite_samples'])}개",
            )
        for channel, ratio in enumerate(stats["clip_ratio"]):
            if ratio > settings.max_clip_ratio:
                _append_error(
                    result,
                    f"{label} ch{channel} clip ratio {ratio:.3%} > "
                    f"{settings.max_clip_ratio:.3%}",
                )

    if source is not None and mics["frames"] != source["frames"]:
        _append_error(
            result,
            f"mics/source 길이 불일치: {mics['frames']} != {source['frames']}",
        )
    shortest = min(mics["frames"], source["frames"]) if source is not None else mics["frames"]
    if shortest < settings.minimum_frames:
        lead_label = "digital lead" if settings.reference_mode == "digital" else "acoustic"
        _append_error(
            result,
            f"학습 세그먼트+{lead_label} 최소길이 미달: "
            f"{shortest} < {settings.minimum_frames}",
        )

    for channel, rms in enumerate(mics["rms_dbfs"]):
        if rms < settings.min_mic_rms_dbfs:
            _append_error(
                result,
                f"mics.wav ch{channel} RMS {rms:.1f}dBFS < "
                f"{settings.min_mic_rms_dbfs:.1f}dBFS",
            )

    if source is not None:
        family = str(entry.get("source_family", "")).lower()
        program = metadata.get("program", {}) if isinstance(metadata, dict) else {}
        program_type = (
            str(program.get("type", "")).lower() if isinstance(program, dict) else ""
        )
        intentional_silence = family == "silence" or program_type == "silence"
        source_rms = source["rms_dbfs"][0] if source["rms_dbfs"] else -240.0
        if not intentional_silence and source_rms < settings.min_source_rms_dbfs:
            _append_error(
                result,
                f"source.wav RMS {source_rms:.1f}dBFS < "
                f"{settings.min_source_rms_dbfs:.1f}dBFS",
            )

    manifest_sr = entry.get("sample_rate")
    if manifest_sr is None:
        _append_error(result, "manifest 필수 필드 누락: sample_rate")
    else:
        try:
            manifest_sr_value = int(manifest_sr)
        except (TypeError, ValueError):
            _append_error(result, f"manifest sample_rate가 정수가 아닙니다: {manifest_sr!r}")
        else:
            if manifest_sr_value != mics["sample_rate"]:
                _append_error(
                    result,
                    f"manifest sample_rate {manifest_sr_value} != mics {mics['sample_rate']}",
                )

    manifest_duration = entry.get("duration_s")
    if manifest_duration is None:
        _append_error(result, "manifest 필수 필드 누락: duration_s")
    else:
        try:
            duration_value = float(manifest_duration)
        except (TypeError, ValueError):
            _append_error(result, f"manifest duration_s가 숫자가 아닙니다: {manifest_duration!r}")
        else:
            if not math.isfinite(duration_value) or abs(duration_value - mics["duration_s"]) > (
                1.0 / settings.sample_rate
            ):
                _append_error(
                    result,
                    f"manifest duration_s {duration_value!r} != mics {mics['duration_s']:.6f}",
                )

    if "channels" in entry:
        try:
            manifest_channels = int(entry["channels"])
        except (TypeError, ValueError):
            _append_error(result, "manifest channels가 정수가 아닙니다")
        else:
            if manifest_channels != mics["channels"]:
                _append_error(
                    result,
                    f"manifest channels {manifest_channels} != mics {mics['channels']}",
                )

    if metadata is not None and "seconds" in metadata:
        try:
            metadata_seconds = float(metadata["seconds"])
        except (TypeError, ValueError):
            _append_error(result, "session.json seconds가 숫자가 아닙니다")
        else:
            if not math.isfinite(metadata_seconds) or abs(
                metadata_seconds - mics["duration_s"]
            ) > (1.0 / settings.sample_rate):
                _append_error(
                    result,
                    f"session.json seconds {metadata_seconds!r} != mics "
                    f"{mics['duration_s']:.6f}",
                )


def _validate_one_session(entry: dict, settings: RecordedQASettings) -> dict:
    session_path = Path(str(entry.get("path", "")))
    result: dict[str, Any] = {
        "path": str(session_path),
        "split": entry.get("split"),
        "session_id": entry.get("session_id"),
        "group_id": entry.get("group_id"),
        "source_family": entry.get("source_family"),
        "errors": [],
        "warnings": [],
        "audio": {},
    }
    _validate_manifest_metadata(entry, session_path, result)

    if not session_path.is_dir():
        _append_error(result, f"세션 디렉터리가 없습니다: {session_path}")
        result["ok"] = False
        return result

    paths = {
        "mics": session_path / "mics.wav",
        "source": session_path / "source.wav",
        "metadata": session_path / "session.json",
    }
    required_files = {"mics", "metadata"}
    if settings.reference_mode == "digital":
        required_files.add("source")
    for label in required_files:
        path = paths[label]
        if not path.is_file():
            _append_error(result, f"필수 파일 누락: {path.name}")

    metadata: dict | None = None
    if paths["metadata"].is_file():
        metadata, metadata_error = _read_session_json(paths["metadata"])
        if metadata_error is not None:
            _append_error(result, metadata_error)
        elif metadata is not None:
            _validate_session_metadata(metadata, entry, result, settings)

    stats: dict[str, dict] = {}
    audio_keys = ("mics", "source") if settings.reference_mode == "digital" else ("mics",)
    for key in audio_keys:
        path = paths[key]
        if not path.is_file():
            continue
        try:
            stats[key] = _stream_audio(path, settings)
        except (OSError, RuntimeError, ValueError) as exc:
            _append_error(result, f"{path.name} 스트리밍 읽기 실패: {exc}")

    result["audio"] = stats
    source_ready = settings.reference_mode != "digital" or "source" in stats
    if "mics" in stats and source_ready:
        _validate_audio(
            entry, metadata, stats["mics"], stats.get("source"), result, settings
        )
        result["duration_s"] = float(stats["mics"]["duration_s"])
    else:
        result["duration_s"] = 0.0
    result["ok"] = not result["errors"]
    return result


def _coverage_summary(results: list[dict]) -> tuple[dict, dict]:
    split_summary: dict[str, dict] = {}
    family_summary: dict[str, dict] = {}
    for result in results:
        split = str(result.get("split"))
        family = str(result.get("source_family"))
        group = str(result.get("group_id"))
        duration = float(result.get("duration_s", 0.0))

        split_item = split_summary.setdefault(
            split,
            {"sessions": 0, "valid_sessions": 0, "duration_s": 0.0, "groups": set(), "families": {}},
        )
        split_item["sessions"] += 1
        split_item["valid_sessions"] += int(bool(result.get("ok")))
        split_item["duration_s"] += duration
        split_item["groups"].add(group)
        split_item["families"][family] = split_item["families"].get(family, 0) + 1

        family_item = family_summary.setdefault(
            family,
            {"sessions": 0, "duration_s": 0.0, "groups": set(), "splits": {}},
        )
        family_item["sessions"] += 1
        family_item["duration_s"] += duration
        family_item["groups"].add(group)
        family_item["splits"][split] = family_item["splits"].get(split, 0) + 1

    for mapping in (split_summary, family_summary):
        for item in mapping.values():
            item["groups"] = len(item["groups"])
    return split_summary, family_summary


def validate_recorded_sessions(
    entries: list[dict], settings: RecordedQASettings, *, manifest_path: str = ""
) -> dict:
    """``read_manifest``가 반환한 세션 경로를 소비해 QA 리포트를 만든다."""

    results = [_validate_one_session(dict(entry), settings) for entry in entries]
    global_errors: list[str] = []
    global_warnings: list[str] = []

    if not entries:
        global_errors.append("manifest에 실측 세션이 없습니다")

    session_ids: dict[str, list[str]] = {}
    paths: dict[str, list[str]] = {}
    group_splits: dict[str, set[str]] = {}
    group_families: dict[str, set[str]] = {}
    for result in results:
        session_id = str(result.get("session_id"))
        path = str(result.get("path"))
        group = str(result.get("group_id"))
        split = str(result.get("split"))
        family = str(result.get("source_family"))
        session_ids.setdefault(session_id, []).append(path)
        paths.setdefault(path, []).append(session_id)
        group_splits.setdefault(group, set()).add(split)
        group_families.setdefault(group, set()).add(family)

    for session_id, session_paths in session_ids.items():
        if len(session_paths) > 1:
            global_errors.append(
                f"중복 session_id={session_id!r}: {', '.join(session_paths)}"
            )
    for path, identifiers in paths.items():
        if len(identifiers) > 1:
            global_errors.append(f"중복 세션 경로={path!r}: {identifiers}")
    for group, splits in group_splits.items():
        if len(splits) > 1:
            global_errors.append(
                f"치명: group_id={group!r}가 여러 split에 걸쳐 있습니다: {sorted(splits)}"
            )
    for group, families in group_families.items():
        if len(families) > 1:
            global_errors.append(
                f"group_id={group!r}의 source_family가 일관되지 않습니다: {sorted(families)}"
            )

    split_summary, family_summary = _coverage_summary(results)
    for split in settings.required_splits:
        if split not in split_summary or split_summary[split]["sessions"] == 0:
            global_errors.append(f"필수 split에 세션이 없습니다: {split}")

    for family, item in sorted(family_summary.items()):
        missing = [split for split in settings.required_splits if split not in item["splits"]]
        if missing:
            message = f"source_family={family!r}가 다음 split에 없습니다: {missing}"
            if settings.allow_incomplete_family_coverage:
                global_warnings.append(message)
            else:
                global_errors.append(message)

    total_duration = sum(float(result.get("duration_s", 0.0)) for result in results)
    report = {
        "ok": not global_errors and all(result.get("ok") for result in results),
        "manifest": manifest_path,
        "settings": {
            "sample_rate": settings.sample_rate,
            "segment_samples": settings.segment_samples,
            "digital_reference_lead_samples": settings.effective_lead_samples,
            "minimum_frames": settings.minimum_frames,
            "block_frames": settings.block_frames,
            "clip_threshold": settings.clip_threshold,
            "max_clip_ratio": settings.max_clip_ratio,
            "min_mic_rms_dbfs": settings.min_mic_rms_dbfs,
            "min_source_rms_dbfs": settings.min_source_rms_dbfs,
            "required_splits": list(settings.required_splits),
            "allow_incomplete_family_coverage": settings.allow_incomplete_family_coverage,
        },
        "summary": {
            "sessions": len(results),
            "valid_sessions": sum(int(bool(result.get("ok"))) for result in results),
            "invalid_sessions": sum(int(not bool(result.get("ok"))) for result in results),
            "groups": len(group_splits),
            "families": len(family_summary),
            "duration_s": total_duration,
            "splits": split_summary,
            "source_families": family_summary,
        },
        "errors": global_errors,
        "warnings": global_warnings,
        "sessions": results,
    }
    return report


def failure_report(message: str, *, manifest_path: str = "") -> dict:
    """manifest/config 로드 자체가 실패했을 때도 저장 가능한 리포트."""

    return {
        "ok": False,
        "manifest": manifest_path,
        "settings": {},
        "summary": {
            "sessions": 0,
            "valid_sessions": 0,
            "invalid_sessions": 0,
            "groups": 0,
            "families": 0,
            "duration_s": 0.0,
            "splits": {},
            "source_families": {},
        },
        "errors": [message],
        "warnings": [],
        "sessions": [],
    }


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_recorded_qa_markdown(report: dict) -> str:
    """JSON 리포트와 같은 내용을 사람이 읽는 Markdown으로 렌더링한다."""

    summary = report.get("summary", {})
    settings = report.get("settings", {})
    lines = [
        "# 실측 ANC 세션 QA 리포트",
        "",
        f"- 판정: **{'PASS' if report.get('ok') else 'FAIL'}**",
        f"- manifest: `{_markdown_cell(report.get('manifest', ''))}`",
        f"- 세션: {summary.get('valid_sessions', 0)}/{summary.get('sessions', 0)} 유효",
        f"- 분량: {float(summary.get('duration_s', 0.0)) / 60.0:.2f}분",
    ]
    if settings:
        lines += [
            f"- 최소 길이: {settings.get('minimum_frames')} samples "
            f"(segment {settings.get('segment_samples')} + lead "
            f"{settings.get('digital_reference_lead_samples')} + 1)",
            f"- 판정값: mic/source RMS ≥ {settings.get('min_mic_rms_dbfs')}/"
            f"{settings.get('min_source_rms_dbfs')}dBFS, clip ≤ "
            f"{100.0 * float(settings.get('max_clip_ratio', 0.0)):.3f}%",
        ]

    lines += [
        "",
        "## Split 커버리지",
        "",
        "| split | sessions | valid | groups | duration(min) | source families |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for split in VALID_SPLITS:
        item = summary.get("splits", {}).get(split, {})
        families = ", ".join(
            f"{name}:{count}" for name, count in sorted(item.get("families", {}).items())
        ) or "-"
        lines.append(
            f"| {split} | {item.get('sessions', 0)} | {item.get('valid_sessions', 0)} | "
            f"{item.get('groups', 0)} | {float(item.get('duration_s', 0.0)) / 60.0:.2f} | "
            f"{_markdown_cell(families)} |"
        )

    lines += [
        "",
        "## Source-family 커버리지",
        "",
        "| family | sessions | groups | duration(min) | train / val / test |",
        "|---|---:|---:|---:|---|",
    ]
    for family, item in sorted(summary.get("source_families", {}).items()):
        split_counts = item.get("splits", {})
        lines.append(
            f"| {_markdown_cell(family)} | {item.get('sessions', 0)} | {item.get('groups', 0)} | "
            f"{float(item.get('duration_s', 0.0)) / 60.0:.2f} | "
            f"{split_counts.get('train', 0)} / {split_counts.get('val', 0)} / "
            f"{split_counts.get('test', 0)} |"
        )

    lines += [
        "",
        "## 세션 검사",
        "",
        "| session | split | family | group | duration(s) | mic RMS ch0/ch1 | source RMS | "
        "max clip | blocks | 판정 |",
        "|---|---|---|---|---:|---|---:|---:|---:|---|",
    ]
    for result in report.get("sessions", []):
        audio = result.get("audio", {})
        mics = audio.get("mics", {})
        source = audio.get("source", {})
        mic_rms = "/".join(f"{value:.1f}" for value in mics.get("rms_dbfs", [])) or "-"
        source_rms_values = source.get("rms_dbfs", [])
        source_rms = f"{source_rms_values[0]:.1f}" if source_rms_values else "-"
        clip_values = list(mics.get("clip_ratio", [])) + list(source.get("clip_ratio", []))
        max_clip = 100.0 * max(clip_values, default=0.0)
        blocks = int(mics.get("blocks_read", 0)) + int(source.get("blocks_read", 0))
        verdict = "PASS" if result.get("ok") else "FAIL: " + "; ".join(result.get("errors", []))
        lines.append(
            f"| {_markdown_cell(result.get('session_id'))} | {_markdown_cell(result.get('split'))} | "
            f"{_markdown_cell(result.get('source_family'))} | {_markdown_cell(result.get('group_id'))} | "
            f"{float(result.get('duration_s', 0.0)):.2f} | {mic_rms} | {source_rms} | "
            f"{max_clip:.3f}% | {blocks} | {_markdown_cell(verdict)} |"
        )

    if report.get("errors"):
        lines += ["", "## 치명 오류", ""]
        lines.extend(f"- {_markdown_cell(message)}" for message in report["errors"])
    if report.get("warnings"):
        lines += ["", "## 경고", ""]
        lines.extend(f"- {_markdown_cell(message)}" for message in report["warnings"])

    lines += [
        "",
        "> 이 검사는 파일을 블록 단위로 읽기만 하며 오디오 출력 장치를 열지 않는다.",
        "",
    ]
    return "\n".join(lines)
