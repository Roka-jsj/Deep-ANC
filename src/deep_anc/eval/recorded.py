"""독립 recorded val/test의 결정론적 오프라인 평가.

실측 세션은 ANC OFF로 녹음되므로 error mic 신호가 ``d``이다. 모델 출력 ``y``에
체크포인트가 보존한 S(z)와 런타임 handoff 지연을 적용해 ``e = d + S*y``를
계산한다. 이 모듈은 오디오 장치를 열거나 소리를 출력하지 않는다.
"""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import soundfile as sf
import torch

from ..config import DEFAULT_HANDOFF_SAMPLES, REPO_ROOT
from ..data.manifest import read_manifest
from ..data.primary_path import resolve_digital_primary_path
from ..data.synth_dataset import _delay_np
from ..dsp.secondary_path import (
    DifferentiableSecondaryPath,
    SecondaryPathData,
    load_secondary_path,
)
from ..models import build_model
from ..train.trainer import validate_training_physics
from .metrics import (
    band_nmse_db,
    intersect_frequency_bands,
    nmse_db,
    octave_band_attenuation,
)


@dataclass(frozen=True)
class RecordedEvalContext:
    """체크포인트에서 복원된 모델·플랜트·물리 메타데이터."""

    model: torch.nn.Module
    plant: DifferentiableSecondaryPath
    cfg: dict
    device: torch.device
    sample_rate: int
    trusted_band_hz: tuple[float, float]
    physics_status: str
    reference_mode: str
    digital_reference_lead_samples: int
    expected_digital_reference_lead_samples: int
    primary_delay_samples: int | None
    secondary_path: SecondaryPathData
    secondary_handoff_samples: int


@dataclass(frozen=True)
class RecordedSegment:
    """한 세션에서 잘라낸 고정 평가 구간."""

    x: np.ndarray  # [2, T]
    d: np.ndarray  # [T]
    session_id: str
    group_id: str
    source_family: str
    start_sample: int


def _config_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256_if_file(value: str | Path) -> str:
    """평가 provenance용 파일 지문. 없는 테스트 placeholder는 빈 값으로 남긴다."""

    path = _config_path(value)
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def validate_resolved_checkpoint(
    state: dict, *, allow_surrogate: bool = False
) -> tuple[dict, int, str, str]:
    """resolved checkpoint와 물리 상태/lead alias를 검증한다.

    반환은 ``(cfg, lead, physics_status, reference_mode)``이다. 기본 모드에서는
    measured P(z)로 학습한 체크포인트만 허용한다. ``allow_surrogate``는 실제
    성능 판정이 아닌 진단을 명시적으로 요청할 때만 사용한다.
    """

    if not isinstance(state, dict) or not isinstance(state.get("cfg"), dict):
        raise ValueError("checkpoint에 resolved cfg가 없습니다")
    cfg = state["cfg"]
    if not isinstance(state.get("model"), dict):
        raise ValueError("checkpoint에 model state_dict가 없습니다")
    for key in ("model", "data", "duct"):
        if not isinstance(cfg.get(key), dict):
            raise ValueError(f"checkpoint resolved cfg에 {key!r} 설정이 없습니다")
    if "physics_status" not in cfg:
        raise ValueError("checkpoint resolved cfg에 physics_status가 없습니다")
    if "trusted_band_hz" not in cfg:
        raise ValueError("checkpoint resolved cfg에 trusted_band_hz가 없습니다")
    if "digital_reference_lead_samples" not in cfg:
        raise ValueError(
            "checkpoint resolved cfg에 digital_reference_lead_samples alias가 없습니다"
        )

    data_cfg = cfg["data"]
    reference_mode = str(data_cfg.get("reference_mode", "digital"))
    if reference_mode not in {"digital", "acoustic"}:
        raise ValueError(
            f"지원하지 않는 checkpoint reference_mode: {reference_mode!r}"
        )
    nested_lead = int(data_cfg.get("digital_reference_lead_samples", 0))
    alias_lead = int(cfg["digital_reference_lead_samples"])
    if nested_lead < 0 or alias_lead < 0:
        raise ValueError("digital-reference lead는 0 이상이어야 합니다")
    if nested_lead != alias_lead:
        raise ValueError(
            "checkpoint digital-reference lead alias 불일치: "
            f"cfg={alias_lead}, data={nested_lead}"
        )
    if reference_mode != "digital" and nested_lead != 0:
        raise ValueError(
            "digital_reference_lead_samples는 reference_mode=digital에서만 "
            "사용할 수 있습니다"
        )

    saved_status = str(cfg["physics_status"])
    computed_status = validate_training_physics(cfg)
    if saved_status != computed_status:
        raise ValueError(
            "checkpoint physics_status와 resolved 설정이 다릅니다: "
            f"saved={saved_status}, resolved={computed_status}"
        )
    if not allow_surrogate and saved_status != "measured_primary_path":
        raise ValueError(
            "recorded 성능 평가는 measured_primary_path checkpoint만 허용합니다: "
            f"{saved_status}. 진단 목적이면 --allow-surrogate를 명시하세요."
        )
    if saved_status == "measured_primary_path":
        primary_path = cfg["duct"].get("digital_reference", {}).get(
            "primary_path_npz"
        )
        if not primary_path:
            raise ValueError(
                "measured_primary_path checkpoint에 primary_path_npz가 없습니다"
            )
    return cfg, nested_lead, saved_status, reference_mode


def load_recorded_eval_context(
    checkpoint: str | Path,
    *,
    allow_surrogate: bool = False,
    device: str | torch.device | None = None,
) -> RecordedEvalContext:
    """체크포인트의 resolved cfg만 사용해 모델과 공칭 S(z)를 복원한다."""

    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg, lead, physics_status, reference_mode = validate_resolved_checkpoint(
        state, allow_surrogate=allow_surrogate
    )
    data_cfg = cfg["data"]
    duct_cfg = cfg["duct"]
    fs = int(data_cfg["sample_rate"])
    if fs <= 0:
        raise ValueError(f"잘못된 checkpoint sample_rate: {fs}")

    secondary_cfg = duct_cfg.get("secondary_path", {})
    secondary_path_value = secondary_cfg.get("npz")
    if not secondary_path_value:
        raise ValueError("checkpoint duct.secondary_path.npz가 없습니다")
    secondary = load_secondary_path(_config_path(secondary_path_value))
    if int(secondary.sample_rate) != fs:
        raise ValueError(
            f"S(z) sample_rate={secondary.sample_rate}Hz != checkpoint={fs}Hz"
        )
    handoff = int(
        secondary_cfg.get("handoff_extra_samples", DEFAULT_HANDOFF_SAMPLES)
    )
    if handoff < 0:
        raise ValueError("secondary handoff 지연은 0 이상이어야 합니다")

    primary_delay: int | None = None
    expected_lead = 0
    if reference_mode == "digital":
        _, primary_delay = resolve_digital_primary_path(
            data_cfg, duct_cfg, fs, secondary
        )
        expected_lead = max(
            0, int(secondary.delay_samples) + handoff - int(primary_delay)
        )
        if lead != expected_lead:
            raise ValueError(
                "checkpoint digital-reference lead가 P/S 지연과 다릅니다: "
                f"checkpoint={lead}, expected={expected_lead} "
                f"(S={secondary.delay_samples}+handoff {handoff}, P={primary_delay})"
            )

    target_band = duct_cfg.get("acoustics", {}).get(
        "realistic_target_band_hz", [80.0, 1000.0]
    )
    computed_trusted = intersect_frequency_bands(
        secondary.excitation_band_hz, target_band, fs / 2.0
    )
    saved_trusted_raw = cfg["trusted_band_hz"]
    saved_trusted = intersect_frequency_bands(
        saved_trusted_raw, saved_trusted_raw, fs / 2.0
    )
    if not np.allclose(saved_trusted, computed_trusted, rtol=0.0, atol=1e-6):
        raise ValueError(
            "checkpoint trusted_band_hz가 현재 resolved P/S 설정과 다릅니다: "
            f"saved={saved_trusted}, expected={computed_trusted}"
        )

    if device is None or str(device) == "auto":
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device를 요청했지만 CUDA를 사용할 수 없습니다")

    model = build_model(cfg["model"])
    model.load_state_dict(state["model"])
    model.eval().to(resolved_device)
    plant = DifferentiableSecondaryPath(
        secondary, handoff_extra_samples=handoff
    ).eval().to(resolved_device)
    return RecordedEvalContext(
        model=model,
        plant=plant,
        cfg=cfg,
        device=resolved_device,
        sample_rate=fs,
        trusted_band_hz=computed_trusted,
        physics_status=physics_status,
        reference_mode=reference_mode,
        digital_reference_lead_samples=lead,
        expected_digital_reference_lead_samples=expected_lead,
        primary_delay_samples=primary_delay,
        secondary_path=secondary,
        secondary_handoff_samples=handoff,
    )


def load_and_audit_recorded_manifest(
    manifest_path: str | Path, split: str
) -> list[dict]:
    """전체 manifest의 path/session/group split 누수를 검사하고 split을 반환."""

    if split not in {"val", "test"}:
        raise ValueError("독립 recorded 평가는 split=val 또는 test만 허용합니다")
    entries = read_manifest(manifest_path)
    if not entries:
        raise ValueError(f"recorded manifest가 비어 있습니다: {manifest_path}")

    seen_paths: dict[str, str] = {}
    seen_sessions: dict[str, tuple[str, str]] = {}
    group_splits: dict[str, str] = {}
    group_families: dict[str, str] = {}
    for index, entry in enumerate(entries):
        for key in ("path", "split", "session_id", "group_id", "source_family"):
            if key not in entry or not str(entry[key]).strip():
                raise ValueError(f"manifest entry #{index}에 {key!r}가 없습니다")
        entry_split = str(entry["split"])
        if entry_split not in {"train", "val", "test"}:
            raise ValueError(
                f"manifest entry #{index}의 split이 잘못되었습니다: {entry_split}"
            )
        resolved_path = str(Path(entry["path"]).expanduser().resolve())
        previous_path_split = seen_paths.setdefault(resolved_path, entry_split)
        if previous_path_split != entry_split:
            raise ValueError(
                f"같은 session path가 여러 split에 있습니다: {resolved_path} "
                f"({previous_path_split}, {entry_split})"
            )
        if previous_path_split == entry_split and any(
            str(Path(previous["path"]).expanduser().resolve()) == resolved_path
            for previous in entries[:index]
        ):
            raise ValueError(f"manifest에 중복 session path가 있습니다: {resolved_path}")

        session_id = str(entry["session_id"])
        previous_session = seen_sessions.setdefault(
            session_id, (entry_split, resolved_path)
        )
        if previous_session != (entry_split, resolved_path):
            raise ValueError(
                f"session_id={session_id!r}가 중복되거나 split을 넘나듭니다"
            )

        group_id = str(entry["group_id"])
        source_family = str(entry["source_family"])
        previous_group_split = group_splits.setdefault(group_id, entry_split)
        if previous_group_split != entry_split:
            raise ValueError(
                f"group_id={group_id!r}가 여러 split에 있습니다: "
                f"{previous_group_split}, {entry_split}"
            )
        previous_family = group_families.setdefault(group_id, source_family)
        if previous_family != source_family:
            raise ValueError(
                f"group_id={group_id!r}가 여러 source_family에 속합니다: "
                f"{previous_family}, {source_family}"
            )

    selected = [dict(entry) for entry in entries if entry["split"] == split]
    if not selected:
        raise ValueError(f"recorded manifest에 '{split}' split 세션이 없습니다")
    return sorted(
        selected,
        key=lambda entry: (
            str(entry["group_id"]),
            str(entry["session_id"]),
            str(entry["path"]),
        ),
    )


def deterministic_segment_starts(
    usable_samples: int,
    segment_samples: int,
    max_segments: int,
    edge_trim_samples: int = 0,
) -> list[int]:
    """양끝 trim 뒤 구간에 고르게 분포한 비중첩 segment 시작점을 반환."""

    usable_samples = int(usable_samples)
    segment_samples = int(segment_samples)
    max_segments = int(max_segments)
    edge_trim_samples = int(edge_trim_samples)
    if segment_samples <= 0:
        raise ValueError("segment_samples는 양수여야 합니다")
    if max_segments <= 0:
        raise ValueError("max_segments는 양수여야 합니다")
    if edge_trim_samples < 0:
        raise ValueError("edge_trim_samples는 0 이상이어야 합니다")
    trimmed_samples = usable_samples - 2 * edge_trim_samples
    count = trimmed_samples // segment_samples
    if count <= 0:
        return []
    candidates = (
        np.arange(count, dtype=np.int64) * segment_samples + edge_trim_samples
    )
    if count <= max_segments:
        return [int(value) for value in candidates]
    indices = np.linspace(0, count - 1, num=max_segments, dtype=np.int64)
    return [int(candidates[index]) for index in indices]


def resolve_feedback_delay(data_cfg: dict, requested: int | None = None) -> int:
    """학습 범위 안의 결정론적 feedback 지연을 선택한다(기본=중앙값)."""

    raw = data_cfg.get("closed_loop", {}).get(
        "feedback_delay_samples", [0, 0]
    )
    if isinstance(raw, (int, float)):
        lo = hi = int(raw)
    else:
        if not isinstance(raw, Sequence) or len(raw) != 2:
            raise ValueError(
                "closed_loop.feedback_delay_samples는 [lo, hi]여야 합니다"
            )
        lo, hi = int(raw[0]), int(raw[1])
    if lo < 0 or hi < lo:
        raise ValueError(f"잘못된 feedback delay 범위: {raw}")
    value = int(round((lo + hi) / 2.0)) if requested is None else int(requested)
    if not lo <= value <= hi:
        raise ValueError(
            f"feedback delay {value}가 checkpoint 학습 범위 [{lo}, {hi}] 밖입니다"
        )
    return value


def resolve_warmup_samples(
    data_cfg: dict,
    sample_rate: int,
    requested_seconds: float | None = None,
) -> int:
    """평가 지표에서 제외할 플랜트 적용 후 warmup 길이를 반환."""

    seconds = (
        float(data_cfg.get("closed_loop", {}).get("warmup_seconds", 0.25))
        if requested_seconds is None
        else float(requested_seconds)
    )
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError("warmup_seconds는 유한한 0 이상 값이어야 합니다")
    return int(round(seconds * int(sample_rate)))


def _read_session_metadata(session_dir: Path) -> dict:
    path = session_dir / "session.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: session metadata를 읽을 수 없습니다: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: 최상위는 JSON 객체여야 합니다")
    return value


def _load_session_audio(
    entry: dict, sample_rate: int, reference_mode: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    session_dir = Path(entry["path"])
    if not session_dir.is_dir():
        raise FileNotFoundError(f"recorded session 디렉터리가 없습니다: {session_dir}")

    metadata = _read_session_metadata(session_dir)
    for key in ("group_id", "source_family"):
        if key in metadata and str(metadata[key]) != str(entry[key]):
            raise ValueError(
                f"{session_dir}: manifest {key}={entry[key]!r}와 "
                f"session.json={metadata[key]!r}가 다릅니다"
            )
    if "sample_rate" in metadata and int(metadata["sample_rate"]) != sample_rate:
        raise ValueError(
            f"{session_dir}: session.json sample_rate={metadata['sample_rate']} "
            f"!= checkpoint={sample_rate}"
        )

    mics_path = session_dir / "mics.wav"
    if not mics_path.exists():
        raise FileNotFoundError(f"mics.wav가 없습니다: {mics_path}")
    mics, mic_rate = sf.read(mics_path, dtype="float32", always_2d=True)
    if int(mic_rate) != sample_rate:
        raise ValueError(
            f"{mics_path}: sample_rate={mic_rate} != checkpoint={sample_rate}"
        )
    if mics.shape[1] < 2:
        raise ValueError(f"{mics_path}: err/ref 2채널이 필요합니다")
    if not np.all(np.isfinite(mics[:, :2])):
        raise ValueError(f"{mics_path}: NaN/Inf가 있습니다")

    source: np.ndarray | None = None
    if reference_mode == "digital":
        source_path = session_dir / "source.wav"
        if not source_path.exists():
            raise FileNotFoundError(
                f"digital-reference 평가에 source.wav가 필요합니다: {source_path}"
            )
        source_audio, source_rate = sf.read(
            source_path, dtype="float32", always_2d=True
        )
        if int(source_rate) != sample_rate:
            raise ValueError(
                f"{source_path}: sample_rate={source_rate} != checkpoint={sample_rate}"
            )
        source = source_audio[:, 0]
        if not np.all(np.isfinite(source)):
            raise ValueError(f"{source_path}: NaN/Inf가 있습니다")

    return mics[:, 0], mics[:, 1], source


def iter_recorded_segments(
    entries: Sequence[dict],
    data_cfg: dict,
    *,
    model_hop: int,
    max_segments_per_session: int = 8,
    segment_seconds: float | None = None,
    feedback_delay_samples: int | None = None,
    edge_trim_seconds: float = 0.25,
) -> Iterator[RecordedSegment]:
    """manifest 세션을 유한하고 결정론적인 비중첩 segment로 변환한다."""

    fs = int(data_cfg["sample_rate"])
    hop = int(model_hop)
    if hop <= 0:
        raise ValueError("model hop은 양수여야 합니다")
    seconds = (
        float(data_cfg["segment_seconds"])
        if segment_seconds is None
        else float(segment_seconds)
    )
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError("segment_seconds는 유한한 양수여야 합니다")
    raw_segment = int(round(seconds * fs))
    segment = (raw_segment // hop) * hop
    if segment < hop:
        raise ValueError(
            f"segment_seconds={seconds}가 model hop {hop}샘플보다 짧습니다"
        )
    edge_trim_seconds = float(edge_trim_seconds)
    if not math.isfinite(edge_trim_seconds) or edge_trim_seconds < 0.0:
        raise ValueError("edge_trim_seconds는 유한한 0 이상 값이어야 합니다")
    edge_trim = int(round(edge_trim_seconds * fs))

    reference_mode = str(data_cfg.get("reference_mode", "digital"))
    lead = int(data_cfg.get("digital_reference_lead_samples", 0))
    feedback_delay = resolve_feedback_delay(data_cfg, feedback_delay_samples)

    for entry in entries:
        err, ref, source = _load_session_audio(entry, fs, reference_mode)
        if reference_mode == "digital":
            assert source is not None
            usable = min(err.size, source.size - lead)
        else:
            usable = min(err.size, ref.size)
        starts = deterministic_segment_starts(
            usable,
            segment,
            max_segments_per_session,
            edge_trim_samples=edge_trim,
        )
        if not starts:
            raise ValueError(
                f"{entry['path']}: lead={lead}, edge trim={edge_trim} 적용 후 "
                f"{segment}샘플 segment가 없습니다"
            )

        delayed_err = _delay_np(err, feedback_delay)
        for start in starts:
            stop = start + segment
            d = np.ascontiguousarray(err[start:stop], dtype=np.float32)
            if reference_mode == "digital":
                assert source is not None
                x_ref = source[start + lead : stop + lead]
            else:
                x_ref = ref[start:stop]
            err_input = delayed_err[start:stop]
            if x_ref.size != segment or err_input.size != segment or d.size != segment:
                raise RuntimeError(f"{entry['path']}: segment 길이 계산 오류")
            x = np.ascontiguousarray(
                np.stack([x_ref, err_input]), dtype=np.float32
            )
            yield RecordedSegment(
                x=x,
                d=d,
                session_id=str(entry["session_id"]),
                group_id=str(entry["group_id"]),
                source_family=str(entry["source_family"]),
                start_sample=int(start),
            )


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("평가 지표가 비었거나 NaN/Inf를 포함합니다")
    worst_count = max(1, int(math.ceil(array.size * 0.1)))
    worst = np.sort(array)[-worst_count:]
    return {
        "mean_db": float(np.mean(array)),
        "median_db": float(np.median(array)),
        "worst10_mean_db": float(np.mean(worst)),
        "worst10_threshold_db": float(np.percentile(array, 90.0)),
        "worst_db": float(np.max(array)),
    }


def evaluate_recorded_segments(
    model: torch.nn.Module,
    plant: DifferentiableSecondaryPath,
    segments: Iterable[RecordedSegment],
    *,
    sample_rate: int,
    trusted_band_hz: tuple[float, float],
    octave_bands_hz: Sequence[float],
    device: str | torch.device = "cpu",
    batch_size: int = 8,
    warmup_samples: int = 0,
) -> dict:
    """segment iterable을 배치 평가하고 작은 메트릭 배열만 메모리에 보존한다."""

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size는 양수여야 합니다")
    warmup_samples = int(warmup_samples)
    if warmup_samples < 0:
        raise ValueError("warmup_samples는 0 이상이어야 합니다")
    device = torch.device(device)
    model.eval()
    plant.eval()

    metadata: list[RecordedSegment] = []
    fullband_values: list[float] = []
    trusted_values: list[float] = []
    octave_values: dict[float, list[float]] = {
        float(center): [] for center in octave_bands_hz
    }
    octave_trusted: dict[float, bool] = {}

    def evaluate_batch(batch: list[RecordedSegment]) -> None:
        x_np = np.stack([segment.x for segment in batch])
        d_np = np.stack([segment.d for segment in batch])
        x = torch.from_numpy(x_np).to(device)
        d = torch.from_numpy(d_np[:, None, :]).to(device)
        with torch.no_grad():
            y = model(x)
            # 측정 FIR에 극성이 포함되어 있으므로 추가 부호 반전 금지.
            e = d + plant(y.float(), {"jitter": 0})
        # 반드시 전체 S(z)를 적용한 뒤 warmup을 자른다. y/d를 먼저 자르면
        # delay+FIR 상태가 사라져 segment 시작의 상쇄량을 잘못 계산한다.
        if warmup_samples >= d.shape[-1]:
            raise ValueError(
                f"warmup_samples={warmup_samples}가 segment 길이 "
                f"{d.shape[-1]} 이상입니다"
            )
        d_metric_np = d[:, 0, warmup_samples:].float().cpu().numpy()
        e_metric_np = e[:, 0, warmup_samples:].float().cpu().numpy()

        for index, segment in enumerate(batch):
            d_item = d_metric_np[index]
            e_item = e_metric_np[index]
            fullband_values.append(nmse_db(d_item, e_item))
            trusted_values.append(
                band_nmse_db(
                    d_item, e_item, sample_rate, trusted_band_hz
                )
            )
            band_rows = octave_band_attenuation(
                d_item,
                e_item,
                sample_rate,
                [float(value) for value in octave_bands_hz],
                trusted_band_hz,
            )
            for row in band_rows:
                center = float(row["center_hz"])
                octave_values.setdefault(center, []).append(
                    float(row["attenuation_db"])
                )
                octave_trusted[center] = bool(row["trusted"])
            metadata.append(segment)

    pending: list[RecordedSegment] = []
    expected_samples: int | None = None
    for segment in segments:
        if segment.x.ndim != 2 or segment.x.shape[0] != 2:
            raise ValueError(
                f"recorded segment x shape은 [2,T]여야 합니다: {segment.x.shape}"
            )
        if segment.d.ndim != 1 or segment.x.shape[-1] != segment.d.size:
            raise ValueError(
                f"recorded segment x/d 길이가 다릅니다: {segment.x.shape}, {segment.d.shape}"
            )
        if not np.all(np.isfinite(segment.x)) or not np.all(np.isfinite(segment.d)):
            raise ValueError("recorded segment에 NaN/Inf가 있습니다")
        if expected_samples is None:
            expected_samples = int(segment.d.size)
        elif int(segment.d.size) != expected_samples:
            raise ValueError(
                "모든 recorded segment 길이는 같아야 합니다: "
                f"{expected_samples} != {segment.d.size}"
            )
        pending.append(segment)
        if len(pending) == batch_size:
            evaluate_batch(pending)
            pending = []
    if pending:
        evaluate_batch(pending)
    if not metadata:
        raise ValueError("평가할 recorded segment가 없습니다")

    trusted_array = np.asarray(trusted_values, dtype=np.float64)
    fullband_array = np.asarray(fullband_values, dtype=np.float64)
    gap_array = trusted_array - fullband_array
    families = sorted({segment.source_family for segment in metadata})
    source_rows: list[dict] = []
    for family in families:
        indices = np.asarray(
            [
                index
                for index, segment in enumerate(metadata)
                if segment.source_family == family
            ],
            dtype=np.int64,
        )
        trusted_stats = _distribution(trusted_array[indices])
        fullband_stats = _distribution(fullband_array[indices])
        source_rows.append(
            {
                "source_family": family,
                "n_segments": int(indices.size),
                "n_sessions": len({metadata[index].session_id for index in indices}),
                "n_groups": len({metadata[index].group_id for index in indices}),
                "trusted": trusted_stats,
                "fullband": fullband_stats,
                "gap_mean_db": float(np.mean(gap_array[indices])),
            }
        )

    octave_rows: list[dict] = []
    for center in sorted(octave_values):
        values = np.asarray(octave_values[center], dtype=np.float64)
        if values.size == 0:
            continue
        worst_count = max(1, int(math.ceil(values.size * 0.1)))
        worst = np.sort(values)[:worst_count]  # 감쇠는 작은 값이 나쁨
        octave_rows.append(
            {
                "center_hz": center,
                "attenuation_mean_db": float(np.mean(values)),
                "attenuation_median_db": float(np.median(values)),
                "attenuation_worst10_mean_db": float(np.mean(worst)),
                "trusted": bool(octave_trusted.get(center, False)),
            }
        )

    return {
        "n_segments": len(metadata),
        "n_sessions": len({segment.session_id for segment in metadata}),
        "n_groups": len({segment.group_id for segment in metadata}),
        "segment_samples": int(metadata[0].d.size),
        "metric_samples_per_segment": int(metadata[0].d.size - warmup_samples),
        "trusted": _distribution(trusted_array),
        "fullband": _distribution(fullband_array),
        "gap_mean_db": float(np.mean(gap_array)),
        "warmup_samples": warmup_samples,
        "per_segment_trusted_db": trusted_array,
        "per_segment_fullband_db": fullband_array,
        "per_segment_gap_db": gap_array,
        "segment_session_id": np.asarray(
            [segment.session_id for segment in metadata], dtype=np.str_
        ),
        "segment_group_id": np.asarray(
            [segment.group_id for segment in metadata], dtype=np.str_
        ),
        "segment_source_family": np.asarray(
            [segment.source_family for segment in metadata], dtype=np.str_
        ),
        "segment_start_sample": np.asarray(
            [segment.start_sample for segment in metadata], dtype=np.int64
        ),
        "source_rows": source_rows,
        "octave_rows": octave_rows,
    }


def write_recorded_metrics(
    result: dict,
    out_dir: str | Path,
    *,
    checkpoint: str | Path,
    manifest: str | Path,
    split: str,
    context: RecordedEvalContext,
    feedback_delay_samples: int,
    allow_surrogate: bool,
    edge_trim_samples: int,
    warmup_samples: int,
) -> tuple[Path, Path]:
    """사람용 Markdown과 기계용 NPZ를 원자적으로 생성한다."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = out_dir / "metrics.md"
    npz_path = out_dir / "metrics.npz"
    trusted = result["trusted"]
    fullband = result["fullband"]
    if int(result.get("warmup_samples", warmup_samples)) != int(warmup_samples):
        raise ValueError("평가 결과와 보고서 warmup_samples가 다릅니다")
    trusted_pass = bool(trusted["mean_db"] < 0.0)
    fullband_pass = bool(fullband["mean_db"] <= 0.0)
    g4_pass = trusted_pass and fullband_pass
    checkpoint_sha256 = _sha256_if_file(checkpoint)
    manifest_sha256 = _sha256_if_file(manifest)

    lines = [
        f"# Recorded {split} 오프라인 평가",
        "",
    ]
    if context.physics_status != "measured_primary_path":
        lines += [
            "> [!WARNING]",
            "> `--allow-surrogate` 진단 결과입니다. 실측 덕트 성능으로 해석하면 안 됩니다.",
            "",
        ]
    lines += [
        f"- 체크포인트: `{Path(checkpoint)}`",
        f"- 체크포인트 SHA-256: `{checkpoint_sha256 or 'unavailable'}`",
        f"- Manifest: `{Path(manifest)}` (`{split}`)",
        f"- Manifest SHA-256: `{manifest_sha256 or 'unavailable'}`",
        f"- 물리 상태: `{context.physics_status}`",
        f"- 세션/그룹/세그먼트: {result['n_sessions']}/{result['n_groups']}/{result['n_segments']}",
        f"- 세그먼트 길이: {result['segment_samples']} samples; 지표 구간: "
        f"{result['metric_samples_per_segment']} samples",
        f"- Trusted 대역: {context.trusted_band_hz[0]:.0f}–{context.trusted_band_hz[1]:.0f} Hz",
        f"- Digital lead: {context.digital_reference_lead_samples} samples",
        f"- S(z) 지연: {context.secondary_path.delay_samples} + handoff "
        f"{context.secondary_handoff_samples} samples",
        f"- Feedback 입력 지연: {feedback_delay_samples} samples",
        f"- 세션 양끝 제외: {edge_trim_samples} samples "
        f"({edge_trim_samples / context.sample_rate:.3f} s/edge)",
        f"- 지표 warmup 제외: {warmup_samples} samples "
        f"({warmup_samples / context.sample_rate:.3f} s, S(z) 적용 후 절단)",
        "",
        "## 전체 결과",
        "",
        "NMSE는 낮을수록 좋습니다. 최악 10%는 세그먼트 NMSE가 큰 상위 10%의 평균입니다.",
        "",
        "| 지표 | 평균 | 중앙값 | 최악 10% 평균 | 최악 |",
        "|---|---:|---:|---:|---:|",
        f"| Trusted NMSE | {trusted['mean_db']:+.2f} dB | "
        f"{trusted['median_db']:+.2f} dB | {trusted['worst10_mean_db']:+.2f} dB | "
        f"{trusted['worst_db']:+.2f} dB |",
        f"| Fullband NMSE | {fullband['mean_db']:+.2f} dB | "
        f"{fullband['median_db']:+.2f} dB | {fullband['worst10_mean_db']:+.2f} dB | "
        f"{fullband['worst_db']:+.2f} dB |",
        "",
        f"Trusted−fullband 평균 간극: **{result['gap_mean_db']:+.2f} dB**",
        "",
        "## G4 독립 recorded 판정",
        "",
        "| 조건 | 기준 | 결과 | 판정 |",
        "|---|---:|---:|---|",
        f"| Trusted 평균 NMSE | < 0 dB | {trusted['mean_db']:+.2f} dB | "
        f"{'PASS' if trusted_pass else 'FAIL'} |",
        f"| Fullband 평균 NMSE | ≤ 0 dB | {fullband['mean_db']:+.2f} dB | "
        f"{'PASS' if fullband_pass else 'FAIL'} |",
        "",
        f"**G4 종합: {'PASS' if g4_pass else 'FAIL'}**",
        "",
        "## Source family별 결과",
        "",
        "| Source family | 세션 | 그룹 | 세그먼트 | Trusted 평균 | Trusted 최악 10% | Fullband 평균 | Fullband 최악 10% | 간극 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["source_rows"]:
        lines.append(
            f"| {row['source_family']} | {row['n_sessions']} | {row['n_groups']} | "
            f"{row['n_segments']} | {row['trusted']['mean_db']:+.2f} | "
            f"{row['trusted']['worst10_mean_db']:+.2f} | "
            f"{row['fullband']['mean_db']:+.2f} | "
            f"{row['fullband']['worst10_mean_db']:+.2f} | "
            f"{row['gap_mean_db']:+.2f} |"
        )
    lines += [
        "",
        "## 옥타브 밴드 감쇠",
        "",
        "감쇠는 높을수록 좋으며, 최악 10%는 감쇠가 작은 하위 10% 평균입니다.",
        "",
        "| 중심 주파수 | 평균 감쇠 | 중앙값 | 최악 10% 평균 | 신뢰 |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in result["octave_rows"]:
        confidence = "O" if row["trusted"] else "낮음*"
        lines.append(
            f"| {row['center_hz']:.0f} Hz | {row['attenuation_mean_db']:+.2f} dB | "
            f"{row['attenuation_median_db']:+.2f} dB | "
            f"{row['attenuation_worst10_mean_db']:+.2f} dB | {confidence} |"
        )
    lines += [
        "",
        "*: S(z) 실측 유효대역 밖이므로 광대역 재보정 전에는 참고용입니다.",
        "",
        "이 평가는 저장된 오디오 파일만 읽는 오프라인 계산이며 실제 오디오를 출력하지 않습니다.",
    ]
    markdown_tmp = markdown_path.with_suffix(".md.tmp")
    markdown_tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    markdown_tmp.replace(markdown_path)

    source_rows = result["source_rows"]
    octave_rows = result["octave_rows"]
    npz_tmp = npz_path.with_suffix(".npz.tmp")
    with open(npz_tmp, "wb") as file_obj:
        np.savez_compressed(
            file_obj,
            checkpoint=np.asarray(str(Path(checkpoint)), dtype=np.str_),
            checkpoint_sha256=np.asarray(checkpoint_sha256, dtype=np.str_),
            manifest=np.asarray(str(Path(manifest)), dtype=np.str_),
            manifest_sha256=np.asarray(manifest_sha256, dtype=np.str_),
            split=np.asarray(split, dtype=np.str_),
            physics_status=np.asarray(context.physics_status, dtype=np.str_),
            allow_surrogate=np.asarray(bool(allow_surrogate)),
            sample_rate=np.asarray(context.sample_rate, dtype=np.int64),
            n_sessions=np.asarray(result["n_sessions"], dtype=np.int64),
            n_groups=np.asarray(result["n_groups"], dtype=np.int64),
            n_segments=np.asarray(result["n_segments"], dtype=np.int64),
            segment_samples=np.asarray(result["segment_samples"], dtype=np.int64),
            metric_samples_per_segment=np.asarray(
                result["metric_samples_per_segment"], dtype=np.int64
            ),
            trusted_band_hz=np.asarray(context.trusted_band_hz, dtype=np.float64),
            digital_reference_lead_samples=np.asarray(
                context.digital_reference_lead_samples, dtype=np.int64
            ),
            primary_delay_samples=np.asarray(
                -1
                if context.primary_delay_samples is None
                else context.primary_delay_samples,
                dtype=np.int64,
            ),
            secondary_delay_samples=np.asarray(
                context.secondary_path.delay_samples, dtype=np.int64
            ),
            secondary_handoff_samples=np.asarray(
                context.secondary_handoff_samples, dtype=np.int64
            ),
            feedback_delay_samples=np.asarray(
                feedback_delay_samples, dtype=np.int64
            ),
            edge_trim_samples=np.asarray(edge_trim_samples, dtype=np.int64),
            warmup_samples=np.asarray(warmup_samples, dtype=np.int64),
            g4_trusted_pass=np.asarray(trusted_pass, dtype=np.bool_),
            g4_fullband_pass=np.asarray(fullband_pass, dtype=np.bool_),
            g4_pass=np.asarray(g4_pass, dtype=np.bool_),
            nmse_trusted_mean_db=np.asarray(trusted["mean_db"]),
            nmse_trusted_median_db=np.asarray(trusted["median_db"]),
            nmse_trusted_worst10_mean_db=np.asarray(
                trusted["worst10_mean_db"]
            ),
            nmse_fullband_mean_db=np.asarray(fullband["mean_db"]),
            nmse_fullband_median_db=np.asarray(fullband["median_db"]),
            nmse_fullband_worst10_mean_db=np.asarray(
                fullband["worst10_mean_db"]
            ),
            nmse_gap_trusted_minus_fullband_mean_db=np.asarray(
                result["gap_mean_db"]
            ),
            per_segment_trusted_db=result["per_segment_trusted_db"],
            per_segment_fullband_db=result["per_segment_fullband_db"],
            per_segment_gap_db=result["per_segment_gap_db"],
            segment_session_id=result["segment_session_id"],
            segment_group_id=result["segment_group_id"],
            segment_source_family=result["segment_source_family"],
            segment_start_sample=result["segment_start_sample"],
            source_family=np.asarray(
                [row["source_family"] for row in source_rows], dtype=np.str_
            ),
            source_n_segments=np.asarray(
                [row["n_segments"] for row in source_rows], dtype=np.int64
            ),
            source_n_sessions=np.asarray(
                [row["n_sessions"] for row in source_rows], dtype=np.int64
            ),
            source_n_groups=np.asarray(
                [row["n_groups"] for row in source_rows], dtype=np.int64
            ),
            source_nmse_trusted_mean_db=np.asarray(
                [row["trusted"]["mean_db"] for row in source_rows],
                dtype=np.float64,
            ),
            source_nmse_trusted_worst10_mean_db=np.asarray(
                [row["trusted"]["worst10_mean_db"] for row in source_rows],
                dtype=np.float64,
            ),
            source_nmse_fullband_mean_db=np.asarray(
                [row["fullband"]["mean_db"] for row in source_rows],
                dtype=np.float64,
            ),
            source_nmse_fullband_worst10_mean_db=np.asarray(
                [row["fullband"]["worst10_mean_db"] for row in source_rows],
                dtype=np.float64,
            ),
            source_gap_trusted_minus_fullband_mean_db=np.asarray(
                [row["gap_mean_db"] for row in source_rows], dtype=np.float64
            ),
            octave_center_hz=np.asarray(
                [row["center_hz"] for row in octave_rows], dtype=np.float64
            ),
            octave_attenuation_mean_db=np.asarray(
                [row["attenuation_mean_db"] for row in octave_rows],
                dtype=np.float64,
            ),
            octave_attenuation_median_db=np.asarray(
                [row["attenuation_median_db"] for row in octave_rows],
                dtype=np.float64,
            ),
            octave_attenuation_worst10_mean_db=np.asarray(
                [row["attenuation_worst10_mean_db"] for row in octave_rows],
                dtype=np.float64,
            ),
            octave_trusted=np.asarray(
                [row["trusted"] for row in octave_rows], dtype=np.bool_
            ),
        )
    npz_tmp.replace(npz_path)
    return markdown_path, npz_path


__all__ = [
    "RecordedEvalContext",
    "RecordedSegment",
    "deterministic_segment_starts",
    "evaluate_recorded_segments",
    "iter_recorded_segments",
    "load_and_audit_recorded_manifest",
    "load_recorded_eval_context",
    "resolve_feedback_delay",
    "resolve_warmup_samples",
    "validate_resolved_checkpoint",
    "write_recorded_metrics",
]
