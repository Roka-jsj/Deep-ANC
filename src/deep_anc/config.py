"""YAML 설정 로드/병합/검증.

모든 스크립트는 이 모듈을 통해 설정을 읽는다. 설정 파일 간 참조
(train_*.yaml 의 model_config / data_config / duct_config)는 여기서 해석한다.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# 저장소 루트 (src/deep_anc/config.py 기준 두 단계 위)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path: str | Path) -> Path:
    """상대 경로는 저장소 루트 기준으로 해석한다 (실행 위치와 무관하게 동작)."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        candidate = Path.cwd() / p
        p = candidate if candidate.exists() else REPO_ROOT / p
    return p


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = _resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"설정 파일이 없습니다: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"{p}: 최상위는 매핑(dict)이어야 합니다")
    return data


def deep_merge(base: dict, override: dict) -> dict:
    """중첩 dict 병합 — override 우선. 리스트는 통째로 교체."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """'a.b.c=value' 형태의 CLI 오버라이드 적용."""
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"오버라이드 형식 오류 (key=value): {item}")
        key, _, raw = item.partition("=")
        value = yaml.safe_load(raw)
        node = out
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def load_train_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """학습 설정 로드 + 참조된 model/data/duct 설정을 함께 해석."""
    cfg = load_yaml(path)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    cfg["model"] = load_yaml(cfg["model_config"])
    cfg["data"] = load_yaml(cfg["data_config"])
    cfg["duct"] = load_yaml(cfg["duct_config"])
    validate_duct(cfg["duct"])
    return cfg


def load_runtime_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    cfg = load_yaml(path)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    cfg["hardware"] = load_yaml(cfg["hardware_config"])
    cfg["duct"] = load_yaml(cfg["duct_config"])
    return cfg


def validate_duct(duct: dict) -> list[str]:
    """duct.yaml 의 미기입(null) 항목을 경고 목록으로 반환 (치명 오류는 예외)."""
    warnings: list[str] = []
    positions = duct.get("positions_m", {})
    for name in ("noise_speaker", "reference_mic", "cancel_speaker", "error_mic"):
        if positions.get(name) is None:
            warnings.append(f"duct.yaml positions_m.{name} 이 비어 있습니다 — 시뮬레이션 정확도에 영향")
    digital = duct.get("digital_reference", {})
    if digital.get("d_noise_delay_samples") is None:
        warnings.append(
            "duct.yaml digital_reference.d_noise_delay_samples 미실측 — "
            "덕트 기하로부터의 추정값을 사용합니다 (scripts/data/calibrate_wideband.py 로 실측 권장)"
        )
    for w in warnings:
        print(f"[duct.yaml 경고] {w}")
    return warnings


def duct_distance_samples(duct: dict, a: str, b: str, sample_rate: int) -> int:
    """두 장비 위치 간 음향 전파 지연(샘플). a, b는 positions_m 키."""
    pos = duct["positions_m"]
    if pos.get(a) is None or pos.get(b) is None:
        raise ValueError(f"duct.yaml positions_m 에 {a}/{b} 값이 필요합니다")
    c = float(duct["duct"]["speed_of_sound_mps"])
    dist = abs(float(pos[a]) - float(pos[b]))
    return int(round(sample_rate * dist / c))


def default_d_noise_delay(duct: dict, sample_rate: int, s_path_delay: int) -> int:
    """digital-ref 1차경로 순수지연 기본값 (미실측 시).

    소음(ch0)과 상쇄(ch1)는 같은 USB 출력 장치를 쓰므로 전기/버퍼 지연이 공통이다.
    측정된 S(z) 지연 = 공통지연 + t_ac(CS→ERR) 이므로,
        D_noise ≈ s_path_delay − t_ac(CS→ERR) + t_ac(NS→ERR)
    (근거: docs/01_physics_limits.md, 교차검증 C2)
    """
    t_cs_err = duct_distance_samples(duct, "cancel_speaker", "error_mic", sample_rate)
    t_ns_err = duct_distance_samples(duct, "noise_speaker", "error_mic", sample_rate)
    return int(s_path_delay - t_cs_err + t_ns_err)
