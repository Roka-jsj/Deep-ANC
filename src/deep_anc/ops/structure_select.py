"""구조 탐색 후보의 승자를 사전 등록 규칙으로 선정한다.

왜 paired 비교인가
------------------
``evaluate_offline.py`` 는 ``metrics.npz`` 에 ``per_item_trusted_db`` 를 저장하고,
모든 후보가 같은 eval seed 로 **동일한 N개 아이템**을 본다. 따라서 후보와 대조군의
차이를 아이템별로 짝지어(paired) 볼 수 있고, 아이템 난이도 분산이 상쇄돼 평균만
비교할 때보다 훨씬 적은 샘플로 유의성을 판단할 수 있다.

왜 last.pt 가 1차 지표인가
--------------------------
``best.pt`` 는 고정 16개 val 배치에서 뽑혀 선택 편향이 있다(HANDOFF 가 "best.pt 만
맹신 금지"를 명시). ``last.pt`` 는 모든 후보가 정확히 같은 step 예산이라 편향이 없다.
``best.pt`` 는 확인 지표로만 쓰고, 1차와 승자가 다르면 ``winner_ambiguous`` 로 판정해
자동 연장을 보류한다.

사전 등록(pre-registration)
---------------------------
이 규칙은 결과를 보기 **전에** 큐 YAML 에 확정한다. 결과를 본 뒤 기준을 바꾸는 것을
구조적으로 막기 위해서다.
"""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]

# 옥타브밴드 표: "| 250 | +8.46 | O |"  (신뢰 표기는 'O' 또는 '낮음*')
_BAND_ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*([+-]?[\d.]+)\s*\|\s*(\S+)\s*\|")
# 소스별 표: "| speech | -25.42 | +25.42 |"
_SOURCE_ROW = re.compile(r"^\|\s*([A-Za-z0-9_]+)\s*\|\s*([+-]?[\d.]+)\s*\|")
# 두 표의 행 모양이 겹치므로(밴드 중심주파수도 \w+ 에 걸린다) 반드시 섹션으로 구분한다.
_BAND_SECTION = "## 기능1"
_SOURCE_SECTION = "## 기능2"

VOLATILE_CONFIG_KEYS = ("ckpt_dir", "resume", "run_until_step", "seed")


def _abs(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_metrics_npz(path: str | Path) -> dict | None:
    """``evaluate_offline.py`` 가 저장한 metrics.npz 를 읽는다."""

    target = _abs(path)
    if not target.exists():
        return None
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError:
        return None
    try:
        with np.load(target, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    except Exception:  # noqa: BLE001 — 손상 파일은 "판독 불가"로 다룬다
        return None


def parse_metrics_markdown(path: str | Path) -> dict:
    """metrics.md 에서 npz 에 없는 소스별·옥타브밴드 지표를 뽑는다.

    ``evaluate_offline.py`` 는 소스별 표와 밴드 표를 Markdown 에만 쓰고 npz 에는
    담지 않는다. 동점 처리 규칙이 이 두 표를 쓰므로 파싱이 필요하다.
    """

    target = _abs(path)
    result: dict[str, Any] = {"per_source_db": {}, "bands": []}
    if not target.exists():
        return result
    section = ""
    for line in target.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            if stripped.startswith(_BAND_SECTION):
                section = "band"
            elif stripped.startswith(_SOURCE_SECTION):
                section = "source"
            else:
                section = ""
            continue
        if section == "band":
            band = _BAND_ROW.match(stripped)
            if band:
                result["bands"].append(
                    {
                        "center_hz": int(band.group(1)),
                        # 이 표는 NMSE 가 아니라 **감쇠**다(양수가 좋다).
                        "attenuation_db": float(band.group(2)),
                        "trusted": band.group(3) == "O",
                    }
                )
        elif section == "source":
            source = _SOURCE_ROW.match(stripped)
            # 두 번째 열은 NMSE 다(음수가 좋다) — 감쇠 열과 부호가 반대이니 섞지 말 것.
            if source and source.group(1) not in {"소스", "source"}:
                result["per_source_db"][source.group(1)] = float(source.group(2))
    return result


def config_fingerprint(run_dir: str | Path) -> str | None:
    """실험 간 코드/설정 동일성 지문.

    ``config_snapshot.yaml`` 에서 휘발성 키(ckpt_dir/resume/run_until_step/seed)를 뺀
    뒤 SHA-256 을 낸다. 후보끼리 지문이 다르면 같은 조건에서 비교한 것이 아니므로
    실격시킨다.
    """

    import hashlib  # noqa: PLC0415

    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        return None
    snapshot = _abs(run_dir) / "config_snapshot.yaml"
    if not snapshot.exists():
        return None
    try:
        cfg = yaml.safe_load(snapshot.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return None
    for key in VOLATILE_CONFIG_KEYS:
        cfg.pop(key, None)
    payload = json.dumps(cfg, ensure_ascii=False, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def bootstrap_ci(
    deltas: Sequence[float], *, resamples: int = 10000, alpha: float = 0.05, seed: int = 20260804
) -> tuple[float, float, float]:
    """paired 차이의 평균과 부트스트랩 신뢰구간. 고정 seed 로 결정적이다."""

    values = [float(v) for v in deltas]
    if not values:
        return 0.0, 0.0, 0.0
    mean = sum(values) / len(values)
    rng = random.Random(seed)
    size = len(values)
    means: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(size):
            total += values[rng.randrange(size)]
        means.append(total / size)
    means.sort()
    low = means[max(0, int(math.floor((alpha / 2) * resamples)))]
    high = means[min(resamples - 1, int(math.ceil((1 - alpha / 2) * resamples)) - 1)]
    return mean, low, high


def _candidate_metrics(run_dir: str | Path, eval_dir: str) -> dict:
    base = _abs(run_dir) / eval_dir
    npz = load_metrics_npz(base / "metrics.npz")
    markdown = parse_metrics_markdown(base / "metrics.md")
    return {"npz": npz, "markdown": markdown, "eval_dir": str(base)}


def _scalar(npz: dict | None, key: str) -> float | None:
    if not npz or key not in npz:
        return None
    try:
        return float(npz[key])
    except (TypeError, ValueError):
        return None


def _per_item(npz: dict | None, key: str) -> list[float] | None:
    if not npz or key not in npz:
        return None
    try:
        return [float(v) for v in npz[key].tolist()]
    except (AttributeError, TypeError, ValueError):
        return None


def evaluate_candidate(
    name: str,
    run_dir: str | Path,
    control: dict,
    *,
    eval_dir: str,
    margin_db: float,
    worsening_tolerance_db: float,
    reference_fingerprint: str | None,
) -> dict:
    """후보 하나를 대조군과 paired 비교하고 실격 여부를 판정한다."""

    metrics = _candidate_metrics(run_dir, eval_dir)
    record: dict[str, Any] = {
        "name": name,
        "run_dir": str(run_dir),
        "eval_dir": metrics["eval_dir"],
        "disqualified": False,
        "reasons": [],
        "config_fingerprint": config_fingerprint(run_dir),
    }

    candidate_items = _per_item(metrics["npz"], "per_item_trusted_db")
    control_items = control.get("per_item_trusted_db")
    if candidate_items is None:
        record["disqualified"] = True
        record["reasons"].append("metrics.npz 의 per_item_trusted_db 를 읽을 수 없음")
        return record
    if control_items is not None and len(candidate_items) != len(control_items):
        record["disqualified"] = True
        record["reasons"].append(
            f"평가 아이템 수가 대조군과 다름: {len(candidate_items)} != {len(control_items)}"
        )
        return record
    if any(not math.isfinite(v) for v in candidate_items):
        record["disqualified"] = True
        record["reasons"].append("per_item_trusted_db 에 비유한 값이 있음")
        return record

    if reference_fingerprint and record["config_fingerprint"] != reference_fingerprint:
        record["disqualified"] = True
        record["reasons"].append(
            f"config fingerprint 불일치: {record['config_fingerprint']} != {reference_fingerprint}"
        )

    record["trusted_db"] = sum(candidate_items) / len(candidate_items)
    record["fullband_db"] = _scalar(metrics["npz"], "nmse_fullband_db")
    record["heldout_trusted_db"] = _scalar(metrics["npz"], "nmse_heldout_trusted_db")

    if control_items is not None:
        deltas = [c - b for c, b in zip(candidate_items, control_items)]
        mean, low, high = bootstrap_ci(deltas)
        record["mean_delta_db"] = mean
        record["ci95_db"] = [low, high]
        record["significant"] = high < -abs(margin_db)
    else:
        record["mean_delta_db"] = None
        record["ci95_db"] = None
        record["significant"] = False

    # do-no-harm: trusted 를 얻자고 fullband 나 학습 그리드 밖 일반화를 망치면 안 된다.
    for key, label in (("fullband_db", "fullband"), ("heldout_trusted_db", "held-out")):
        mine, theirs = record.get(key), control.get(key)
        if mine is None or theirs is None:
            continue
        if mine > theirs + worsening_tolerance_db:
            record["disqualified"] = True
            record["reasons"].append(
                f"{label} NMSE 가 대조군 대비 {mine - theirs:.2f}dB 악화 "
                f"(허용 {worsening_tolerance_db:.2f}dB)"
            )

    per_source = metrics["markdown"]["per_source_db"]
    record["per_source_db"] = per_source
    record["worst_source_db"] = max(per_source.values()) if per_source else None
    trusted_bands = [b for b in metrics["markdown"]["bands"] if b["trusted"]]
    record["trusted_bands_nonpositive"] = sum(
        1 for b in trusted_bands if b["attenuation_db"] <= 0.0
    )
    return record


def decide_structure_winner(
    decision: dict, *, log: Callable[[str], None] | None = None
) -> dict:
    """사전 등록 규칙으로 승자를 정한다.

    ``decision`` 키:
      ``control``       {name, run_dir}
      ``candidates``    [{name, run_dir}, ...]
      ``primary_eval``  기본 ``eval_pilot_last`` (선택 편향 없음 — 1차 지표)
      ``confirm_eval``  기본 ``eval_pilot_best`` (확인 지표)
      ``margin_db``     기본 0.30 — 이보다 작은 차이는 잡음으로 본다
    """

    emit = log or (lambda message: None)
    control_spec = decision.get("control") or {}
    candidates = list(decision.get("candidates") or [])
    primary_eval = str(decision.get("primary_eval", "eval_pilot_last"))
    confirm_eval = str(decision.get("confirm_eval", "eval_pilot_best"))
    margin = float(decision.get("margin_db", 0.30))
    tolerance = float(decision.get("worsening_tolerance_db", 1.0))

    if not control_spec.get("run_dir"):
        raise ValueError("decision.control.run_dir 이 필요합니다")

    def _rank(eval_dir: str) -> tuple[dict, list[dict]]:
        control_metrics = _candidate_metrics(control_spec["run_dir"], eval_dir)
        items = _per_item(control_metrics["npz"], "per_item_trusted_db")
        control_markdown = control_metrics["markdown"]
        control = {
            "name": control_spec.get("name", "control"),
            "run_dir": str(control_spec["run_dir"]),
            "per_item_trusted_db": items,
            "trusted_db": (sum(items) / len(items)) if items else None,
            "fullband_db": _scalar(control_metrics["npz"], "nmse_fullband_db"),
            "heldout_trusted_db": _scalar(control_metrics["npz"], "nmse_heldout_trusted_db"),
            "per_source_db": control_markdown["per_source_db"],
            "worst_source_db": (
                max(control_markdown["per_source_db"].values())
                if control_markdown["per_source_db"]
                else None
            ),
            "trusted_bands_nonpositive": sum(
                1 for b in control_markdown["bands"] if b["trusted"] and b["attenuation_db"] <= 0.0
            ),
            "config_fingerprint": config_fingerprint(control_spec["run_dir"]),
            "disqualified": False,
            "reasons": [],
            "significant": False,
            "mean_delta_db": 0.0,
        }
        ranked = [
            evaluate_candidate(
                str(entry["name"]),
                entry["run_dir"],
                control,
                eval_dir=eval_dir,
                margin_db=margin,
                worsening_tolerance_db=tolerance,
                reference_fingerprint=control["config_fingerprint"],
            )
            for entry in candidates
        ]
        ranked.sort(key=lambda r: (r["mean_delta_db"] if r["mean_delta_db"] is not None else 0.0))
        return control, ranked

    def _winner(control: dict, ranked: list[dict]) -> dict:
        viable = [r for r in ranked if not r["disqualified"] and r["significant"]]
        if not viable:
            # 유의한 개선이 없으면 가장 싸고 Jetson P99 위험이 낮은 대조군을 유지한다.
            return control
        best = viable[0]
        close = [r for r in viable if abs(r["mean_delta_db"] - best["mean_delta_db"]) <= margin]
        if len(close) > 1:
            # 동점 처리 — 절대 목표에 직접 매핑한다.
            # ① 기능2(모든 소리): 최악 소스가 가장 좋은 것. 평균이 아니라 최악값 문제다.
            # ② 기능1(저역+고역): trusted 밴드 중 감쇠 <= 0 인 밴드가 적은 것.
            close.sort(
                key=lambda r: (
                    r["worst_source_db"] if r["worst_source_db"] is not None else 0.0,
                    r["trusted_bands_nonpositive"],
                    r["mean_delta_db"],
                )
            )
            best = close[0]
        return best

    control_primary, ranked_primary = _rank(primary_eval)
    primary = _winner(control_primary, ranked_primary)
    control_confirm, ranked_confirm = _rank(confirm_eval)
    confirm = _winner(control_confirm, ranked_confirm)

    ambiguous = primary["name"] != confirm["name"]
    verdict: dict[str, Any] = {
        "rule": (
            f"paired held-out trusted NMSE({primary_eval}) vs {control_primary['name']}, "
            f"bootstrap95, margin {margin:.2f}dB; {confirm_eval} 로 확인"
        ),
        "primary_eval": primary_eval,
        "confirm_eval": confirm_eval,
        "control": control_primary["name"],
        "ranking": ranked_primary,
        "confirm_ranking": ranked_confirm,
        "primary_winner": primary["name"],
        "confirm_winner": confirm["name"],
        "ambiguous": ambiguous,
        "winner": None if ambiguous else primary["name"],
        "winner_run_dir": None if ambiguous else primary["run_dir"],
        "caveat": (
            "이 신뢰구간은 평가 아이템 간 분산만 덮는다. run 간(seed) 분산은 덮지 않으므로 "
            "seed 반복 결과가 나오기 전에는 확정적 우열 주장으로 쓰지 않는다."
        ),
    }
    if ambiguous:
        verdict["summary"] = (
            f"winner_ambiguous: {primary_eval}={primary['name']} 이지만 "
            f"{confirm_eval}={confirm['name']} — 연장을 자동 시작하지 않는다"
        )
    else:
        verdict["summary"] = (
            f"승자 {primary['name']} (Δ {primary.get('mean_delta_db')}, "
            f"CI {primary.get('ci95_db')})"
        )
    emit(verdict["summary"])
    return verdict


__all__ = [
    "bootstrap_ci",
    "config_fingerprint",
    "decide_structure_winner",
    "evaluate_candidate",
    "load_metrics_npz",
    "parse_metrics_markdown",
]
