"""fine-tune pipeline 의 단계 상태와 종료 코드 단일 출처.

여기서 만드는 status 는 **관측용(advisory)** 이다. 어떤 준비/완료 게이트도 이 파일을
근거로 생략되지 않는다 — 재개 판단은 항상 디스크 사실(``last.pt`` 존재)에서만 나온다.
status.json 을 지우거나 위조해도 게이트 판정이 바뀌면 안 된다.

종료 코드를 상수로 모으는 이유: 3(자식 train.lock 탈락)과 4(pipeline.lock 탈락)는
운영 대응이 다르다. 4는 "내가 중복 실행했다"(자동화가 조용히 무시해도 된다),
3은 "다른 경로로 이미 학습이 돌고 있다"(조사 대상)이다. 리터럴로 흩어놓으면 이 구분이
금방 무너진다.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

EXIT_OK = 0
EXIT_NOT_READY = 1  # readiness 게이트 FAIL — 설계된 fail-closed 진입 거부
EXIT_CONFIG = 2  # config/경로 오류, best-only 모호 재개
EXIT_TRAIN_LOCKED = 3  # 자식 train.py 가 train.lock 탈락 (scripts/train/train.py 와 같은 값)
EXIT_PIPELINE_LOCKED = 4  # 같은 run 에 다른 pipeline 이 이미 실행 중
EXIT_STAGE_FAILED = 5  # 학습/평가/완료 게이트 실패

PHASES = (
    "init",
    "readiness",
    "not_ready",
    "ready",
    "training",
    "evaluating",
    "completion",
    "done",
    "failed",
    "train_locked",
)

# 재실행마다 반드시 달라지는 값들. idempotency 비교에서 제외한다.
VOLATILE_KEYS = (
    "pid",
    "hostname",
    "started_at_utc",
    "updated_at_utc",
    "finished_at_utc",
    "duration_seconds",
    "generated_at_utc",
)


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_write_text(path: str | Path, content: str) -> None:
    """부분 파일이 관측되지 않도록 tmp → fsync → replace 로 쓴다."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(target)


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def config_fingerprint(cfg: dict) -> str:
    """override 까지 적용된 resolved config 의 정규화 지문."""

    return sha256_text(json.dumps(cfg, ensure_ascii=False, sort_keys=True, default=str))


def stable_view(status: dict | None) -> dict:
    """idempotency 계약의 정의 — 휘발성 키를 제거한 상태.

    ``checked_at_utc``/``pid``/``duration`` 이 매번 다르므로 바이트 동일성은 idempotency
    의 정의가 될 수 없다. 대신 "같은 입력의 재실행이 상태를 파괴하지 않고 같은 판정으로
    수렴한다"를 이 뷰의 동일성으로 정의한다. 테스트와 프로덕션이 같은 함수를 참조하게
    해서 계약이 갈라지지 않게 한다.
    """

    if not status:
        return {}
    view = {k: v for k, v in status.items() if k not in VOLATILE_KEYS}
    readiness = view.get("readiness")
    if isinstance(readiness, dict):
        view["readiness"] = {k: v for k, v in readiness.items() if k != "checked_at_utc"}
    steps = view.get("steps")
    if isinstance(steps, list):
        view["steps"] = [
            {k: v for k, v in step.items() if k not in VOLATILE_KEYS} for step in steps
        ]
    return view


def read_status(path: str | Path) -> dict | None:
    """상태 파일 읽기. 없거나 깨졌으면 None (예외를 던지지 않는다 — advisory 다)."""

    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


class PipelineStatus:
    """단계 전이를 원자적으로 기록한다."""

    def __init__(
        self,
        path: str | Path,
        *,
        mode: str,
        run_key: str,
        run_dir: str | Path,
        state_dir: str | Path,
        lock_path: str | Path,
        config_path: str | Path,
        config_sha256: str,
        overrides: list[str],
        fingerprint: str,
    ) -> None:
        self.path = Path(path)
        self._started = time.monotonic()
        self.data: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "advisory": True,
            "mode": mode,
            "phase": "init",
            "exit_code": None,
            "failed_step": None,
            "run_key": run_key,
            "run_dir": str(run_dir),
            "state_dir": str(state_dir),
            "lock_path": str(lock_path),
            "config_path": str(config_path),
            "config_sha256": config_sha256,
            "overrides": list(overrides),
            "resolved_config_fingerprint": fingerprint,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at_utc": utc_now_iso(),
            "steps": [],
        }
        self._flush()

    def _flush(self) -> None:
        self.data["updated_at_utc"] = utc_now_iso()
        atomic_write_text(
            self.path,
            json.dumps(self.data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )

    def update(self, phase: str | None = None, **fields: Any) -> None:
        if phase is not None:
            if phase not in PHASES:
                raise ValueError(f"알 수 없는 phase: {phase}")
            self.data["phase"] = phase
        self.data.update(fields)
        self._flush()

    def record_step(
        self, name: str, argv: list[str], returncode: int, duration_seconds: float
    ) -> None:
        self.data["steps"].append(
            {
                "name": name,
                "argv": list(argv),
                "returncode": int(returncode),
                "finished_at_utc": utc_now_iso(),
                "duration_seconds": round(float(duration_seconds), 3),
            }
        )
        self._flush()

    def finish(self, phase: str, exit_code: int, **fields: Any) -> int:
        self.data["finished_at_utc"] = utc_now_iso()
        self.data["duration_seconds"] = round(time.monotonic() - self._started, 3)
        self.data["exit_code"] = int(exit_code)
        self.update(phase, **fields)
        return int(exit_code)


__all__ = [
    "EXIT_CONFIG",
    "EXIT_NOT_READY",
    "EXIT_OK",
    "EXIT_PIPELINE_LOCKED",
    "EXIT_STAGE_FAILED",
    "EXIT_TRAIN_LOCKED",
    "PHASES",
    "SCHEMA_VERSION",
    "VOLATILE_KEYS",
    "PipelineStatus",
    "atomic_write_text",
    "config_fingerprint",
    "read_status",
    "sha256_text",
    "stable_view",
    "utc_now_iso",
]
