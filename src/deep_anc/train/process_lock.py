"""fine-tune watcher/pipeline/train의 중복 실행을 막는 프로세스 lock.

Linux ``flock``은 프로세스 종료 시 커널이 자동 해제한다. lock 파일은 상태 확인을
위해 남기되, 파일의 존재가 아니라 실제 advisory lock 보유 여부만 권한으로 쓴다.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import os
import socket
from pathlib import Path
from typing import IO, Any

from ..config import REPO_ROOT


class LockHeldError(RuntimeError):
    """다른 프로세스가 같은 역할/run lock을 보유하고 있다."""


def resolve_run_dir(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def finetune_run_key(value: str | Path) -> str:
    """호스트 내부의 resolved ckpt_dir에 안정적인 짧은 키를 부여한다."""

    run_dir = resolve_run_dir(value)
    digest = hashlib.sha256(str(run_dir).encode("utf-8")).hexdigest()[:16]
    safe_name = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in run_dir.name
    ).strip("-") or "finetune"
    return f"{safe_name}-{digest}"


def autostart_state_dir(value: str | Path) -> Path:
    return REPO_ROOT / "results" / "finetune_autostart" / finetune_run_key(value)


class ProcessLock:
    """non-blocking exclusive lock context manager with owner metadata."""

    def __init__(self, path: str | Path, *, role: str, metadata: dict | None = None) -> None:
        self.path = Path(path)
        self.role = str(role)
        self.metadata = dict(metadata or {})
        self._handle: IO[str] | None = None

    def acquire(self) -> "ProcessLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip()
            handle.close()
            message = f"이미 실행 중인 {self.role} lock: {self.path}"
            if owner:
                message += f"; owner={owner}"
            raise LockHeldError(message) from exc

        owner: dict[str, Any] = {
            "schema_version": 1,
            "role": self.role,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "acquired_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            **self.metadata,
        }
        handle.seek(0)
        handle.truncate()
        json.dump(owner, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        self._handle = handle
        return self

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "ProcessLock":
        return self.acquire()

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.release()


__all__ = [
    "LockHeldError",
    "ProcessLock",
    "autostart_state_dir",
    "finetune_run_key",
    "resolve_run_dir",
]
