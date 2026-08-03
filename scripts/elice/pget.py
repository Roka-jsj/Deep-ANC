#!/usr/bin/env python3
"""병렬 범위(Range) 다운로더 — 연결당 속도 제한 우회.

사용법: ``pget.py URL OUT [N]``
"""

from __future__ import annotations

import errno
import fcntl
import os
import re
import sys
import threading
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator


MAX_ATTEMPTS = 8
RETRY_DELAY_SECONDS = 2.0
REQUEST_TIMEOUT_SECONDS = 60
PROGRESS_INTERVAL_SECONDS = 5.0
CHUNK_SIZE = 1 << 20


class DownloadError(RuntimeError):
    """다운로드를 완전한 파일로 마칠 수 없을 때 발생한다."""


@dataclass(frozen=True)
class _RemoteObject:
    total: int
    etag: str | None
    last_modified: str | None

    @property
    def if_range(self) -> str | None:
        # RFC 7233의 If-Range에는 weak ETag를 사용할 수 없다.
        if self.etag is not None and not self.etag.upper().startswith("W/"):
            return self.etag
        return self.last_modified


def _header(response: object, name: str) -> str | None:
    """HTTPMessage뿐 아니라 테스트의 단순 dict도 대소문자 없이 읽는다."""

    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get(name)
    if value is None:
        for key, candidate in headers.items():
            if key.lower() == name.lower():
                value = candidate
                break
    if value is None:
        return None
    return str(value).strip()


@contextmanager
def _exclusive_output_lock(output_path: Path) -> Iterator[None]:
    """같은 출력 경로의 다른 pget을 비차단 방식으로 거부한다.

    잠금 파일은 의도적으로 지우지 않는다. 프로세스가 죽으면 커널이 flock을
    자동 해제하므로 남은 파일은 곧바로 재사용할 수 있다. 반대로 unlock 뒤
    파일을 지우면 다른 프로세스가 이미 잡은 inode와 새 inode에 잠금이 갈라질
    수 있어 활성 잠금을 훼손하게 된다.
    """

    lock_path = Path(f"{output_path}.part.lock")
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise DownloadError(f"출력 잠금 파일 열기 실패: {lock_path}: {exc}") from exc

    locked = False
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise DownloadError(
                    f"같은 출력 경로의 다운로드가 이미 실행 중입니다: {output_path}"
                ) from exc
            raise DownloadError(f"출력 잠금 획득 실패: {lock_path}: {exc}") from exc
        yield
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _retry_pause(attempt: int, attempts: int, delay: float) -> None:
    if attempt + 1 < attempts and delay > 0:
        time.sleep(delay)


def _content_length(
    url: str,
    opener: Callable[..., object],
    attempts: int,
    retry_delay: float,
) -> _RemoteObject:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, method="HEAD")
            with opener(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                value = _header(response, "Content-Length")
                etag = _header(response, "ETag")
                last_modified = _header(response, "Last-Modified")
            if value is None:
                raise DownloadError("HEAD 응답에 Content-Length가 없습니다")
            total = int(value)
            if total < 0:
                raise DownloadError(f"잘못된 Content-Length: {value}")
            return _RemoteObject(total, etag, last_modified)
        except Exception as exc:
            last_error = exc
            print(f"[HEAD] retry {attempt + 1}/{attempts}: {exc}", flush=True)
            _retry_pause(attempt, attempts, retry_delay)
    raise DownloadError(f"HEAD 재시도 소진: {last_error}")


def _status_code(response: object) -> int | None:
    status = getattr(response, "status", None)
    if status is None:
        getcode = getattr(response, "getcode", None)
        if getcode is not None:
            status = getcode()
    return status


def _validate_range_response(
    response: object,
    start: int,
    end: int,
    remote: _RemoteObject,
) -> None:
    status = _status_code(response)
    if status != 206:
        raise DownloadError(
            f"서버가 Range 요청을 무시했습니다 (HTTP {status}, 206 필요)"
        )

    value = _header(response, "Content-Range")
    match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+)", value or "", re.I)
    if match is None:
        raise DownloadError(f"잘못된 Content-Range: {value!r}")
    actual = tuple(int(part) for part in match.groups())
    if actual != (start, end, remote.total):
        raise DownloadError(
            f"Content-Range 불일치: {value!r}, "
            f"bytes {start}-{end}/{remote.total} 필요"
        )

    response_etag = _header(response, "ETag")
    if (
        remote.etag is not None
        and response_etag is not None
        and response_etag != remote.etag
    ):
        raise DownloadError(
            f"ETag 불일치: HEAD={remote.etag!r}, Range={response_etag!r}"
        )
    response_modified = _header(response, "Last-Modified")
    if (
        remote.last_modified is not None
        and response_modified is not None
        and response_modified != remote.last_modified
    ):
        raise DownloadError(
            "Last-Modified 불일치: "
            f"HEAD={remote.last_modified!r}, Range={response_modified!r}"
        )


def _remove_partial(part_path: Path) -> None:
    try:
        part_path.unlink(missing_ok=True)
    except OSError as exc:
        print(f"[WARN] 임시 파일 삭제 실패: {part_path}: {exc}", flush=True)


def _download_locked(
    url: str,
    output_path: Path,
    connections: int,
    *,
    opener: Callable[..., object],
    attempts: int,
    retry_delay: float,
    progress_interval: float,
) -> None:
    part_path = Path(f"{output_path}.part")
    remote = _content_length(url, opener, attempts, retry_delay)
    total = remote.total
    print(f"total {total / 1e9:.2f} GB, {connections} connections", flush=True)

    try:
        with part_path.open("wb") as partial:
            partial.truncate(total)
    except OSError as exc:
        raise DownloadError(f"임시 파일 생성 실패: {part_path}: {exc}") from exc

    done = [0] * connections
    errors: list[str | None] = [None] * connections

    def worker(index: int) -> None:
        low = total * index // connections
        high = total * (index + 1) // connections - 1
        expected = high - low + 1
        if expected <= 0:
            return

        last_error: Exception | None = None
        for attempt in range(attempts):
            connection_start = done[index]
            start = low + connection_start
            try:
                headers = {"Range": f"bytes={start}-{high}"}
                if remote.if_range is not None:
                    headers["If-Range"] = remote.if_range
                request = urllib.request.Request(url, headers=headers)
                with opener(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                    _validate_range_response(response, start, high, remote)
                    with part_path.open("r+b") as partial:
                        partial.seek(start)
                        remaining = expected - done[index]
                        while True:
                            # remaining+1을 허용해 요청 범위보다 많은 응답도 검출한다.
                            chunk = response.read(min(CHUNK_SIZE, remaining + 1))
                            if not chunk:
                                if remaining:
                                    raise DownloadError(
                                        f"범위 {low}-{high} 응답 부족 "
                                        f"({done[index]}/{expected} bytes)"
                                    )
                                return
                            if len(chunk) > remaining:
                                done[index] = connection_start
                                raise DownloadError(
                                    f"범위 {low}-{high} 응답 초과 "
                                    f"({len(chunk)} > 남은 {remaining} bytes)"
                                )
                            written = partial.write(chunk)
                            if written != len(chunk):
                                raise OSError(
                                    f"임시 파일 쓰기 부족 ({written}/{len(chunk)} bytes)"
                                )
                            done[index] += written
                            remaining -= written
            except Exception as exc:
                # 필요한 바이트를 다 받았어도 EOF 확인에 실패했다면 그 연결분은
                # 검증되지 않은 것이므로 다음 시도에서 다시 받는다.
                if done[index] == expected:
                    done[index] = connection_start
                last_error = exc
                print(
                    f"[{index}] retry {attempt + 1}/{attempts}: {exc}", flush=True
                )
                _retry_pause(attempt, attempts, retry_delay)

        errors[index] = f"범위 {low}-{high} 재시도 소진: {last_error}"

    threads = [
        threading.Thread(target=worker, args=(index,), daemon=True)
        for index in range(connections)
    ]
    started = time.monotonic()
    for thread in threads:
        thread.start()

    while any(thread.is_alive() for thread in threads):
        deadline = time.monotonic() + progress_interval
        for thread in threads:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(timeout=remaining)
        received = sum(done)
        elapsed = max(1.0, time.monotonic() - started)
        print(
            f"{received / 1e9:.2f}/{total / 1e9:.2f} GB  "
            f"{received / 1e6 / elapsed:.1f} MB/s",
            flush=True,
        )

    received = sum(done)
    failures = [error for error in errors if error is not None]
    if failures or received != total:
        _remove_partial(part_path)
        detail = "; ".join(failures) if failures else "스레드 완료 후 바이트 수 불일치"
        raise DownloadError(f"다운로드 실패 ({received}/{total} bytes): {detail}")

    try:
        with part_path.open("r+b") as partial:
            partial.flush()
            os.fsync(partial.fileno())
        os.replace(part_path, output_path)
    except OSError as exc:
        _remove_partial(part_path)
        raise DownloadError(f"완성 파일 교체 실패: {exc}") from exc
    print("DONE", flush=True)


def download(
    url: str,
    output: str | os.PathLike[str],
    connections: int = 16,
    *,
    opener: Callable[..., object] | None = None,
    attempts: int | None = None,
    retry_delay: float | None = None,
    progress_interval: float | None = None,
) -> None:
    """URL을 병렬 Range 요청으로 받아 성공 시에만 *output*으로 교체한다."""

    if connections <= 0:
        raise DownloadError("연결 수 N은 1 이상이어야 합니다")
    attempts = MAX_ATTEMPTS if attempts is None else attempts
    retry_delay = RETRY_DELAY_SECONDS if retry_delay is None else retry_delay
    progress_interval = (
        PROGRESS_INTERVAL_SECONDS if progress_interval is None else progress_interval
    )
    if attempts <= 0:
        raise DownloadError("재시도 횟수는 1 이상이어야 합니다")
    if progress_interval <= 0:
        raise DownloadError("진행 로그 주기는 0보다 커야 합니다")
    if opener is None:
        opener = urllib.request.urlopen

    output_path = Path(output)
    with _exclusive_output_lock(output_path):
        _download_locked(
            url,
            output_path,
            connections,
            opener=opener,
            attempts=attempts,
            retry_delay=retry_delay,
            progress_interval=progress_interval,
        )


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) not in (2, 3):
        print("사용법: pget.py URL OUT [N]", file=sys.stderr)
        return 2
    try:
        connections = int(args[2]) if len(args) == 3 else 16
        download(args[0], args[1], connections)
    except (ValueError, DownloadError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
