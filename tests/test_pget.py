"""scripts/elice/pget.py의 재시도와 완전성 검증."""

from __future__ import annotations

import io
import re
import threading

from scripts.elice import pget


class _Response:
    def __init__(self, body: bytes, headers: dict[str, str], status: int = 200):
        self._body = io.BytesIO(body)
        self.headers = headers
        self.status = status

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class _FakeUrlopen:
    def __init__(
        self,
        payload: bytes,
        *,
        head_failures: int = 0,
        incomplete_ranges: bool = False,
        head_etag: str | None = None,
        range_etag: str | None = None,
        last_modified: str | None = None,
    ):
        self.payload = payload
        self.head_failures = head_failures
        self.incomplete_ranges = incomplete_ranges
        self.head_etag = head_etag
        self.range_etag = head_etag if range_etag is None else range_etag
        self.last_modified = last_modified
        self.head_calls = 0
        self.if_ranges: list[str | None] = []
        self._lock = threading.Lock()

    def __call__(self, request, timeout=None):
        if request.get_method() == "HEAD":
            with self._lock:
                self.head_calls += 1
                call = self.head_calls
            if call <= self.head_failures:
                raise OSError("temporary SSL EOF")
            headers = {"Content-Length": str(len(self.payload))}
            if self.head_etag is not None:
                headers["ETag"] = self.head_etag
            if self.last_modified is not None:
                headers["Last-Modified"] = self.last_modified
            return _Response(b"", headers, status=200)

        value = request.get_header("Range")
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", value or "")
        assert match is not None
        request_headers = {key.lower(): value for key, value in request.header_items()}
        with self._lock:
            self.if_ranges.append(request_headers.get("if-range"))
        start, end = (int(part) for part in match.groups())
        body = self.payload[start : end + 1]
        if self.incomplete_ranges and body:
            body = body[:-1]
        headers = {"Content-Range": f"bytes {start}-{end}/{len(self.payload)}"}
        if self.range_etag is not None:
            headers["ETag"] = self.range_etag
        if self.last_modified is not None:
            headers["Last-Modified"] = self.last_modified
        return _Response(
            body,
            headers,
            status=206,
        )


class _BlockingHeadUrlopen(_FakeUrlopen):
    def __init__(self, payload: bytes):
        super().__init__(payload)
        self.entered = threading.Event()
        self.release = threading.Event()
        self._blocked_once = False
        self._block_lock = threading.Lock()

    def __call__(self, request, timeout=None):
        if request.get_method() == "HEAD":
            with self._block_lock:
                should_block = not self._blocked_once
                self._blocked_once = True
            if should_block:
                self.entered.set()
                if not self.release.wait(timeout=5):
                    raise TimeoutError("test did not release blocked HEAD")
        return super().__call__(request, timeout=timeout)


def test_success_uses_part_then_atomic_replace(tmp_path):
    payload = bytes(range(251)) * 17
    output = tmp_path / "archive.bin"
    fake = _FakeUrlopen(payload)

    pget.download(
        "https://example.invalid/archive.bin",
        output,
        connections=4,
        opener=fake,
        attempts=2,
        retry_delay=0,
        progress_interval=0.01,
    )

    assert output.read_bytes() == payload
    assert not (tmp_path / "archive.bin.part").exists()


def test_head_content_length_is_retried(tmp_path):
    payload = b"head retry payload"
    output = tmp_path / "retry.bin"
    fake = _FakeUrlopen(payload, head_failures=1)

    pget.download(
        "https://example.invalid/retry.bin",
        output,
        connections=2,
        opener=fake,
        attempts=3,
        retry_delay=0,
        progress_interval=0.01,
    )

    assert fake.head_calls == 2
    assert output.read_bytes() == payload


def test_incomplete_ranges_make_cli_nonzero_and_leave_no_output(
    tmp_path, monkeypatch
):
    output = tmp_path / "incomplete.bin"
    fake = _FakeUrlopen(b"0123456789abcdef", incomplete_ranges=True)
    monkeypatch.setattr(pget.urllib.request, "urlopen", fake)
    monkeypatch.setattr(pget, "MAX_ATTEMPTS", 2)
    monkeypatch.setattr(pget, "RETRY_DELAY_SECONDS", 0)
    monkeypatch.setattr(pget, "PROGRESS_INTERVAL_SECONDS", 0.01)

    result = pget.main(
        ["https://example.invalid/incomplete.bin", str(output), "2"]
    )

    assert result != 0
    assert not output.exists()
    assert not (tmp_path / "incomplete.bin.part").exists()


def test_same_output_concurrent_cli_is_rejected(tmp_path, monkeypatch):
    output = tmp_path / "shared.bin"
    fake = _BlockingHeadUrlopen(b"one writer only")
    first_errors = []

    def first_download():
        try:
            pget.download(
                "https://example.invalid/shared.bin",
                output,
                connections=2,
                opener=fake,
                attempts=1,
                retry_delay=0,
                progress_interval=0.01,
            )
        except Exception as exc:  # pragma: no cover - assertion reports the value
            first_errors.append(exc)

    first = threading.Thread(target=first_download)
    first.start()
    assert fake.entered.wait(timeout=2)
    monkeypatch.setattr(pget.urllib.request, "urlopen", fake)

    try:
        result = pget.main(
            ["https://example.invalid/shared.bin", str(output), "2"]
        )
        assert result != 0
        assert not output.exists()
    finally:
        fake.release.set()
        first.join(timeout=5)

    assert not first.is_alive()
    assert first_errors == []
    assert output.read_bytes() == b"one writer only"


def test_unlocked_stale_lock_file_is_safely_reused(tmp_path):
    output = tmp_path / "stale.bin"
    lock = tmp_path / "stale.bin.part.lock"
    lock.touch()

    pget.download(
        "https://example.invalid/stale.bin",
        output,
        connections=1,
        opener=_FakeUrlopen(b"stale lock is only an inode"),
        attempts=1,
        retry_delay=0,
        progress_interval=0.01,
    )

    assert output.read_bytes() == b"stale lock is only an inode"
    assert lock.exists()


def test_etag_mismatch_sends_if_range_and_makes_cli_nonzero(
    tmp_path, monkeypatch
):
    output = tmp_path / "changed.bin"
    fake = _FakeUrlopen(
        b"object changed during download",
        head_etag='"version-1"',
        range_etag='"version-2"',
    )
    monkeypatch.setattr(pget.urllib.request, "urlopen", fake)
    monkeypatch.setattr(pget, "MAX_ATTEMPTS", 1)
    monkeypatch.setattr(pget, "RETRY_DELAY_SECONDS", 0)
    monkeypatch.setattr(pget, "PROGRESS_INTERVAL_SECONDS", 0.01)

    result = pget.main(
        ["https://example.invalid/changed.bin", str(output), "1"]
    )

    assert result != 0
    assert fake.if_ranges == ['"version-1"']
    assert not output.exists()
    assert not (tmp_path / "changed.bin.part").exists()
