"""측정 산출물 표준 — 모든 측정 도구가 같은 형식으로 파일을 남긴다.

왜 필요한가
-----------
실기 측정은 스피커를 울려야 다시 얻을 수 있다. 그런데 지금까지 결과가 stdout 로그와
markdown 표에만 남아서, 나중에 다시 계산하거나 다른 실행과 비교하려면 사람이 표를 눈으로
읽어 옮겨야 했다. 실제로 6분짜리 실기 세션의 리포트를 마지막 write 오류로 잃은 적도 있다.

그래서 산출물을 다음 4종으로 고정한다. 어느 도구든 같은 규칙이다.

    <run_dir>/metrics.csv     기계가 읽는 단일 표 (한 행 = 한 측정 단위)
    <run_dir>/summary.md      사람이 읽는 요약 (같은 수치, 판정 포함)
    <run_dir>/raw/*.npz       원신호·중간 배열 (재계산용, 손실 없음)
    <run_dir>/wav/*.wav       들어볼 수 있는 신호 (해당되는 도구만)

CSV 가 단일 출처다. summary.md 는 CSV 에서 파생되며, 둘이 어긋나면 CSV 를 믿는다.
"""

from __future__ import annotations

import csv
import json
import os
import wave
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "atomic_write_bytes",
    "atomic_write_text",
    "run_directory",
    "write_csv",
    "write_json",
    "write_series_csv",
    "write_wav",
    "write_wav_pair",
]


def atomic_write_bytes(path: Path, payload: bytes) -> Path:
    """tmp → fsync → replace. 부분 기록된 파일이 관측되지 않게 한다."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


def atomic_write_text(path: Path, text: str) -> Path:
    return atomic_write_bytes(Path(path), text.encode("utf-8"))


def run_directory(root: str | Path, prefix: str, stamp: str) -> Path:
    """``<root>/<prefix>_<stamp>/`` 를 만들고 raw/ wav/ 를 준비한다.

    stamp 를 인자로 받는 것은 의도적이다 — 같은 실행의 여러 산출물이 서로 다른 초에
    걸쳐 만들어지면 디렉터리가 갈라진다.
    """

    run_dir = Path(root) / f"{prefix}_{stamp}"
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    (run_dir / "wav").mkdir(parents=True, exist_ok=True)
    return run_dir


def _csv_value(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        # 목록은 CSV 한 칸에 넣지 않는다 — write_series_csv 로 긴 형식이 되어야 한다.
        return json.dumps(np.asarray(value).tolist(), ensure_ascii=False)
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], *,
              columns: Sequence[str] | None = None) -> Path:
    """딕셔너리 목록을 CSV 로 쓴다. 열 순서는 첫 행 순서(또는 columns)로 고정한다."""

    rows = list(rows)
    if not rows:
        raise ValueError("빈 행으로 CSV 를 쓰지 않는다 — 측정 실패를 성공처럼 남기게 된다")
    if columns is None:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        columns = seen
    buffer: list[str] = []
    import io

    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _csv_value(row.get(key)) for key in columns})
    buffer.append(stream.getvalue())
    return atomic_write_text(path, "".join(buffer))


def write_series_csv(path: Path, series: Mapping[str, Sequence[Any]], *,
                     index_name: str = "index") -> Path:
    """같은 길이의 계열들을 긴 형식(long) CSV 로 쓴다.

    반복별 onset, 주기별 지연처럼 "판정에 쓰인 개별 관측치"는 요약값과 함께 반드시 남긴다.
    유효한 것만 남기면 걸러낸 관측에 나쁜 소식이 숨는다.
    """

    names = list(series)
    if not names:
        raise ValueError("빈 계열로 CSV 를 쓰지 않는다")
    lengths = {len(np.atleast_1d(np.asarray(series[name], dtype=object))) for name in names}
    if len(lengths) != 1:
        raise ValueError(f"계열 길이가 다릅니다: { {n: len(series[n]) for n in names} }")
    count = lengths.pop()
    rows = [
        {index_name: i, **{name: series[name][i] for name in names}}
        for i in range(count)
    ]
    return write_csv(path, rows, columns=[index_name, *names])


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    def default(value: Any) -> Any:
        if isinstance(value, (np.floating, np.integer, np.bool_)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"JSON 으로 직렬화할 수 없습니다: {type(value)!r}")

    text = json.dumps(payload, ensure_ascii=False, indent=2, default=default)
    return atomic_write_text(path, text + "\n")


def write_wav(path: Path, signal: np.ndarray, sample_rate: int, *,
              scale: float = 1.0) -> Path:
    """float 신호를 16-bit PCM WAV 로 쓴다. ``scale`` 은 호출자가 정한 공통 배율이다.

    파일마다 정규화하지 않는 것이 핵심이다 — OFF/ON 을 각각 정규화하면 크기 차이가
    사라져 상쇄가 귀에서 없어진다.
    """

    values = np.asarray(signal, dtype=np.float64) * float(scale)
    clipped = np.clip(values, -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype("<i2")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with wave.open(str(tmp), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
    os.replace(tmp, path)
    return path


def write_wav_pair(directory: Path, name: str, off: np.ndarray, on: np.ndarray,
                   sample_rate: int, *, headroom: float = 0.9) -> dict[str, Path]:
    """OFF/ON/AB 3종을 **같은 배율**로 쓴다. 배율은 OFF 피크 기준이다.

    AB 는 앞 절반 OFF, 뒤 절반 ON 을 이어 붙인 한 파일이다 — 두 파일을 번갈아 여는 것보다
    차이가 훨씬 잘 들린다.
    """

    off = np.asarray(off, dtype=np.float64)
    on = np.asarray(on, dtype=np.float64)
    scale = float(headroom) / max(float(np.max(np.abs(off))), 1e-12)
    directory = Path(directory)
    length = min(off.size, on.size)
    half = length // 2
    ab = np.concatenate([off[:half], on[half:length]])
    return {
        "off": write_wav(directory / f"{name}_off.wav", off, sample_rate, scale=scale),
        "on": write_wav(directory / f"{name}_on.wav", on, sample_rate, scale=scale),
        "ab": write_wav(directory / f"{name}_ab.wav", ab, sample_rate, scale=scale),
    }


def markdown_table(rows: Sequence[Mapping[str, Any]],
                   columns: Sequence[str],
                   *, align: Mapping[str, str] | None = None) -> Iterable[str]:
    """CSV 와 같은 행/열로 markdown 표를 만든다 — 두 산출물이 갈라지지 않게."""

    align = align or {}
    yield "| " + " | ".join(columns) + " |"
    yield "|" + "|".join(align.get(column, "---") for column in columns) + "|"
    for row in rows:
        yield "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
