"""측정 산출물 표준(`deep_anc.eval.artifacts`)의 계약을 고정한다.

이 모듈이 지키는 약속은 두 가지이고, 둘 다 실기 사고에서 나왔다.

1. **부분 기록된 파일이 관측되지 않는다.** 측정은 스피커를 다시 울려야 얻으므로,
   쓰다 만 CSV 를 다음 실행이 읽고 "측정했다"고 판단하면 안 된다.
2. **OFF/ON WAV 는 같은 배율로 쓴다.** 파일마다 정규화하면 크기 차이가 사라져
   상쇄 효과가 귀에서 없어진다 — 들어서 확인하는 목적 자체가 무너진다.
"""

from __future__ import annotations

import csv
import json
import wave

import numpy as np
import pytest

from deep_anc.eval.artifacts import (
    atomic_write_text,
    markdown_table,
    run_directory,
    write_csv,
    write_json,
    write_series_csv,
    write_wav,
    write_wav_pair,
)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_run_directory_prepares_raw_and_wav(tmp_path):
    run_dir = run_directory(tmp_path, "session", "20260804_120000")
    assert run_dir.name == "session_20260804_120000"
    assert (run_dir / "raw").is_dir()
    assert (run_dir / "wav").is_dir()


def test_run_directory_is_idempotent(tmp_path):
    first = run_directory(tmp_path, "session", "stamp")
    (first / "raw" / "keep.npz").write_bytes(b"x")
    second = run_directory(tmp_path, "session", "stamp")
    assert first == second
    assert (second / "raw" / "keep.npz").exists(), "재호출이 기존 산출물을 지우면 안 된다"


def test_atomic_write_leaves_no_partial_file(tmp_path):
    target = tmp_path / "deep" / "summary.md"
    atomic_write_text(target, "본문")
    assert target.read_text(encoding="utf-8") == "본문"
    assert not list(tmp_path.rglob("*.tmp")), "임시 파일이 남으면 안 된다"


def test_write_csv_keeps_first_row_column_order(tmp_path):
    path = write_csv(tmp_path / "m.csv", [
        {"scenario": "tone300", "att_db": 6.26},
        {"scenario": "band", "att_db": 5.14, "extra": 1},
    ])
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().strip()
    assert header == "scenario,att_db,extra"


def test_write_csv_rejects_empty_rows(tmp_path):
    # 빈 CSV 를 쓰면 "측정했는데 결과가 없다"와 "측정에 실패했다"를 구분할 수 없게 된다.
    with pytest.raises(ValueError):
        write_csv(tmp_path / "m.csv", [])


def test_write_csv_normalises_numpy_and_bool(tmp_path):
    path = write_csv(tmp_path / "m.csv", [
        {"a": np.float64(1.5), "b": np.int64(3), "c": np.bool_(True), "d": [1, 2]},
    ])
    row = read_csv(path)[0]
    assert row["a"] == "1.5"
    assert row["b"] == "3"
    assert row["c"] == "1"
    assert json.loads(row["d"]) == [1, 2]


def test_write_series_csv_is_long_format(tmp_path):
    path = write_series_csv(tmp_path / "s.csv", {
        "onset": [690, 674, 687],
        "valid": [True, True, False],
    }, index_name="repeat")
    rows = read_csv(path)
    assert [r["repeat"] for r in rows] == ["0", "1", "2"]
    assert [r["onset"] for r in rows] == ["690", "674", "687"]


def test_write_series_csv_rejects_ragged_series(tmp_path):
    with pytest.raises(ValueError):
        write_series_csv(tmp_path / "s.csv", {"a": [1, 2, 3], "b": [1, 2]})


def test_write_json_handles_numpy(tmp_path):
    path = write_json(tmp_path / "r.json", {"x": np.float32(2.5), "y": np.arange(3)})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["x"] == pytest.approx(2.5)
    assert payload["y"] == [0, 1, 2]


def test_write_wav_is_16bit_mono_at_given_rate(tmp_path):
    path = write_wav(tmp_path / "a.wav", np.zeros(480, dtype=np.float32), 48000)
    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 48000
        assert handle.getnframes() == 480


def test_write_wav_clips_instead_of_wrapping(tmp_path):
    # 오버플로가 랩어라운드하면 조용한 신호가 최대 진폭 잡음으로 들린다.
    path = write_wav(tmp_path / "a.wav", np.array([2.0, -2.0], dtype=np.float32), 48000)
    with wave.open(str(path), "rb") as handle:
        pcm = np.frombuffer(handle.readframes(2), dtype="<i2")
    assert pcm.tolist() == [32767, -32767]


def test_wav_pair_uses_one_shared_scale(tmp_path):
    """ON 이 OFF 보다 조용하면 파일에서도 조용해야 한다 — 이것이 이 모듈의 핵심 계약이다."""

    rng = np.random.default_rng(0)
    off = rng.normal(0.0, 0.10, 48000)
    on = off * 0.25                      # 정확히 12dB 감쇠
    paths = write_wav_pair(tmp_path, "band", off, on, 48000)

    def rms(path):
        with wave.open(str(path), "rb") as handle:
            pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
        return float(np.sqrt(np.mean((pcm / 32767.0) ** 2)))

    ratio_db = 20.0 * np.log10(rms(paths["off"]) / rms(paths["on"]))
    assert ratio_db == pytest.approx(12.0, abs=0.3)


def test_wav_pair_ab_file_is_off_then_on(tmp_path):
    off = np.full(2000, 0.5)
    on = np.full(2000, 0.05)
    paths = write_wav_pair(tmp_path, "x", off, on, 48000)
    with wave.open(str(paths["ab"]), "rb") as handle:
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    half = pcm.size // 2
    assert np.abs(pcm[:half]).mean() > 5 * np.abs(pcm[half:]).mean()


def test_wav_pair_survives_silent_off_segment(tmp_path):
    # 무음 OFF 에서 0 나눗셈으로 죽으면, 측정은 끝났는데 산출물을 잃는다.
    paths = write_wav_pair(tmp_path, "silent", np.zeros(480), np.zeros(480), 48000)
    assert all(path.exists() for path in paths.values())


def test_markdown_table_matches_csv_columns(tmp_path):
    rows = [{"scenario": "tone300", "att_db": 6.26}]
    columns = ["scenario", "att_db"]
    write_csv(tmp_path / "m.csv", rows, columns=columns)
    lines = list(markdown_table(rows, columns))
    assert lines[0] == "| scenario | att_db |"
    assert read_csv(tmp_path / "m.csv")[0]["scenario"] == "tone300"
