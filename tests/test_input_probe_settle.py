"""입력 preflight 가 **기동 트랜지언트를 판정에 쓰지 않는다**는 계약을 고정한다.

왜 이 테스트가 존재하는가
------------------------
I2S 입력은 스트림을 연 직후 약 0.5초 동안 큰 트랜지언트를 낸다(실측 -36.3 dBFS,
peak 0.062). 정상 바닥은 -67.4 dBFS / peak 0.002 다. 앞부분을 버리지 않고 2초를 통째로
재면 RMS 가 -42 dBFS 로 나오는데, 이는 생존 게이트 ``min_rms_dbfs=-80`` 를 **트랜지언트만으로**
통과한다는 뜻이다. 즉 마이크가 죽어 있어도 "살아 있다"고 판정한다 — 이 게이트의 목적이
정확히 무력화된다.

이 오염은 저장소 기록에서 재현된다: 2초 창 예측 -42.31 dBFS 대 로그 -42.34/-42.40/-42.49,
5초 창 예측 -46.27 dBFS 대 문서 -46.33. 그래서 숫자가 아니라 **동작**을 테스트한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from deep_anc import audio_io

FS = 48000
FULL_SCALE = 2 ** 31

# 실측값을 그대로 쓴다 — 이 두 수가 이 테스트의 물리적 근거다.
TRANSIENT_RMS_DBFS = -36.3
TRANSIENT_SECONDS = 0.5
DEAD_MIC_RMS_DBFS = -95.0


def _int32(signal: np.ndarray) -> np.ndarray:
    return np.clip(signal * FULL_SCALE, -FULL_SCALE, FULL_SCALE - 1).astype(np.int32)


def _stream(dead_after_transient: bool, frames: int, rng: np.random.Generator) -> np.ndarray:
    """앞 0.5초 트랜지언트 + 이후 구간(정상 마이크 또는 죽은 마이크)."""

    transient_frames = min(frames, int(TRANSIENT_SECONDS * FS))
    tail_db = DEAD_MIC_RMS_DBFS if dead_after_transient else -67.4
    signal = rng.normal(0.0, 10.0 ** (tail_db / 20.0), frames)
    signal[:transient_frames] = rng.normal(
        0.0, 10.0 ** (TRANSIENT_RMS_DBFS / 20.0), transient_frames
    )
    return _int32(np.stack([signal, signal], axis=1))


class _FakeSounddevice:
    """sd.rec/sd.wait 만 흉내낸다 — 실제 장치를 열지 않는다."""

    def __init__(self, dead_after_transient: bool) -> None:
        self.dead_after_transient = dead_after_transient
        self.requested_frames: int | None = None
        self._rng = np.random.default_rng(20260804)

    def rec(self, frames, samplerate, channels, dtype, device):  # noqa: ANN001
        self.requested_frames = int(frames)
        return _stream(self.dead_after_transient, int(frames), self._rng)

    def wait(self):  # noqa: ANN201
        return None


@pytest.fixture()
def audio_cfg():
    return {
        "sample_rate": FS,
        "input": {"card": "APE", "pcm": 1},
        "output": {"card": "Audio", "pcm": 0},
    }


@pytest.fixture(autouse=True)
def _patch_device(monkeypatch):
    monkeypatch.setattr(
        audio_io, "resolve_alsa_portaudio_device", lambda *a, **k: 1
    )


def _install(monkeypatch, fake: _FakeSounddevice) -> None:
    import sys
    import types

    module = types.ModuleType("sounddevice")
    module.rec = fake.rec
    module.wait = fake.wait
    monkeypatch.setitem(sys.modules, "sounddevice", module)


def test_dead_mic_is_rejected_despite_startup_transient(monkeypatch, audio_cfg):
    """이 테스트가 이 파일의 존재 이유다 — 트랜지언트가 죽은 마이크를 살려내면 안 된다."""

    fake = _FakeSounddevice(dead_after_transient=True)
    _install(monkeypatch, fake)
    report = audio_io.capture_input_probe(audio_cfg, seconds=2.0)
    assert all(not channel["valid"] for channel in report["channels"]), (
        "기동 트랜지언트만으로 -80dBFS 게이트를 통과하면 무신호 마이크를 살아 있다고 판정한다"
    )


def test_live_mic_still_passes(monkeypatch, audio_cfg):
    fake = _FakeSounddevice(dead_after_transient=False)
    _install(monkeypatch, fake)
    report = audio_io.capture_input_probe(audio_cfg, seconds=2.0)
    assert all(channel["valid"] for channel in report["channels"])


def test_probe_captures_extra_frames_for_settle(monkeypatch, audio_cfg):
    fake = _FakeSounddevice(dead_after_transient=False)
    _install(monkeypatch, fake)
    report = audio_io.capture_input_probe(audio_cfg, seconds=2.0, settle_seconds=1.0)
    assert fake.requested_frames == int(3.0 * FS), "settle 만큼 더 캡처해야 한다"
    assert report["frames"] == int(2.0 * FS), "판정에는 요청한 길이만 쓴다"
    assert report["settle_seconds"] == pytest.approx(1.0)


def test_reported_peak_excludes_transient(monkeypatch, audio_cfg):
    """보고되는 peak 도 트랜지언트가 아니어야 한다 — 실측에서 30배 차이가 났다."""

    fake = _FakeSounddevice(dead_after_transient=False)
    _install(monkeypatch, fake)
    settled = audio_io.capture_input_probe(audio_cfg, seconds=2.0, settle_seconds=1.0)
    raw = audio_io.capture_input_probe(audio_cfg, seconds=2.0, settle_seconds=0.0)
    assert settled["channels"][0]["peak"] < 0.2 * raw["channels"][0]["peak"]


def test_settle_zero_reproduces_old_contaminated_behaviour(monkeypatch, audio_cfg):
    """settle=0 은 옛 동작이다. 이 경로가 남아 있으면 회귀가 조용히 돌아온다."""

    fake = _FakeSounddevice(dead_after_transient=True)
    _install(monkeypatch, fake)
    report = audio_io.capture_input_probe(audio_cfg, seconds=2.0, settle_seconds=0.0)
    assert all(channel["valid"] for channel in report["channels"]), (
        "옛 동작(settle=0)에서는 죽은 마이크가 통과했다 — 이 사실이 수정의 근거다"
    )
