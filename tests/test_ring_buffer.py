"""SPSC 링버퍼 검증 — 데이터 무결성, 랩어라운드, 언더런 재동기(결함 #9/#10 회귀)."""

import numpy as np

from deep_anc.realtime.ring_buffer import SPSCRing


def test_roundtrip_wraparound():
    ring = SPSCRing(2, capacity_samples=1000)
    rng = np.random.default_rng(0)
    sent, received = [], []
    for _ in range(40):                            # 40×256 ≫ capacity → 랩어라운드 다수
        blk = rng.standard_normal((2, 256)).astype(np.float32)
        ring.push(blk)
        sent.append(blk)
        out = ring.pop(256)
        assert out is not None
        received.append(out)
    assert np.allclose(np.concatenate(sent, axis=1), np.concatenate(received, axis=1))
    assert ring.overruns == 0 and ring.underruns == 0


def test_push_full_drops_new_not_read_pos():
    """가득 찬 버퍼에서 push 는 새 블록을 버리고 read_pos(소비자 소유)를 건드리지 않는다."""
    ring = SPSCRing(1, capacity_samples=512)
    a = np.ones((1, 512), dtype=np.float32)
    ring.push(a)
    read_before = ring.read_pos
    ring.push(np.full((1, 256), 2.0, dtype=np.float32))   # 공간 없음 → drop-new
    assert ring.overruns == 1
    assert ring.read_pos == read_before
    out = ring.pop(512)
    assert np.allclose(out, a)                     # 기존 데이터 보존


def test_pop_latest_resyncs_after_underrun():
    """언더런 후 백로그가 keep_backlog 로 잘려 지연이 영구 누적되지 않는다."""
    ring = SPSCRing(1, capacity_samples=4096)
    # 소비자가 한 번 놓침 (언더런)
    out, ok = ring.pop_latest(256, keep_backlog=256)
    assert not ok and ring.underruns == 1
    # 생산자가 3블록 밀어넣음 (백로그 3 hop = 지연 3 hop 상황)
    for v in (1.0, 2.0, 3.0):
        ring.push(np.full((1, 256), v, dtype=np.float32))
    # pop_latest 는 백로그를 1 hop 으로 줄이고 가장 최신 블록을 반환
    out, ok = ring.pop_latest(256, keep_backlog=256)
    assert ok
    assert np.allclose(out, 3.0)
    assert ring.drops == 512                       # 2블록 폐기가 카운터로 표면화
    assert ring.available() == 0                   # 정상 핸드오프(1 hop)로 복귀
