"""자기생성 digital-reference 선행 공급 정렬 회귀 테스트."""

import numpy as np
import pytest

from deep_anc.realtime.noise_gen import DigitalReferenceBuffer
from deep_anc.realtime.engines import checkpoint_digital_reference_lead_samples
from deep_anc.realtime.run_realtime import validate_digital_reference_lead
from deep_anc.train.trainer import checkpoint_training_lead, cfg_snapshot


def test_zero_lead_is_noop_without_aliasing():
    aligner = DigitalReferenceBuffer(0)
    generated = np.arange(8, dtype=np.float32)

    playback, reference = aligner.process(generated)

    np.testing.assert_array_equal(playback, generated)
    np.testing.assert_array_equal(reference, generated)
    playback[0] = -1.0
    assert reference[0] == 0.0


def test_future_reference_alignment_across_variable_blocks():
    lead = 5
    aligner = DigitalReferenceBuffer(lead)
    generated_blocks = [
        np.arange(0, 3, dtype=np.float32),
        np.arange(3, 11, dtype=np.float32),
        np.arange(11, 15, dtype=np.float32),
    ]

    pairs = [aligner.process(block) for block in generated_blocks]
    playback = np.concatenate([pair[0] for pair in pairs])
    reference = np.concatenate([pair[1] for pair in pairs])
    generated = np.concatenate(generated_blocks)

    np.testing.assert_array_equal(reference, generated)
    np.testing.assert_array_equal(
        playback, np.concatenate([np.zeros(lead, dtype=np.float32), generated[:-lead]])
    )
    np.testing.assert_array_equal(reference[:-lead], playback[lead:])


def test_multichannel_signal_and_gate_share_the_same_delay():
    aligner = DigitalReferenceBuffer(2)
    future = np.stack(
        [np.array([10, 11, 12, 13], dtype=np.float32), np.ones(4, dtype=np.float32)]
    )

    played, reference = aligner.process(future)

    np.testing.assert_array_equal(played[0], [0, 0, 10, 11])
    np.testing.assert_array_equal(played[1], [0, 0, 1, 1])
    np.testing.assert_array_equal(reference, future)


@pytest.mark.parametrize("lead", [-1, -109])
def test_negative_lead_is_rejected(lead):
    with pytest.raises(ValueError, match="0 이상"):
        DigitalReferenceBuffer(lead)


def test_runtime_rejects_checkpoint_lead_mismatch():
    assert validate_digital_reference_lead("digital", 109, 109) == 109
    with pytest.raises(ValueError, match="runtime=109, checkpoint=0"):
        validate_digital_reference_lead("digital", 109, 0)
    with pytest.raises(ValueError, match="reference=digital"):
        validate_digital_reference_lead("mic", 109, None)


def test_legacy_checkpoint_metadata_is_explicitly_lead_zero():
    assert checkpoint_digital_reference_lead_samples({"cfg": {"model": {}}}) == 0
    assert checkpoint_digital_reference_lead_samples(
        {"cfg": {"digital_reference_lead_samples": 109}}
    ) == 109
    assert checkpoint_training_lead({"cfg": {"model": {}}}) == 0
    assert checkpoint_training_lead(
        {"cfg": {"data": {"digital_reference_lead_samples": 109}}}
    ) == 109


def test_checkpoint_snapshot_preserves_training_lead():
    cfg = {
        "model": {"name": "test"},
        "stage": "open_loop",
        "loss": {},
        "data": {"sample_rate": 48_000, "digital_reference_lead_samples": 109},
    }
    assert cfg_snapshot(cfg)["digital_reference_lead_samples"] == 109
