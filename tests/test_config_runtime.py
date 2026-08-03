from pathlib import Path

import yaml

from deep_anc.config import load_runtime_config


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_runtime_overrides_apply_to_loaded_hardware_and_duct(tmp_path):
    hardware = tmp_path / "hardware.yaml"
    duct = tmp_path / "duct.yaml"
    runtime = tmp_path / "runtime.yaml"
    _write_yaml(hardware, {"audio": {"block_size": 256, "latency": "low"}})
    _write_yaml(duct, {"secondary_path": {"handoff_extra_samples": 256}})
    _write_yaml(
        runtime,
        {
            "hardware_config": str(hardware),
            "duct_config": str(duct),
            "hop": 256,
        },
    )

    cfg = load_runtime_config(
        runtime,
        [
            "hardware.audio.block_size=512",
            "hardware.audio.latency=high",
            "duct.secondary_path.handoff_extra_samples=512",
            "hop=512",
        ],
    )

    assert cfg["hardware"]["audio"]["block_size"] == 512
    assert cfg["hardware"]["audio"]["latency"] == "high"
    assert cfg["duct"]["secondary_path"]["handoff_extra_samples"] == 512
    assert cfg["hop"] == 512
