"""GPU1 구조 탐색 watcher의 완료 gate·격리·실패 안전성."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from scripts.eval.evaluate_offline import resolve_checkpoint_config


REPO_ROOT = Path(__file__).resolve().parents[1]
SEARCH_SCRIPT = REPO_ROOT / "scripts/elice/run_structure_search.sh"


def _checkpoint(model_name: str = "hybrid_anc_tiny", step: int = 100_000) -> dict:
    cfg = {
        "model": {"name": model_name},
        "data": {"digital_reference_lead_samples": 109},
        "schedule": {"total_steps": 100_000},
        "physics_status": "secondary_surrogate_representation_pretrain",
        "digital_reference_lead_samples": 109,
    }
    return {"step": step, "best_metric": -12.5, "cfg": cfg, "model": {}}


def _atomic_save(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _make_fake_search_repo(tmp_path: Path, *, model_name: str = "hybrid_anc_tiny") -> Path:
    root = tmp_path / "Deep-ANC"
    (root / "scripts/elice").mkdir(parents=True)
    (root / "scripts/train").mkdir(parents=True)
    (root / "scripts/eval").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "runs").mkdir()
    shutil.copy2(SEARCH_SCRIPT, root / "scripts/elice/run_structure_search.sh")

    python = root / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(
        f"#!/bin/bash\nexec {sys.executable!s} \"$@\"\n", encoding="utf-8"
    )
    python.chmod(0o755)

    for name in (
        "train_pretrain.yaml",
        "model_tiny.yaml",
        "model_tiny_long.yaml",
        "model_tiny_attn.yaml",
        "model_tiny_long_attn.yaml",
    ):
        (root / "configs" / name).write_text("name: fake\n", encoding="utf-8")

    last = _checkpoint(model_name=model_name)
    best = _checkpoint(model_name=model_name, step=5_000)
    ckpt = root / "runs/pretrain_tiny_corrected/ckpt"
    _atomic_save(last, ckpt / "last.pt")
    _atomic_save(best, ckpt / "best.pt")

    (root / "runs/train_base_corrected.log").write_text(
        "step   40000 | loss -1 | nmse_t -1 dB | nmse_f -1 dB | lr 1e-4 | 1.70 it/s\n",
        encoding="utf-8",
    )

    (root / "scripts/train/train.py").write_text(
        """
import os
import sys
import time
from pathlib import Path

import torch


def setting(name):
    for index, value in enumerate(sys.argv):
        if value == "--set" and index + 1 < len(sys.argv):
            key, _, raw = sys.argv[index + 1].partition("=")
            if key == name:
                return raw
    raise RuntimeError(name)


model = setting("model_config")
run_dir = Path(setting("ckpt_dir"))
with Path("runs/invocations.log").open("a", encoding="utf-8") as stream:
    stream.write(
        f"train|{model}|gpu={os.environ.get('CUDA_VISIBLE_DEVICES')}|"
        f"total={setting('schedule.total_steps')}|until={setting('run_until_step')}\\n"
    )
if os.environ.get("FAKE_FAIL_MODEL") and os.environ["FAKE_FAIL_MODEL"] in model:
    raise SystemExit(42)

ckpt = run_dir / "ckpt"
ckpt.mkdir(parents=True, exist_ok=True)
state = {"step": 500, "best_metric": -1.0, "cfg": {}, "model": {}}
tmp = ckpt / "last.tmp"
torch.save(state, tmp)
tmp.replace(ckpt / "last.pt")
time.sleep(float(os.environ.get("FAKE_TRAIN_HOLD_SECONDS", "0.4")))
state["step"] = 20000
for name in ("last.pt", "best.pt"):
    tmp = ckpt / f"{name}.tmp"
    torch.save(state, tmp)
    tmp.replace(ckpt / name)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (root / "scripts/eval/evaluate_offline.py").write_text(
        """
import os
import sys
from pathlib import Path

args = sys.argv[1:]
out = Path(args[args.index("--out") + 1])
out.mkdir(parents=True, exist_ok=True)
(out / "metrics.md").write_text("ok\\n", encoding="utf-8")
with Path("runs/invocations.log").open("a", encoding="utf-8") as stream:
    stream.write(f"eval|{args[args.index('--ckpt') + 1]}|gpu={os.environ.get('CUDA_VISIBLE_DEVICES')}\\n")
""".strip()
        + "\n",
        encoding="utf-8",
    )

    fake_bin = root / "fake-bin"
    fake_bin.mkdir()
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(
        """#!/bin/bash
if [[ " $* " == *" --query-gpu=index "* ]]; then
  printf '0\\n1\\n'
elif [ "${FAKE_GPU1_BUSY:-0}" = "1" ]; then
  printf '9999, foreign, 1024\\n'
fi
""",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)
    return root


def _environment(root: Path, **extra: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{root / 'fake-bin'}:{env['PATH']}",
            "STRUCTURE_WAIT_POLL_SECONDS": "0.05",
            "STRUCTURE_ETA_POLL_SECONDS": "0.05",
            "STRUCTURE_GPU_SETTLE_SECONDS": "0",
            "STRUCTURE_GPU_FREE_RETRIES": "1",
            "STRUCTURE_GPU_FREE_RETRY_SECONDS": "0",
            "STRUCTURE_ETA_EVAL_RESERVE_SECONDS": "0",
        }
    )
    env.update(extra)
    return env


def _run(root: Path, **extra_env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "scripts/elice/run_structure_search.sh"],
        cwd=root,
        env=_environment(root, **extra_env),
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )


def test_structure_search_runs_control_first_on_gpu1_with_fair_schedule(tmp_path):
    root = _make_fake_search_repo(tmp_path)

    result = _run(root)

    assert result.returncode == 0, result.stderr
    lines = (root / "runs/invocations.log").read_text(encoding="utf-8").splitlines()
    train_lines = [line for line in lines if line.startswith("train|")]
    assert [line.split("|")[1] for line in train_lines] == [
        "configs/model_tiny.yaml",
        "configs/model_tiny_long.yaml",
        "configs/model_tiny_attn.yaml",
        "configs/model_tiny_long_attn.yaml",
    ]
    assert all("gpu=1" in line for line in lines)
    assert all("total=100000|until=20000" in line for line in train_lines)
    for name in ("tiny_control", "tiny_long", "tiny_attn", "tiny_long_attn"):
        assert (root / f"runs/search_{name}/eta_probe.txt").is_file()
        assert (root / f"runs/search_{name}/ckpt/last.pt").is_file()


def test_structure_search_rejects_foreign_tiny_checkpoint(tmp_path):
    root = _make_fake_search_repo(tmp_path, model_name="hybrid_anc_base")

    result = _run(root)

    assert result.returncode != 0
    assert "model.name" in result.stderr
    assert not (root / "runs/invocations.log").exists()


def test_structure_search_rejects_existing_output_without_overwrite(tmp_path):
    root = _make_fake_search_repo(tmp_path)
    existing = root / "runs/search_tiny_control.log"
    existing.write_text("보존\n", encoding="utf-8")

    result = _run(root)

    assert result.returncode != 0
    assert "기존 산출물을 덮어쓰지 않습니다" in result.stderr
    assert existing.read_text(encoding="utf-8") == "보존\n"
    assert not (root / "runs/invocations.log").exists()


def test_structure_search_rejects_stale_live_pid_identity(tmp_path):
    root = _make_fake_search_repo(tmp_path)
    (root / "runs/train_tiny_corrected.pid").write_text(
        f"{os.getpid()}\n", encoding="utf-8"
    )

    result = _run(root)

    assert result.returncode != 0
    assert "stale PID" in result.stderr
    assert not (root / "runs/invocations.log").exists()


def test_structure_search_rejects_busy_gpu1(tmp_path):
    root = _make_fake_search_repo(tmp_path)

    result = _run(root, FAKE_GPU1_BUSY="1")

    assert result.returncode != 0
    assert "GPU1에 다른 compute process" in result.stderr
    assert not (root / "runs/invocations.log").exists()


def test_structure_search_stops_after_first_candidate_failure(tmp_path):
    root = _make_fake_search_repo(tmp_path)

    result = _run(root, FAKE_FAIL_MODEL="model_tiny.yaml")

    assert result.returncode != 0
    lines = (root / "runs/invocations.log").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert "configs/model_tiny.yaml" in lines[0]
    assert not (root / "runs/search_tiny_long.pid").exists()


def test_structure_search_signal_terminates_active_process_group(tmp_path):
    root = _make_fake_search_repo(tmp_path)
    process = subprocess.Popen(
        ["bash", "scripts/elice/run_structure_search.sh"],
        cwd=root,
        env=_environment(root, FAKE_TRAIN_HOLD_SECONDS="30"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    pid_file = root / "runs/search_tiny_control.pid"
    deadline = time.monotonic() + 10
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert pid_file.exists()
    child_pid = int(pid_file.read_text(encoding="utf-8"))

    process.send_signal(signal.SIGTERM)
    _stdout, stderr = process.communicate(timeout=15)

    assert process.returncode == 143, stderr
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_evaluate_offline_prefers_checkpoint_resolved_config(tmp_path):
    state = {"cfg": {"data": {"sample_rate": 12345}, "duct": {"marker": "saved"}}}

    data = resolve_checkpoint_config(state, "data", None, "missing.yaml")
    duct = resolve_checkpoint_config(state, "duct", None, "missing.yaml")

    assert data == {"sample_rate": 12345}
    assert duct == {"marker": "saved"}
    data["sample_rate"] = 1
    assert state["cfg"]["data"]["sample_rate"] == 12345

    override = tmp_path / "data.yaml"
    override.write_text("sample_rate: 48000\n", encoding="utf-8")
    assert resolve_checkpoint_config(state, "data", str(override), "missing.yaml") == {
        "sample_rate": 48000
    }
