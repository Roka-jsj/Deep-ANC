"""Elice 부트스트랩/학습 시작 셸의 안전 불변식."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ELICE_SCRIPTS = (
    REPO_ROOT / "scripts/elice/bootstrap_all.sh",
    REPO_ROOT / "scripts/elice/setup_env.sh",
    REPO_ROOT / "scripts/elice/run_parallel_models.sh",
    REPO_ROOT / "scripts/elice/run_pretrain.sh",
    REPO_ROOT / "scripts/elice/run_structure_search.sh",
)


@pytest.mark.parametrize("script", ELICE_SCRIPTS)
def test_elice_shell_scripts_parse(script: Path):
    result = subprocess.run(
        ["bash", "-n", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_bootstrap_has_explicit_completeness_and_empty_array_guards():
    text = ELICE_SCRIPTS[0].read_text(encoding="utf-8")

    assert '"${pids[@]:-}"' not in text
    assert 'for p in "${pids[@]}"' in text
    assert "-eq 2000" in text
    assert "meta/esc50.csv" in text
    assert "-eq 8000" in text
    assert "-eq 16" in text
    assert "-eq 3600" in text
    assert "unzip -tq" in text
    assert 'flock -n 8' in text
    assert "active_train=$(pgrep -af '[t]rain\\.py' || true)" in text
    assert 'for p in "${extract_pids[@]}"' in text
    assert "file_list_complete" in text
    assert "dns_marker_complete" in text
    assert "구버전에서 이미 해제된 데이터 보존" not in text
    assert "expected_shape = (300, 8192)" in text
    assert "np.isfinite(value).all()" in text

    runner = ELICE_SCRIPTS[2].read_text(encoding="utf-8")
    assert "rollback_startup" in runner
    assert "startup_committed=1" in runner

    ddp_runner = ELICE_SCRIPTS[3].read_text(encoding="utf-8")
    assert '[ ! -x "$VENV_TORCHRUN" ]' in ddp_runner
    assert 'kill -0 "$PID"' in ddp_runner

    search_runner = ELICE_SCRIPTS[4].read_text(encoding="utf-8")
    assert "validate_tiny_completion" in search_runner
    assert 'EXPECTED_TINY_MODEL="hybrid_anc_tiny"' in search_runner
    assert "tiny_pid_is_expected" in search_runner
    assert "wait_for_gpu1_free" in search_runner
    assert "terminate_active_child" in search_runner
    assert "configs/model_tiny.yaml" in search_runner
    assert '--set run_until_step="$PILOT_STEPS"' in search_runner
    assert 'eval_pilot_${artifact}' in search_runner
    assert "eta_probe.txt" in search_runner


def _make_fake_runner(
    tmp_path: Path,
    *,
    gpu_count: int,
    train_exit: int,
    keep_gpu0_alive: bool = False,
) -> Path:
    root = tmp_path / "Deep-ANC"
    scripts = root / "scripts/elice"
    scripts.mkdir(parents=True)
    shutil.copy2(ELICE_SCRIPTS[2], scripts / "run_parallel_models.sh")

    python = root / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    keep_alive_script = (
        'if [ "${CUDA_VISIBLE_DEVICES:-}" = "0" ]; then\n'
        "  trap 'exit 0' TERM INT\n"
        "  (trap '' TERM INT; sleep 30) &\n"
        "  child=$!\n"
        "  printf '%s\\n' \"$child\" > runs/fake_orphan_child.pid\n"
        "  wait \"$child\"\n"
        "  exit 0\n"
        "fi\n"
        if keep_gpu0_alive
        else ""
    )
    python.write_text(
        "#!/bin/sh\n"
        'if [ "${1:-}" = "-c" ]; then\n'
        f"  echo {gpu_count}\n"
        "  exit 0\n"
        "fi\n"
        f"{keep_alive_script}"
        'echo "모의 학습 프로세스 종료" >&2\n'
        f"exit {train_exit}\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    return root


def test_parallel_runner_does_not_overwrite_existing_log(tmp_path: Path):
    root = _make_fake_runner(tmp_path, gpu_count=1, train_exit=0)
    log = root / "runs/train_base_corrected.log"
    log.parent.mkdir()
    log.write_text("보존할 로그\n", encoding="utf-8")

    result = subprocess.run(
        ["bash", "scripts/elice/run_parallel_models.sh"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode != 0
    assert "자동 덮어쓰지 않습니다" in result.stderr
    assert log.read_text(encoding="utf-8") == "보존할 로그\n"
    assert not (root / "runs/train_base_corrected.pid").exists()


def test_parallel_runner_reports_immediate_process_exit(tmp_path: Path):
    root = _make_fake_runner(tmp_path, gpu_count=2, train_exit=42)

    result = subprocess.run(
        ["bash", "scripts/elice/run_parallel_models.sh"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert result.returncode != 0
    assert "base 학습 프로세스" in result.stderr
    assert "tiny 학습 프로세스" in result.stderr
    assert "모의 학습 프로세스 종료" in result.stderr
    rollback_dirs = list((root / "runs").glob("failed_start_*"))
    assert len(rollback_dirs) == 1
    assert (rollback_dirs[0] / "train_base_corrected.pid").is_file()
    assert (rollback_dirs[0] / "train_tiny_corrected.pid").is_file()
    assert not (root / "runs/train_base_corrected.pid").exists()
    assert not (root / "runs/train_tiny_corrected.pid").exists()


def test_parallel_runner_rolls_back_survivor_on_partial_start_failure(tmp_path: Path):
    root = _make_fake_runner(
        tmp_path,
        gpu_count=2,
        train_exit=42,
        keep_gpu0_alive=True,
    )

    result = subprocess.run(
        ["bash", "scripts/elice/run_parallel_models.sh"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert result.returncode != 0
    assert "tiny 학습 프로세스" in result.stderr
    assert "시작 작업을 되돌립니다" in result.stderr
    rollback_dirs = list((root / "runs").glob("failed_start_*"))
    assert len(rollback_dirs) == 1
    base_pid = int((rollback_dirs[0] / "train_base_corrected.pid").read_text())
    orphan_pid = int((root / "runs/fake_orphan_child.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(base_pid, 0)
    with pytest.raises(ProcessLookupError):
        os.kill(orphan_pid, 0)
    assert not (root / "runs/train_base_corrected.log").exists()
    assert not (root / "runs/pretrain_base_corrected").exists()
