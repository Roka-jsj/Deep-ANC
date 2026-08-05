#!/usr/bin/env python3
"""준비 게이트→fine-tune→독립 val/test→완료 게이트를 순차 실행한다.

오디오 장치는 열지 않는다. P/S·recorded·완료된 pretrain 이 준비되지 않았으면 첫
게이트에서 종료하며 **``runs/`` 아래에 아무것도 만들지 않는다** — 학습 디렉터리의
존재가 "학습이 실제로 시작됐다"는 의미를 유지해야 하기 때문이다. 감사·상태 산출물은
전부 ``results/finetune_autostart/<run-key>/`` 로 간다.

중단 뒤 ``last.pt`` 가 있으면 같은 run 을 자동 재개하고, ``best.pt`` 만 남은 모호한
상태는 덮어쓰지 않는다. 재개 판단은 항상 디스크 사실로만 하며 status.json 은 쓰지 않는다.

  .venv/bin/python scripts/train/run_finetune_pipeline.py \
    --config configs/train_finetune.yaml \
    --set data.digital_primary_path_mode=measured

  --check-only   준비 리포트만 생성 (학습 미시작)
  --status       lock 없이 현재 상태만 출력

종료 코드: 0 OK / 1 NOT READY / 2 config·모호한 재개 / 3 다른 프로세스가 같은 run 을
학습 중(train.lock) / 4 pipeline 중복 실행(pipeline.lock) / 5 단계 실패
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.config import REPO_ROOT, load_train_config  # noqa: E402
from deep_anc.train.finetune_readiness import (  # noqa: E402
    audit_finetune_readiness,
    render_audit_markdown,
)
from deep_anc.train.pipeline_status import (  # noqa: E402
    EXIT_CONFIG,
    EXIT_NOT_READY,
    EXIT_OK,
    EXIT_PIPELINE_LOCKED,
    EXIT_STAGE_FAILED,
    EXIT_TRAIN_LOCKED,
    PipelineStatus,
    atomic_write_text,
    config_fingerprint,
    read_status,
    sha256_text,
)
from deep_anc.train.process_lock import (  # noqa: E402
    LockHeldError,
    ProcessLock,
    autostart_state_dir,
    finetune_run_key,
    resolve_run_dir,
)


class StepFailed(RuntimeError):
    """자식 단계 실패. 어느 단계에서 어떤 코드로 죽었는지를 함께 나른다.

    returncode 만으로는 "train.py 의 3(중복 학습)"과 "다른 자식의 우연한 3"을 구분할 수
    없어서 단계 이름을 반드시 같이 싣는다.
    """

    def __init__(self, step: str, returncode: int) -> None:
        super().__init__(f"{step} 단계가 exit {returncode}로 실패했습니다")
        self.step = step
        self.returncode = returncode


def _repo_state_dir(value: str | Path) -> Path:
    path = Path(value).expanduser()
    path = path if path.is_absolute() else REPO_ROOT / path
    path = path.resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"state-dir은 Deep_ANC 내부여야 합니다: {path}") from exc
    return path


def _resolved_input_path(value: str | Path) -> Path:
    """부모가 감사한 바로 그 파일의 절대경로.

    상대 경로를 그대로 자식에게 넘기면, 저장소 밖 CWD 에서 실행했을 때 부모는
    ``$CWD/configs/...`` 를 감사하고 자식 train.py 는 ``$REPO/configs/...`` 를 학습한다.
    감사한 설정과 학습한 설정이 달라지는 fail-open 이라 반드시 해석해서 넘긴다.
    """

    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = Path.cwd() / path
    return (cwd_candidate if cwd_candidate.exists() else REPO_ROOT / path).resolve()


def _run(command: list[str], *, step: str, status: PipelineStatus) -> None:
    print("+ " + " ".join(command), flush=True)
    started = time.monotonic()
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)  # noqa: S603
    status.record_step(step, command, completed.returncode, time.monotonic() - started)
    if completed.returncode != 0:
        raise StepFailed(step, completed.returncode)


def _train_lock_busy(train_lock: Path) -> bool:
    """자식이 잡을 train.lock 이 이미 물려 있는지 미리 본다.

    이건 최적화일 뿐 권한이 아니다(TOCTOU 가 있다). 실제 배제는 자식 rank0 의 flock 이
    하고 우리는 그 exit 3 을 그대로 3으로 매핑한다. 여기서 미리 걸러내는 이유는 무거운
    recorded 전수 QA 를 헛돌리지 않기 위해서다.
    """

    if not train_lock.exists():
        return False
    try:
        ProcessLock(train_lock, role="probe").acquire().release()
    except LockHeldError:
        return True
    except OSError:
        return False
    return False


def _warn_legacy_run_audit(run_dir: Path) -> None:
    """옛 코드가 학습 디렉터리에 남긴 감사 산출물을 알린다(삭제하지 않는다)."""

    legacy = run_dir / "audit" / "readiness.json"
    if legacy.exists() and not (run_dir / "ckpt").exists():
        print(
            f"[주의] 학습이 시작된 적 없는데 감사 산출물이 남아 있습니다: {legacy}\n"
            f"       구버전 파이프라인의 잔재입니다. 보존이 필요하면 "
            f"results/finetune_autostart/<run-key>/audit/legacy/ 로 옮기세요.",
            file=sys.stderr,
        )


def _print_status(status_path: Path, pipeline_lock: Path, train_lock: Path) -> None:
    payload = read_status(status_path)
    print(f"status: {status_path}")
    if payload is None:
        print("  (없음 — 아직 실행된 적이 없습니다)")
    else:
        print(f"  phase={payload.get('phase')} exit_code={payload.get('exit_code')}")
        print(f"  mode={payload.get('mode')} run_dir={payload.get('run_dir')}")
        readiness = payload.get("readiness") or {}
        if readiness:
            print(f"  readiness.ok={readiness.get('ok')} 실패={readiness.get('failed_checks')}")
        for step in payload.get("steps") or []:
            print(f"    - {step['name']}: exit {step['returncode']} ({step['duration_seconds']}s)")
    for label, path in (("pipeline", pipeline_lock), ("train", train_lock)):
        held = _train_lock_busy(path)
        print(f"  {label}.lock 보유중={held} ({path})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_finetune.yaml")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], help="key=value override"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="준비 리포트만 생성하고 학습은 시작하지 않음 (같은 pipeline.lock 으로 보호됨)",
    )
    parser.add_argument(
        "--state-dir",
        default=None,
        help=(
            "감사·상태 산출 경로(기본: results/finetune_autostart/<run-key>). "
            "lock 은 중복 실행 우회를 막기 위해 항상 기본 경로에 둔다. Deep_ANC 내부만 허용"
        ),
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="lock 없이 현재 status.json 과 lock 보유자만 출력하고 종료",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # (A) lock 이전: 설정 해석. 실패하면 어떤 파일도 만들지 않는다.
    try:
        config_path = _resolved_input_path(args.config)
        cfg = load_train_config(config_path, args.overrides)
        run_dir = resolve_run_dir(cfg["ckpt_dir"])
        run_dir.relative_to(REPO_ROOT.resolve())
        lock_dir = autostart_state_dir(run_dir)
        state_dir = _repo_state_dir(args.state_dir) if args.state_dir else lock_dir
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"[중단] fine-tune config 오류: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    audit_dir = state_dir / "audit"
    status_path = state_dir / "status.json"
    pipeline_lock = lock_dir / "pipeline.lock"
    train_lock = lock_dir / "train.lock"

    # (B) --status 는 lock 없이 읽기만 한다.
    if args.status:
        _print_status(status_path, pipeline_lock, train_lock)
        return EXIT_OK

    if state_dir != lock_dir:
        print(f"[주의] --state-dir 사용: lock 은 {pipeline_lock} 로 고정됩니다", file=sys.stderr)
    _warn_legacy_run_audit(run_dir)

    # (C) readiness 감사보다 먼저 lock. 탈락자는 state_dir 에 아무것도 쓰지 않는다.
    try:
        lock = ProcessLock(
            pipeline_lock,
            role="fine-tune pipeline",
            metadata={
                "run_dir": str(run_dir),
                "state_dir": str(state_dir),
                "mode": "check-only" if args.check_only else "pipeline",
            },
        ).acquire()
    except LockHeldError as exc:
        print(f"[중복] {exc}", file=sys.stderr)
        print(
            "  진행 상황은 --status 로, 독립 감사는 check_finetune.py 로 확인하세요.",
            file=sys.stderr,
        )
        return EXIT_PIPELINE_LOCKED

    # 이미 acquire() 했으므로 ``with lock:`` 을 쓰면 __enter__ 가 같은 경로를 다시 잡으려다
    # 자기 자신과 충돌한다. 획득은 한 번, 해제는 finally 로 한다.
    try:
        run_dir_existed = run_dir.exists()
        status = PipelineStatus(
            status_path,
            mode="check-only" if args.check_only else "pipeline",
            run_key=finetune_run_key(run_dir),
            run_dir=run_dir,
            state_dir=state_dir,
            lock_path=pipeline_lock,
            config_path=config_path,
            config_sha256=sha256_text(config_path.read_text(encoding="utf-8")),
            overrides=list(args.overrides),
            fingerprint=config_fingerprint(cfg),
        )

        # 학습 단계가 있을 때만: 무거운 QA 를 돌리기 전에 중복 학습을 먼저 걸러낸다.
        if not args.check_only and _train_lock_busy(train_lock):
            print("[중단] 다른 프로세스가 같은 run 을 학습 중입니다(train.lock)", file=sys.stderr)
            return status.finish("train_locked", EXIT_TRAIN_LOCKED)

        status.update("readiness")
        readiness = audit_finetune_readiness(cfg)
        atomic_write_text(
            audit_dir / "readiness.json",
            json.dumps(readiness, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        atomic_write_text(audit_dir / "readiness.md", render_audit_markdown(readiness))
        status.update(
            readiness={
                "ok": readiness["ok"],
                "report": str(audit_dir / "readiness.json"),
                "checked_at_utc": readiness.get("checked_at_utc"),
                "failed_checks": [c["id"] for c in readiness["checks"] if not c["ok"]],
            },
            # NOT READY 인데 학습 디렉터리가 생겼다면 불변식이 깨진 것이다. 배포 환경에서도
            # 회귀를 잡을 수 있게 관측값을 남긴다.
            run_dir_created_before_ready=(not run_dir_existed and run_dir.exists()),
        )
        if not readiness["ok"]:
            print("[NOT READY] fine-tune 학습을 시작하지 않습니다", file=sys.stderr)
            for item in readiness["checks"]:
                if not item["ok"]:
                    print(f"  - {item['id']}: {item['message']}", file=sys.stderr)
            return status.finish("not_ready", EXIT_NOT_READY)

        print("[READY] P/S·lead·pretrain·recorded 진입 게이트 PASS", flush=True)
        if args.check_only:
            return status.finish("ready", EXIT_OK)

        # ``.resolve()`` 를 쓰면 안 된다. venv 의 bin/python 은 시스템 인터프리터를 가리키는
        # **심볼릭 링크**라, 따라가는 순간 venv 의 site-packages 를 잃고 자식이
        # "No module named yaml" 로 죽는다. sys.executable 은 이미 절대경로다.
        python = sys.executable
        override_args = [part for value in args.overrides for part in ("--set", value)]
        ckpt_dir = run_dir / "ckpt"
        best = ckpt_dir / "best.pt"
        last = ckpt_dir / "last.pt"
        if best.exists() and not last.exists():
            print(
                f"[중단] best.pt만 있고 last.pt가 없어 안전하게 재개할 수 없습니다: {ckpt_dir}",
                file=sys.stderr,
            )
            return status.finish("failed", EXIT_CONFIG, failed_step="resume_ambiguous")

        train_command = [
            python,
            str(REPO_ROOT / "scripts" / "train" / "train.py"),
            "--config",
            str(config_path),
            *override_args,
        ]
        if last.exists():
            train_command += ["--resume", str(last)]

        try:
            status.update(
                "training",
                resume={
                    "resumed_from": str(last) if last.exists() else None,
                    "detected_by": "filesystem",
                },
            )
            _run(train_command, step="train", status=status)
            if not best.is_file() or not last.is_file():
                raise RuntimeError("학습 종료 뒤 best.pt/last.pt가 모두 존재하지 않습니다")

            status.update("evaluating")
            manifest = str(cfg["recorded_manifest"])
            for split in ("val", "test"):
                _run(
                    [
                        python,
                        str(REPO_ROOT / "scripts" / "eval" / "evaluate_recorded.py"),
                        "--ckpt",
                        str(best),
                        "--manifest",
                        manifest,
                        "--split",
                        split,
                        "--out",
                        str(run_dir / f"eval_recorded_{split}"),
                    ],
                    step=f"evaluate_recorded_{split}",
                    status=status,
                )

            status.update("completion")
            _run(
                [
                    python,
                    str(REPO_ROOT / "scripts" / "train" / "check_finetune.py"),
                    "--config",
                    str(config_path),
                    *override_args,
                    "--completion-checkpoint",
                    str(best),
                    "--out-dir",
                    str(audit_dir),
                ],
                step="completion",
                status=status,
            )
        except StepFailed as exc:
            if exc.step == "train" and exc.returncode == EXIT_TRAIN_LOCKED:
                print(
                    "[중단] 다른 프로세스가 같은 run 을 학습 중입니다(train.lock)",
                    file=sys.stderr,
                )
                return status.finish("train_locked", EXIT_TRAIN_LOCKED)
            print(f"[FAIL] fine-tune pipeline: {exc}", file=sys.stderr)
            return status.finish("failed", EXIT_STAGE_FAILED, failed_step=exc.step)
        except (KeyError, OSError, RuntimeError) as exc:
            print(f"[FAIL] fine-tune pipeline: {exc}", file=sys.stderr)
            return status.finish("failed", EXIT_STAGE_FAILED)

        print(f"[COMPLETE] fine-tuning + independent val/test G4 PASS: {best}")
        return status.finish(
            "done",
            EXIT_OK,
            artifacts={
                "best": str(best),
                "last": str(last),
                "eval_val": str(run_dir / "eval_recorded_val"),
                "eval_test": str(run_dir / "eval_recorded_test"),
                "completion_report": str(audit_dir / "completion.json"),
            },
        )
    finally:
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
