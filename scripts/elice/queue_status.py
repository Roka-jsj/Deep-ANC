#!/usr/bin/env python3
"""큐 상태 리더 — **표준 라이브러리만** 쓴다.

torch/numpy 를 import 하지 않는 것이 요점이다. 원격은 32 vCPU 중 28개를 두 학습의
DataLoader 가 쓰고 있어서, 상태를 볼 때마다 무거운 import 를 하면 학습 속도를 깎는다.
시스템 ``python3`` 로도 바로 돌아간다.

    python3 scripts/elice/queue_status.py           # 사람용 표
    python3 scripts/elice/queue_status.py --json    # 기계용
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = REPO_ROOT / "runs" / "queue"

# 하트비트가 이 배수만큼 오래되면 감독자 자체가 죽은 것으로 본다.
STALE_MULTIPLIER = 6
DEFAULT_INTERVAL_SECONDS = 30


def _age_seconds(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        stamp = dt.datetime.fromisoformat(iso)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=dt.timezone.utc)
    return (dt.datetime.now(dt.timezone.utc) - stamp).total_seconds()


def load_all() -> dict:
    payload: dict = {}
    if not STATE_DIR.exists():
        return payload
    for path in sorted(STATE_DIR.glob("gpu*.json")):
        try:
            payload[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            payload[path.stem] = {"error": f"판독 불가: {exc}"}
    return payload


def render(payload: dict) -> str:
    if not payload:
        return f"큐 상태 파일이 없습니다: {STATE_DIR}"
    lines: list[str] = []
    now_kst = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=9)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    lines.append(f"# 큐 상태 ({now_kst} KST)")
    for name in sorted(payload):
        status = payload[name]
        lines.append("")
        if "error" in status:
            lines.append(f"## {name}: {status['error']}")
            continue
        age = _age_seconds(status.get("generated_at_utc"))
        stale = age is not None and age > STALE_MULTIPLIER * DEFAULT_INTERVAL_SECONDS
        badge = "  [STALE — 감독자가 죽었을 수 있음]" if stale else ""
        lines.append(
            f"## {name} — state={status.get('state')} "
            f"pid={status.get('pid')} 유휴누적={status.get('idle_seconds_total')}s{badge}"
        )
        if age is not None:
            lines.append(f"   갱신 {age:.0f}초 전 · 큐 {status.get('queue_file')}")
        waiting = status.get("waiting_on")
        if waiting:
            lines.append(f"   대기: {json.dumps(waiting, ensure_ascii=False)}")
        current = status.get("current_job")
        if current:
            step = current.get("step")
            rate = current.get("rate_it_s")
            progress = f" step={step}" if step is not None else ""
            progress += f" {rate}it/s" if rate else ""
            lines.append(
                f"   실행중: {current.get('id')} (attempt {current.get('attempt')}, "
                f"pid {current.get('pid')}){progress}"
            )
            lines.append(f"     로그: {current.get('log')}")
        jobs = status.get("jobs") or {}
        if jobs:
            lines.append("   완료/판정:")
            for job_id in jobs:
                record = jobs[job_id]
                lines.append(f"     - {job_id}: {record.get('state')} — {record.get('detail')}")
        decisions = status.get("decisions") or {}
        for key in decisions:
            lines.append(f"   결정 {key}: {decisions[key].get('summary')}")
        if status.get("recommendation"):
            lines.append(f"   권고: {status['recommendation']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = load_all()
    if args.json:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(render(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
