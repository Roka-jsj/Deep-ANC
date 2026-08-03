#!/usr/bin/env bash
# GPU 작업 큐 감독자 런처.
#
# SSH 가 끊겨도 계속 돌도록 setsid nohup 으로 띄우고 stdin 을 끊는다. 감독자 자신의
# 중복 실행 방지는 파이썬 쪽 flock 이 하므로 여기서는 하지 않는다(런처가 lock 을 잡으면
# 종료할 때 해제돼 의미가 없다).
#
#   bash scripts/elice/run_job_queue.sh 1        # GPU1 큐 기동
#   bash scripts/elice/run_job_queue.sh 0        # GPU0 큐 기동
#   DRY_RUN=1 bash scripts/elice/run_job_queue.sh 1
set -euo pipefail

GPU="${1:?사용법: run_job_queue.sh <gpu-index> [queue-yaml]}"
QUEUE="${2:-configs/elice/queue_gpu${GPU}.yaml}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

PYTHON=".venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="python3"

if [ ! -f "$QUEUE" ]; then
  echo "[오류] 큐 정의가 없습니다: $QUEUE" >&2
  exit 2
fi

# 기동 전 스키마 검증 — 잘못된 큐로 감독자를 띄우지 않는다.
"$PYTHON" scripts/elice/job_queue.py verify --queue "$QUEUE" --gpu "$GPU"

mkdir -p runs/queue/logs
LOG="runs/queue/supervisor_gpu${GPU}.log"

if [ "${DRY_RUN:-0}" = "1" ]; then
  exec "$PYTHON" scripts/elice/job_queue.py run --queue "$QUEUE" --gpu "$GPU" --dry-run
fi

# < /dev/null 이 없으면 SSH 종료 시 감독자가 SIGHUP/EOF 로 죽는다.
setsid nohup "$PYTHON" scripts/elice/job_queue.py run \
  --queue "$QUEUE" --gpu "$GPU" >> "$LOG" 2>&1 < /dev/null &
echo "STARTED gpu=$GPU pid=$! queue=$QUEUE log=$LOG"
