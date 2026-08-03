#!/bin/bash
# 사전학습 실행 (Elice) — 세션 끊김 대비 nohup + 로그 tee.
# 사용: bash scripts/elice/run_pretrain.sh [config(기본 configs/train_pretrain.yaml)]
# GPU 수를 감지해 2장 이상이면 torchrun DDP 로 실행한다.
set -euo pipefail
cd "$(dirname "$0")/../.."
VENV_PYTHON="$PWD/.venv/bin/python"
VENV_TORCHRUN="$PWD/.venv/bin/torchrun"

if [ ! -x "$VENV_PYTHON" ]; then
  echo "[오류] $VENV_PYTHON 없음 — setup_env.sh 를 먼저 실행하세요." >&2
  exit 1
fi

CONFIG="${1:-configs/train_pretrain.yaml}"
NGPU=$("$VENV_PYTHON" -c "import torch; print(torch.cuda.device_count())")
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="runs/train_${STAMP}.log"
mkdir -p runs

if [ "$NGPU" -ge 2 ]; then
  if [ ! -x "$VENV_TORCHRUN" ]; then
    echo "[오류] $VENV_TORCHRUN 없음 — setup_env.sh 를 다시 확인하세요." >&2
    exit 1
  fi
  CMD=("$VENV_TORCHRUN" --nproc_per_node="$NGPU" scripts/train/train.py --config "$CONFIG")
else
  CMD=("$VENV_PYTHON" scripts/train/train.py --config "$CONFIG")
fi

echo "실행: ${CMD[*]}  (GPU ${NGPU}장, 로그 $LOG)"
nohup "${CMD[@]}" >>"$LOG" 2>&1 &
PID=$!
echo "$PID" > runs/train_${STAMP}.pid
sleep 5
if ! kill -0 "$PID" 2>/dev/null; then
  echo "[오류] 학습 프로세스(PID $PID)가 시작 직후 종료했습니다." >&2
  tail -n 20 "$LOG" >&2 || true
  exit 1
fi
echo "백그라운드 시작 (PID $PID). 모니터링:"
echo "  tail -f $LOG"
echo "  .venv/bin/tensorboard --logdir runs --port 6006   # VS Code 포트포워딩으로 접속"
