#!/bin/bash
# 사전학습 실행 (Elice) — 세션 끊김 대비 nohup + 로그 tee.
# 사용: bash scripts/elice/run_pretrain.sh [config(기본 configs/train_pretrain.yaml)]
# GPU 수를 감지해 2장 이상이면 torchrun DDP 로 실행한다.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

CONFIG="${1:-configs/train_pretrain.yaml}"
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="runs/train_${STAMP}.log"
mkdir -p runs

if [ "$NGPU" -ge 2 ]; then
  CMD=(torchrun --nproc_per_node="$NGPU" scripts/train/train.py --config "$CONFIG")
else
  CMD=(python scripts/train/train.py --config "$CONFIG")
fi

echo "실행: ${CMD[*]}  (GPU ${NGPU}장, 로그 $LOG)"
nohup "${CMD[@]}" >>"$LOG" 2>&1 &
PID=$!
echo "$PID" > runs/train_${STAMP}.pid
echo "백그라운드 시작 (PID $PID). 모니터링:"
echo "  tail -f $LOG"
echo "  tensorboard --logdir runs --port 6006   # VS Code 포트포워딩으로 접속"
