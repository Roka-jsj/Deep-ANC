#!/bin/bash
# 2×GPU 모델 병렬 학습: GPU0=base, GPU1=tiny (사용자 지정 운용 방식).
# GPU 1장뿐이면 base 만 실행한다. nohup — SSH 끊겨도 계속 돈다.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
mkdir -p runs

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")

CUDA_VISIBLE_DEVICES=0 setsid nohup python scripts/train/train.py \
  --config configs/train_pretrain.yaml --set num_workers=10 \
  > runs/train_base.log 2>&1 < /dev/null &
echo "base  → GPU0 (PID $!, runs/train_base.log)"

if [ "$NGPU" -ge 2 ]; then
  CUDA_VISIBLE_DEVICES=1 setsid nohup python scripts/train/train.py \
    --config configs/train_pretrain.yaml \
    --set model_config=configs/model_tiny.yaml --set ckpt_dir=runs/pretrain_tiny \
    --set batch_size=32 --set num_workers=10 \
    > runs/train_tiny.log 2>&1 < /dev/null &
  echo "tiny  → GPU1 (PID $!, runs/train_tiny.log)"
else
  echo "GPU 1장 — tiny 는 base 완료 후 실행하세요"
fi
sleep 5
pgrep -af "train.py" | head -4 || true
