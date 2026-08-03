#!/bin/bash
# corrected tiny가 정상 100k를 완료하면 GPU1에서 tiny 대조군+구조 후보 3개를
# 같은 100k LR 스케줄의 0→20k 구간으로 순차 비교한다.
# 기존 학습/체크포인트는 절대 덮어쓰지 않으며 base/GPU0는 건드리지 않는다.
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON="$PWD/.venv/bin/python"
TINY_PID_FILE="runs/train_tiny_corrected.pid"
TINY_LAST="runs/pretrain_tiny_corrected/ckpt/last.pt"
TINY_BEST="runs/pretrain_tiny_corrected/ckpt/best.pt"
PILOT_STEPS=20000
ETA_PROBE_STEP=500
EXPECTED_TINY_STEP=100000
EXPECTED_TINY_MODEL="hybrid_anc_tiny"
EXPECTED_PHYSICS="secondary_surrogate_representation_pretrain"
EXPECTED_LEAD=109

# 테스트/운영 환경에 따라 대기 간격만 줄일 수 있다. 학습 예산은 고정이다.
WAIT_POLL_SECONDS=${STRUCTURE_WAIT_POLL_SECONDS:-30}
ETA_POLL_SECONDS=${STRUCTURE_ETA_POLL_SECONDS:-10}
GPU_SETTLE_SECONDS=${STRUCTURE_GPU_SETTLE_SECONDS:-15}
GPU_FREE_RETRIES=${STRUCTURE_GPU_FREE_RETRIES:-6}
GPU_FREE_RETRY_SECONDS=${STRUCTURE_GPU_FREE_RETRY_SECONDS:-5}
ETA_EVAL_RESERVE_SECONDS=${STRUCTURE_ETA_EVAL_RESERVE_SECONDS:-600}

if [ ! -x "$PYTHON" ]; then
  echo "[오류] $PYTHON 없음" >&2
  exit 1
fi

mkdir -p runs
exec 8>runs/.structure_search.lock
if ! flock -n 8; then
  echo "[오류] 구조 탐색 watcher가 이미 실행 중입니다." >&2
  exit 1
fi

active_pid=""

terminate_active_child() {
  if [ -z "$active_pid" ] || ! kill -0 "$active_pid" 2>/dev/null; then
    return
  fi
  echo "[중단] 구조 탐색 자식 process group $active_pid 종료 중..." >&2
  kill -TERM -- "-$active_pid" 2>/dev/null || kill -TERM "$active_pid" 2>/dev/null || true
  for _ in $(seq 1 20); do
    if ! kill -0 "$active_pid" 2>/dev/null; then
      wait "$active_pid" 2>/dev/null || true
      return
    fi
    sleep 0.5
  done
  echo "[중단] 자식이 TERM에 응답하지 않아 KILL합니다: $active_pid" >&2
  kill -KILL -- "-$active_pid" 2>/dev/null || kill -KILL "$active_pid" 2>/dev/null || true
  wait "$active_pid" 2>/dev/null || true
}

on_exit() {
  status=$?
  trap - EXIT HUP INT TERM
  terminate_active_child
  exit "$status"
}

trap on_exit EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

checkpoint_step() {
  "$PYTHON" - "$1" <<'PY'
import sys
import torch

state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(state.get("step", -1)))
PY
}

validate_tiny_completion() {
  "$PYTHON" - "$TINY_LAST" "$TINY_BEST" \
    "$EXPECTED_TINY_STEP" "$EXPECTED_TINY_MODEL" "$EXPECTED_PHYSICS" "$EXPECTED_LEAD" <<'PY'
import math
import sys
from pathlib import Path

import torch

last_path, best_path = (Path(value) for value in sys.argv[1:3])
expected_step = int(sys.argv[3])
expected_model = sys.argv[4]
expected_physics = sys.argv[5]
expected_lead = int(sys.argv[6])

errors = []


def load(path, label):
    if not path.is_file():
        errors.append(f"{label} checkpoint 없음: {path}")
        return {}
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        errors.append(f"{label} checkpoint 로드 실패: {path}: {exc}")
        return {}


def identity(state, label):
    cfg = state.get("cfg") or {}
    model_name = (cfg.get("model") or {}).get("name")
    physics = cfg.get("physics_status")
    lead = cfg.get(
        "digital_reference_lead_samples",
        (cfg.get("data") or {}).get("digital_reference_lead_samples"),
    )
    total = (cfg.get("schedule") or {}).get("total_steps")
    run_until = cfg.get("run_until_step")
    if run_until is None:
        run_until = total
    if model_name != expected_model:
        errors.append(f"{label} model.name={model_name!r} != {expected_model!r}")
    if physics != expected_physics:
        errors.append(f"{label} physics_status={physics!r} != {expected_physics!r}")
    if lead != expected_lead:
        errors.append(f"{label} lead={lead!r} != {expected_lead}")
    if total != expected_step:
        errors.append(f"{label} schedule.total_steps={total!r} != {expected_step}")
    if run_until != expected_step:
        errors.append(f"{label} run_until_step={run_until!r} != {expected_step}")


last = load(last_path, "last")
best = load(best_path, "best")
if last:
    identity(last, "last")
    if last.get("step") != expected_step:
        errors.append(f"last step={last.get('step')!r} != {expected_step}")
if best:
    identity(best, "best")
    best_step = best.get("step")
    if not isinstance(best_step, int) or not 1 <= best_step <= expected_step:
        errors.append(f"best step 범위 오류: {best_step!r}")

for label, state in (("last", last), ("best", best)):
    if not state:
        continue
    metric = state.get("best_metric")
    if not isinstance(metric, (int, float)) or not math.isfinite(float(metric)):
        errors.append(f"{label} best_metric이 유한하지 않습니다: {metric!r}")

if last and best:
    last_metric = float(last.get("best_metric", math.inf))
    best_metric = float(best.get("best_metric", math.inf))
    if not math.isclose(last_metric, best_metric, rel_tol=0.0, abs_tol=1e-6):
        errors.append(
            f"last/best best_metric 불일치: last={last_metric}, best={best_metric}"
        )

if errors:
    for error in errors:
        print(f"[오류] tiny 완료 gate: {error}", file=sys.stderr)
    raise SystemExit(2)

print(
    "tiny 완료 gate 통과: "
    f"step={last['step']}, model={expected_model}, physics={expected_physics}, "
    f"lead={expected_lead}, best_step={best['step']}, "
    f"best={float(best['best_metric']):.3f}dB"
)
PY
}

tiny_pid_is_expected() {
  local pid=$1
  local cmdline
  if [ ! -r "/proc/$pid/cmdline" ]; then
    return 1
  fi
  cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline")
  [[ "$cmdline" == *"scripts/train/train.py"* \
    && "$cmdline" == *"pretrain_tiny_corrected"* \
    && "$cmdline" == *"configs/model_tiny.yaml"* ]]
}

if [ -f "$TINY_PID_FILE" ]; then
  tiny_pid=$(tr -d '[:space:]' < "$TINY_PID_FILE")
  if [[ ! "$tiny_pid" =~ ^[0-9]+$ ]]; then
    echo "[오류] tiny PID 파일 형식 오류: $TINY_PID_FILE" >&2
    exit 1
  fi
  if kill -0 "$tiny_pid" 2>/dev/null; then
    if ! tiny_pid_is_expected "$tiny_pid"; then
      echo "[오류] PID $tiny_pid는 corrected tiny 학습이 아닙니다. stale PID 파일을 확인하세요." >&2
      exit 1
    fi
    echo "tiny PID $tiny_pid 정체성 확인 — 정상 완료 대기 중..."
    while kill -0 "$tiny_pid" 2>/dev/null; do
      sleep "$WAIT_POLL_SECONDS"
    done
  fi
fi

# 단순 프로세스 종료가 아니라 identity까지 검증한 정상 100k last/best를 authority로 쓴다.
validate_tiny_completion

gpu1_busy_rows() {
  nvidia-smi -i 1 \
    --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d; /No running processes found/d'
}

wait_for_gpu1_free() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[오류] nvidia-smi가 없어 GPU1 격리를 검증할 수 없습니다." >&2
    return 1
  fi
  if ! nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null \
      | grep -qx '1'; then
    echo "[오류] GPU1이 존재하지 않습니다 — 구조 탐색은 GPU1 전용입니다." >&2
    return 1
  fi

  local busy=""
  for _ in $(seq 1 "$GPU_FREE_RETRIES"); do
    busy=$(gpu1_busy_rows)
    if [ -z "$busy" ]; then
      return 0
    fi
    sleep "$GPU_FREE_RETRY_SECONDS"
  done
  echo "[오류] GPU1에 다른 compute process가 있어 구조 탐색을 시작하지 않습니다:" >&2
  printf '%s\n' "$busy" >&2
  return 1
}

# DataLoader 자식과 CUDA context가 정리될 여유. Base/GPU0는 계속 학습한다.
sleep "$GPU_SETTLE_SECONDS"
wait_for_gpu1_free

models=(
  configs/model_tiny.yaml
  configs/model_tiny_long.yaml
  configs/model_tiny_attn.yaml
  configs/model_tiny_long_attn.yaml
)
names=(tiny_control tiny_long tiny_attn tiny_long_attn)

# 모든 설정/산출물 경로를 먼저 검사해 일부 후보만 시작되는 상태를 막는다.
for model in "${models[@]}"; do
  if [ ! -f "$model" ]; then
    echo "[오류] 모델 설정 없음: $model" >&2
    exit 1
  fi
done
for name in "${names[@]}"; do
  for path in "runs/search_${name}" "runs/search_${name}.log" "runs/search_${name}.pid"; do
    if [ -e "$path" ] || [ -L "$path" ]; then
      echo "[오류] 기존 산출물을 덮어쓰지 않습니다: $path" >&2
      exit 1
    fi
  done
done

record_eta_probe() {
  local name=$1
  local index=$2
  local probe_step=$3
  local started_epoch=$4
  local run_dir=$5
  local now_epoch elapsed base_step base_rate eta_text
  now_epoch=$(date +%s)
  elapsed=$((now_epoch - started_epoch))
  if [ "$elapsed" -lt 1 ]; then
    elapsed=1
  fi

  base_step=-1
  base_rate=0
  if [ -f runs/train_base_corrected.log ]; then
    base_step=$(grep '^step ' runs/train_base_corrected.log | tail -n 1 | awk '{print $2}')
    base_rate=$(grep '^step ' runs/train_base_corrected.log | tail -n 1 | awk '{print $(NF-1)}')
    base_step=${base_step:--1}
    base_rate=${base_rate:-0}
  fi

  eta_text=$("$PYTHON" - "$name" "$index" "$probe_step" "$elapsed" \
    "$PILOT_STEPS" "${#models[@]}" "$ETA_EVAL_RESERVE_SECONDS" \
    "$base_step" "$base_rate" <<'PY'
import datetime as dt
import math
import sys
from zoneinfo import ZoneInfo

name = sys.argv[1]
index = int(sys.argv[2])
probe_step = int(sys.argv[3])
elapsed = max(1, int(sys.argv[4]))
pilot_steps = int(sys.argv[5])
model_count = int(sys.argv[6])
eval_reserve = int(sys.argv[7])
base_step = int(sys.argv[8])
base_rate = float(sys.argv[9])

rate = probe_step / elapsed
remaining_runs = model_count - index
remaining_train_steps = max(0, pilot_steps - probe_step) + max(0, remaining_runs - 1) * pilot_steps
projected_seconds = remaining_train_steps / max(rate, 1e-9) + remaining_runs * eval_reserve
now = dt.datetime.now(dt.timezone.utc)
kst = ZoneInfo("Asia/Seoul")
search_end = now + dt.timedelta(seconds=projected_seconds)

lines = [
    f"[ETA 진단] {name} step {probe_step}: {rate:.3f} it/s ({elapsed}s)",
    f"[ETA 진단] 현재 속도+후보당 평가 여유 {eval_reserve}s 기준 전체 탐색 종료 예상: "
    f"{search_end.astimezone(kst):%Y-%m-%d %H:%M:%S KST}",
]
if 0 <= base_step < 100000 and math.isfinite(base_rate) and base_rate > 0:
    base_end = now + dt.timedelta(seconds=(100000 - base_step) / base_rate)
    margin = (base_end - search_end).total_seconds()
    lines.append(
        f"[ETA 진단] base 종료 예상: {base_end.astimezone(kst):%Y-%m-%d %H:%M:%S KST}, "
        f"탐색 여유 {margin / 3600:+.2f}h"
    )
    if margin < 0:
        lines.append("[ETA 경고] 탐색이 base보다 늦을 전망이나 진단값이므로 현재 학습은 중단하지 않습니다.")
else:
    lines.append("[ETA 진단] base step/rate를 읽지 못해 base 대비 여유는 계산하지 못했습니다.")
print("\n".join(lines))
PY
  )
  # 학습 프로세스가 같은 log를 일반 write 모드로 열고 있으므로 동시 append하지 않는다.
  # 별도 파일과 watcher stdout에 기록해 학습 로그 offset 충돌을 피한다.
  printf '%s\n' "$eta_text" | tee "$run_dir/eta_probe.txt"
}

for index in "${!models[@]}"; do
  model=${models[$index]}
  name=${names[$index]}
  run_dir="runs/search_${name}"
  log="runs/search_${name}.log"
  pid_file="runs/search_${name}.pid"

  wait_for_gpu1_free
  echo "[$name] GPU1 구조 pilot 시작: schedule 100k, run_until $PILOT_STEPS"
  started_epoch=$(date +%s)
  CUDA_VISIBLE_DEVICES=1 setsid "$PYTHON" scripts/train/train.py \
    --config configs/train_pretrain.yaml \
    --set model_config="$model" --set ckpt_dir="$run_dir" \
    --set batch_size=128 --set num_workers=14 --set prefetch_factor=4 \
    --set schedule.total_steps=100000 --set schedule.warmup_steps=1250 \
    --set run_until_step="$PILOT_STEPS" \
    --set eval_every=500 --set early_stop_patience=0 \
    8>&- > "$log" 2>&1 < /dev/null &
  active_pid=$!
  printf '%s\n' "$active_pid" > "$pid_file"

  eta_recorded=0
  while kill -0 "$active_pid" 2>/dev/null; do
    state=$(ps -o stat= -p "$active_pid" 2>/dev/null | tr -d '[:space:]' || true)
    if [[ "$state" == Z* ]] || [ -z "$state" ]; then
      break
    fi
    last="$run_dir/ckpt/last.pt"
    if [ "$eta_recorded" -eq 0 ] && [ -f "$last" ]; then
      observed_step=$(checkpoint_step "$last" 2>/dev/null || printf '%s' -1)
      if [ "$observed_step" -ge "$ETA_PROBE_STEP" ]; then
        record_eta_probe \
          "$name" "$index" "$observed_step" "$started_epoch" "$run_dir"
        eta_recorded=1
      fi
    fi
    sleep "$ETA_POLL_SECONDS"
  done

  train_status=0
  wait "$active_pid" || train_status=$?
  active_pid=""
  if [ "$train_status" -ne 0 ]; then
    echo "[오류] $name 학습 실패(status=$train_status) — $log 확인. 다음 후보는 시작하지 않습니다." >&2
    exit 1
  fi

  last="$run_dir/ckpt/last.pt"
  best="$run_dir/ckpt/best.pt"
  if [ ! -f "$last" ] || [ "$(checkpoint_step "$last")" -ne "$PILOT_STEPS" ]; then
    echo "[오류] $name checkpoint가 $PILOT_STEPS step까지 완성되지 않았습니다." >&2
    exit 1
  fi
  if [ ! -f "$best" ]; then
    echo "[오류] $name best checkpoint가 없습니다." >&2
    exit 1
  fi
  if [ "$eta_recorded" -eq 0 ]; then
    record_eta_probe "$name" "$index" "$PILOT_STEPS" "$started_epoch" "$run_dir"
  fi

  # best 하나만 고르는 편향을 피하려고 같은 held-out 합성 평가를 best/last 모두 실행한다.
  for artifact in best last; do
    CUDA_VISIBLE_DEVICES=1 setsid "$PYTHON" scripts/eval/evaluate_offline.py \
      --ckpt "$run_dir/ckpt/${artifact}.pt" --n-items 32 \
      --out "$run_dir/eval_pilot_${artifact}" 8>&- >> "$log" 2>&1 < /dev/null &
    active_pid=$!
    eval_status=0
    wait "$active_pid" || eval_status=$?
    active_pid=""
    if [ "$eval_status" -ne 0 ]; then
      echo "[오류] $name/$artifact 평가 실패(status=$eval_status) — 다음 후보는 시작하지 않습니다." >&2
      exit 1
    fi
  done
  echo "[$name] 완료: $run_dir"
done

echo "tiny 대조군+구조 후보 3종 완료. 동일 20k best/last held-out 지표와 Jetson P99를 비교할 것."
