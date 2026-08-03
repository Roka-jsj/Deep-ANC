#!/bin/bash
# ONNX → TensorRT FP16 엔진 빌드 (Jetson).
# 사용: bash scripts/export/build_trt.sh runs/export/model.onnx [출력.plan]
#
# 필요: 현재 환경에 사전 제공된 trtexec. 이 프로젝트에서는 apt/sudo로 설치하지 않는다.
set -euo pipefail
cd "$(dirname "$0")/../.."

ONNX="${1:?사용법: build_trt.sh <model.onnx> [out.plan]}"
PLAN="${2:-${ONNX%.onnx}_fp16.plan}"

TRTEXEC=""
for c in /usr/src/tensorrt/bin/trtexec "$(command -v trtexec || true)"; do
  if [ -n "$c" ] && [ -x "$c" ]; then TRTEXEC="$c"; break; fi
done
if [ -z "$TRTEXEC" ]; then
  echo "[오류] trtexec 이 없습니다. 프로젝트 정책상 apt/sudo 설치는 금지됩니다." >&2
  echo "  현행 배포 경로인 tiny + ONNX Runtime CPU를 사용하세요 (docs/06)." >&2
  exit 1
fi

"$TRTEXEC" \
  --onnx="$ONNX" \
  --saveEngine="$PLAN" \
  --fp16 --noTF32 \
  --useCudaGraph \
  --iterations=200 --avgRuns=50 \
  --exportTimes="${PLAN%.plan}_times.json"

# 엔진 메타(JSON)를 plan 옆에 복사 — TrtEngine 이 state_names 를 읽는다
META="${ONNX%.onnx}.json"
[ -f "$META" ] && cp "$META" "${PLAN%.plan}.json"
echo "완료: $PLAN"
echo "벤치: .venv/bin/python scripts/bench/measure_inference_latency.py --config configs/runtime.yaml --set engine.type=trt --set engine.plan=$PLAN"
