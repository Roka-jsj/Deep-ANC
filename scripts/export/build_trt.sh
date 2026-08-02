#!/bin/bash
# ONNX → TensorRT FP16 엔진 빌드 (Jetson).
# 사용: bash scripts/export/build_trt.sh runs/export/model.onnx [출력.plan]
#
# 필요: trtexec (apt libnvinfer-bin — sudo 필요, 시스템 변경이므로 사용자 판단.
#       프로젝트 정책상 자동 설치하지 않는다. docs/06 참조)
set -euo pipefail
cd "$(dirname "$0")/../.."

ONNX="${1:?사용법: build_trt.sh <model.onnx> [out.plan]}"
PLAN="${2:-${ONNX%.onnx}_fp16.plan}"

TRTEXEC=""
for c in /usr/src/tensorrt/bin/trtexec "$(command -v trtexec || true)"; do
  if [ -n "$c" ] && [ -x "$c" ]; then TRTEXEC="$c"; break; fi
done
if [ -z "$TRTEXEC" ]; then
  echo "[오류] trtexec 이 없습니다. TensorRT 실행 도구가 필요합니다:" >&2
  echo "  sudo apt-get install libnvinfer-bin python3-libnvinfer   # 시스템 변경 — 사용자 판단" >&2
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
echo "벤치: python scripts/bench/measure_inference_latency.py --config configs/runtime.yaml --set engine.type=trt --set engine.plan=$PLAN"
