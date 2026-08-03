#!/bin/bash
# 과거 300 Hz / 약 2 dB FxLMS 기준선을 원본 디렉터리 무수정으로 재현한다.
# 스피커 출력 스크립트: 사용자 입회 + 앰프 볼륨 최저 상태에서만 실행.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO_ROOT"

if [ "${1:-}" != "--confirm-volume-minimum" ]; then
  echo "[중단] 사용자 입회와 앰프 볼륨 최저를 확인한 뒤 --confirm-volume-minimum을 지정하세요." >&2
  exit 2
fi

LEGACY_SCRIPT=/home/capston/anc_project/main_realtime_anc.py
LEGACY_MODEL="$REPO_ROOT/assets/measured/secondary_path_legacy_512high.npz"
WEIGHTS_OUT="$REPO_ROOT/results/legacy_fxlms/control_filter_last.npy"

if [ ! -f "$LEGACY_SCRIPT" ]; then
  echo "[오류] 읽기 전용 legacy 스크립트가 없습니다: $LEGACY_SCRIPT" >&2
  exit 1
fi
if [ ! -f "$LEGACY_MODEL" ]; then
  echo "[오류] 저장소 내부 legacy S(z) 복사본이 없습니다: $LEGACY_MODEL" >&2
  exit 1
fi

# 이 명령은 출력 장치를 열지 않는다. ERR ch0가 고정값/무음/클리핑이면 여기서 종료한다.
"$REPO_ROOT/.venv/bin/python" scripts/bench/check_audio_input.py

mkdir -p "$(dirname "$WEIGHTS_OUT")"
echo "[안전] 입력 PASS. ANC는 OFF로 시작합니다. A/Space=ON/OFF, Q=종료."

# -B/PYTHONDONTWRITEBYTECODE로 읽기 전용 anc_project에 __pycache__를 쓰지 않는다.
# 모델과 정상 종료 weight도 모두 Deep_ANC 내부 경로로 명시한다.
exec env PYTHONDONTWRITEBYTECODE=1 python3 -B "$LEGACY_SCRIPT" \
  --model "$LEGACY_MODEL" \
  --noise-type tone \
  --frequency 300 \
  --noise-amplitude 0.05 \
  --noise-delay-ms 70 \
  --mu 0.001 \
  --control-limit 0.10 \
  --block-size 512 \
  --latency high \
  --weights-output "$WEIGHTS_OUT"
