#!/bin/bash
# 새 Elice 인스턴스 원샷 부트스트랩 — 환경 + 데이터 + 병렬 학습까지 한 번에.
# 사용 (새 인스턴스의 홈에서):
#   git clone https://github.com/Roka-jsj/Deep-ANC.git && bash Deep-ANC/scripts/elice/bootstrap_all.sh
# 이미 실행한 적이 있으면 완전성이 검증된 단계만 건너뛴다 (재실행 안전).
#
# 데이터 구성 (48kHz):
#   DNS noise_fullband 2샤드(실환경 소음 ~11GB) + clean_fullband 음성 1샤드(~4.7GB)
#   + ESC-50(환경음) + FMA-small(음악) + DEMAND(실환경) + MIMII fan(기계소음)
# Azure blob 은 연결당 속도제한이 있어 반드시 pget.py(병렬 range)로 받는다.
set -euo pipefail

# --no-train: 환경·데이터·RIR·검증까지만 하고 학습은 띄우지 않는다.
# 사전학습이 끝난 뒤 새 인스턴스에서 **파인튜닝**을 돌릴 때 쓴다. 마지막 단계가
# 2-GPU 사전학습을 전제하므로 단일 GPU 인스턴스에서는 그대로 두면 실패한다.
START_TRAINING=1
for arg in "$@"; do
  case "$arg" in
    --no-train) START_TRAINING=0 ;;
    *) echo "[오류] 알 수 없는 인자: $arg" >&2; exit 2 ;;
  esac
done

REPO=~/Deep-ANC
cd "$REPO"

# pget 자체 잠금만으로는 wget .part, unzip 대상, manifest 생성을 보호할 수 없다.
# ignored data/ 아래의 고정 inode를 유지하고 셸 종료 시 커널이 잠금만 해제한다.
mkdir -p "$REPO/data"
exec 8>"$REPO/data/.bootstrap_all.lock"
if ! flock -n 8; then
  echo "[오류] 다른 bootstrap_all.sh가 이미 실행 중입니다. 중복 실행하지 않습니다." >&2
  exit 1
fi
active_train=$(pgrep -af '[t]rain\.py' || true)
if [ -n "$active_train" ]; then
  echo "[오류] 기존 train.py 학습이 실행 중이므로 데이터/manifest를 건드리지 않습니다:" >&2
  echo "$active_train" >&2
  exit 1
fi

# 복구 중인 원격 작업 사본에는 의도적인 로컬 수정이 있을 수 있다. 그 수정 위에
# pull/merge하지 않되, pull 실패를 숨기지도 않고 현재 검증된 체크아웃으로 계속한다.
if [ -n "$(git status --porcelain --untracked-files=normal)" ]; then
  echo "[경고] 로컬 변경이 있어 git pull 을 건너뜁니다. 현재 체크아웃으로 계속합니다." >&2
elif ! git pull --ff-only origin main; then
  echo "[경고] git pull --ff-only 실패. 현재 체크아웃으로 부트스트랩을 계속합니다." >&2
fi

VENV_PYTHON="$REPO/.venv/bin/python"
SETUP_MARKER="$REPO/.venv/.setup-complete"

environment_probe() {
  [ -x "$VENV_PYTHON" ] && "$VENV_PYTHON" - <<'PY' >/dev/null 2>&1
import deep_anc
import h5py
import matplotlib
import numpy
import onnx
import onnxruntime
import pytest
import scipy
import soundfile
import tensorboard
import torch
import tqdm
import yaml

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit(1)
torch.empty(1, device="cuda")
PY
}

environment_complete() {
  [ -f "$SETUP_MARKER" ] && environment_probe
}

echo "=== [1/6] 환경 (venv + torch cu121 + 패키지) ==="
# 완료 마커 도입 전부터 사용하던 인스턴스는 전체 요구 패키지와 CUDA probe를
# 통과하는 경우에만 마커를 이관한다. 유효한 환경을 불필요하게 재설치하지 않는다.
if [ ! -f "$SETUP_MARKER" ] && environment_probe; then
  touch "$SETUP_MARKER"
  echo "[setup] 기존 검증 완료 환경에 완료 마커를 생성했습니다."
fi
if ! environment_complete; then
  echo "[setup] 완료 마커/import/CUDA probe 중 하나가 유효하지 않아 환경을 구성합니다."
  bash scripts/elice/setup_env.sh
fi
if ! environment_complete; then
  echo "[오류] setup_env.sh 이후에도 Python/CUDA 환경 검증에 실패했습니다." >&2
  exit 1
fi

PGET=("$VENV_PYTHON" "$REPO/scripts/elice/pget.py")
DNS_BASE="https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband"
ZEN="https://zenodo.org/records"

echo "=== [2/6] 데이터 다운로드 (병렬) ==="
mkdir -p data/raw/noise
cd data/raw/noise

declare -A DL=(
  [shard000.tar.bz2]="$DNS_BASE/noise_fullband/datasets_fullband.noise_fullband.audioset_000.tar.bz2"
  [shard001.tar.bz2]="$DNS_BASE/noise_fullband/datasets_fullband.noise_fullband.audioset_001.tar.bz2"
  [speech000.tar.bz2]="$DNS_BASE/clean_fullband/datasets_fullband.clean_fullband.read_speech_000_0.00_3.75.tar.bz2"
)
declare -A DEST=(
  [shard000.tar.bz2]=dns_fullband
  [shard001.tar.bz2]=dns_fullband
  [speech000.tar.bz2]=speech
)
DEMAND_ENVIRONMENTS=(DKITCHEN DWASHING OOFFICE OHALLWAY TMETRO TCAR)

file_count() {
  local root=$1
  local pattern=$2
  if [ ! -d "$root" ]; then
    echo 0
    return
  fi
  find "$root" -type f -iname "$pattern" -print | wc -l
}

esc50_complete() {
  [ "$(file_count esc50/ESC-50-master/audio '*.wav')" -eq 2000 ] &&
    [ -s esc50/ESC-50-master/meta/esc50.csv ]
}

fma_complete() {
  [ "$(file_count music/fma_small '*.mp3')" -eq 8000 ]
}

demand_environment_complete() {
  local environment=$1
  [ "$(file_count "demand/$environment" '*.wav')" -eq 16 ]
}

demand_complete() {
  local environment
  for environment in "${DEMAND_ENVIRONMENTS[@]}"; do
    demand_environment_complete "$environment" || return 1
  done
}

mimii_complete() {
  [ "$(file_count machine '*.wav')" -eq 3600 ]
}

file_list_complete() {
  local marker=$1
  local destination=$2
  local relative

  [ -s "$marker" ] || return 1
  while IFS= read -r relative; do
    [ -n "$relative" ] || continue
    [ -s "$destination/$relative" ] || return 1
  done < "$marker"
}

dns_marker_complete() {
  local archive=$1
  local destination=$2
  file_list_complete "${archive}.extracted" "$destination"
}

zip_valid() {
  [ -f "$1" ] && unzip -tq "$1" >/dev/null 2>&1
}

# wget 결과는 최종 archive와 다른 .part에 받은 뒤 ZIP 검사를 통과한 경우에만
# 최종 이름으로 바꾼다. 중단된 .part는 다음 실행에서 안전하게 덮어쓴다.
ensure_wget_zip() {
  local url=$1
  local archive=$2
  local part="${archive}.part"
  if zip_valid "$archive"; then
    echo "[reuse] 무결성 확인된 $archive"
    return
  fi
  if ! wget -q -O "$part" "$url"; then
    echo "[오류] $archive 다운로드 실패" >&2
    rm -f "$part"
    return 1
  fi
  if ! unzip -tq "$part" >/dev/null 2>&1; then
    echo "[오류] 다운로드한 $archive ZIP 무결성 검사 실패" >&2
    rm -f "$part"
    return 1
  fi
  mv -f "$part" "$archive"
}

# pget.py는 archive.part에 모든 Range를 받은 뒤 바이트 수를 검증하고 원자적으로
# archive로 교체한다. 그 후 ZIP 자체도 검사해 이중으로 완전성을 확인한다.
ensure_pget_zip() {
  local url=$1
  local archive=$2
  local connections=$3
  if zip_valid "$archive"; then
    echo "[reuse] 무결성 확인된 $archive"
    return
  fi
  "${PGET[@]}" "$url" "$archive" "$connections"
  if ! unzip -tq "$archive" >/dev/null 2>&1; then
    echo "[오류] 다운로드한 $archive ZIP 무결성 검사 실패" >&2
    return 1
  fi
}

download_dns_archive() {
  local url=$1
  local archive=$2
  "${PGET[@]}" "$url" "$archive" 12
  if ! bzip2 -t "$archive"; then
    echo "[오류] 다운로드한 $archive bzip2 무결성 검사 실패" >&2
    rm -f "${archive}.done"
    return 1
  fi
  touch "${archive}.done"
}

download_esc50() {
  local archive=esc50.zip
  ensure_wget_zip \
    "https://codeload.github.com/karolpiczak/ESC-50/zip/refs/heads/master" \
    "$archive"
  mkdir -p esc50
  unzip -oq "$archive" -d esc50
  if ! esc50_complete; then
    echo "[오류] ESC-50 추출 불완전: WAV $(file_count esc50/ESC-50-master/audio '*.wav')/2000 또는 meta 누락" >&2
    return 1
  fi
  rm -f "$archive"
}

download_fma() {
  local archive=fma_small.zip
  ensure_pget_zip "https://os.unil.cloud.switch.ch/fma/fma_small.zip" "$archive" 8
  mkdir -p music
  unzip -oq "$archive" -d music
  if ! fma_complete; then
    echo "[오류] FMA-small 추출 불완전: MP3 $(file_count music/fma_small '*.mp3')/8000" >&2
    return 1
  fi
  rm -f "$archive"
}

download_demand() {
  local environment archive
  mkdir -p demand
  for environment in "${DEMAND_ENVIRONMENTS[@]}"; do
    if demand_environment_complete "$environment"; then
      echo "[skip] DEMAND $environment (WAV 16/16)"
      continue
    fi
    archive="demand/${environment}_48k.zip"
    ensure_wget_zip \
      "$ZEN/1227121/files/${environment}_48k.zip?download=1" \
      "$archive"
    unzip -oq "$archive" -d demand
    if ! demand_environment_complete "$environment"; then
      echo "[오류] DEMAND $environment 추출 불완전: WAV $(file_count "demand/$environment" '*.wav')/16" >&2
      return 1
    fi
    rm -f "$archive"
  done
}

download_mimii() {
  local archive=mimii_fan.zip
  ensure_pget_zip "$ZEN/6529888/files/fan.zip?download=1" "$archive" 8
  mkdir -p machine
  unzip -oq "$archive" -d machine
  if ! mimii_complete; then
    echo "[오류] MIMII fan 추출 불완전: WAV $(file_count machine '*.wav')/3600" >&2
    return 1
  fi
  rm -f "$archive"
}

pids=()
declare -A download_labels=()
start_download() {
  local label=$1
  shift
  "$@" &
  local pid=$!
  pids+=("$pid")
  download_labels["$pid"]=$label
}

for f in "${!DL[@]}"; do
  d=${DEST[$f]}
  if dns_marker_complete "$f" "$d"; then
    echo "[skip] $f (추출 파일 목록 검증 완료)"
    continue
  fi
  # 두 noise 샤드는 같은 대상 디렉터리를 사용한다. 따라서 대상에 WAV가 하나라도
  # 있다는 이유로 특정 샤드를 완료 처리하면 안 된다. 구버전의 무표식 추출본은
  # 보존한 채 해당 archive만 다시 받아 덮어 추출하고, 이후 파일 목록으로 검증한다.
  if [ -f "$f" ] && [ -f "${f}.done" ]; then
    echo "[skip] $f (다운로드+무결성 검사 완료)"
    continue
  fi
  if [ -f "$f" ] && bzip2 -t "$f"; then
    touch "${f}.done"
    echo "[reuse] $f (기존 archive 무결성 확인)"
    continue
  fi
  rm -f "${f}.done"
  start_download "DNS $f" download_dns_archive "${DL[$f]}" "$f"
done

if esc50_complete; then
  echo "[skip] ESC-50 (WAV 2000/2000 + meta)"
else
  start_download "ESC-50" download_esc50
fi

if fma_complete; then
  echo "[skip] FMA-small (MP3 8000/8000)"
else
  start_download "FMA-small" download_fma
fi

if demand_complete; then
  echo "[skip] DEMAND (6환경 × WAV 16 = 96/96)"
else
  start_download "DEMAND" download_demand
fi

# 기계소음: MIMII DG fan (16kHz, 저역 학습용 — QA 리포트에 표기됨)
if mimii_complete; then
  echo "[skip] MIMII fan (WAV 3600/3600)"
else
  start_download "MIMII fan" download_mimii
fi

download_failed=0
# 빈 배열이면 "${pids[@]}"는 0개 단어로 확장되므로 wait를 한 번도 호출하지 않는다.
for p in "${pids[@]}"; do
  if ! wait "$p"; then
    echo "[오류] ${download_labels[$p]} 다운로드/추출 실패 (PID $p) — 검증된 데이터는 보존됩니다." >&2
    download_failed=1
  fi
done
if [ "$download_failed" -ne 0 ]; then
  echo "[오류] 다운로드 단계가 완료되지 않았습니다. 네트워크 확인 후 bootstrap_all.sh 를 재실행하세요." >&2
  exit 1
fi

# skip 경로와 새 다운로드 경로 모두 같은 완전성 게이트를 마지막에 통과해야 한다.
if ! esc50_complete || ! fma_complete || ! demand_complete || ! mimii_complete; then
  echo "[오류] 데이터셋 완전성 최종 검사 실패" >&2
  exit 1
fi

echo "=== [3/6] DNS 샤드 무결성 검사 + 해제 ==="
for f in "${!DEST[@]}"; do
  if [ -f "$f" ] && ! bzip2 -t "$f"; then
    echo "[오류] $f 손상 — 다음 실행에서 재다운로드합니다." >&2
    rm -f "${f}.done"
    exit 1
  fi
done

# 어떤 archive의 목록 검증이 실패해도 이미 시작된 추출 작업이 남지 않도록
# 모든 목록을 먼저 검증한 뒤 두 번째 loop에서만 background 추출을 시작한다.
extract_archives=()
for f in "${!DEST[@]}"; do
  if [ -f "$f" ]; then
    d=${DEST[$f]}
    mkdir -p "$d"
    marker_building="${f}.extracted.building"
    rm -f "$marker_building"
    if ! tar -tjf "$f" | awk 'tolower($0) ~ /\.wav$/ { print }' > "$marker_building"; then
      echo "[오류] $f 파일 목록을 읽지 못했습니다." >&2
      for prepared in "${!DEST[@]}"; do
        rm -f "${prepared}.extracted.building"
      done
      rm -f "${f}.done"
      exit 1
    fi
    if [ ! -s "$marker_building" ]; then
      echo "[오류] $f 안에 WAV 파일이 없습니다." >&2
      for prepared in "${!DEST[@]}"; do
        rm -f "${prepared}.extracted.building"
      done
      rm -f "${f}.done"
      exit 1
    fi
    extract_archives+=("$f")
  fi
done

extract_pids=()
declare -A extract_files=()
for f in "${extract_archives[@]}"; do
  d=${DEST[$f]}
  tar -xjf "$f" -C "$d" &
  pid=$!
  extract_pids+=("$pid")
  extract_files["$pid"]=$f
done

extract_failed=0
for p in "${extract_pids[@]}"; do
  f=${extract_files[$p]}
  d=${DEST[$f]}
  marker_building="${f}.extracted.building"
  if wait "$p"; then
    if file_list_complete "$marker_building" "$d"; then
      mv -f "$marker_building" "${f}.extracted"
      rm -f "$f" "${f}.done"
      echo "[완료] $f 해제 및 파일 목록 검증 (PID $p)"
    else
      echo "[오류] $f 해제 후 파일 목록 검증 실패 — archive를 보존합니다." >&2
      rm -f "$marker_building" "${f}.done"
      extract_failed=1
    fi
  else
    echo "[오류] $f 해제 실패 (PID $p) — archive를 보존합니다." >&2
    rm -f "$marker_building" "${f}.done"
    extract_failed=1
  fi
done
if [ "$extract_failed" -ne 0 ]; then
  echo "[오류] DNS 샤드 해제가 완료되지 않았습니다. bootstrap_all.sh 를 재실행하세요." >&2
  exit 1
fi
cd "$REPO"

echo "=== [4/6] manifest + RIR 뱅크 + 데이터셋 QA ==="
"$VENV_PYTHON" scripts/data/prepare_noise_pool.py
RIR_BANK=data/rir_bank/duct_rirs_v1.npz
RIR_BUILDING=data/rir_bank/duct_rirs_v1.building.npz
rir_bank_complete() {
  local path=$1
  [ -f "$path" ] && "$VENV_PYTHON" - "$path" <<'PY' >/dev/null 2>&1
import sys

import numpy as np

expected_shape = (300, 8192)
with np.load(sys.argv[1]) as bank:
    for key in ("p_ref", "p_err", "f_fb"):
        value = bank[key]
        if value.shape != expected_shape or not np.isfinite(value).all():
            raise SystemExit(1)
PY
}
if ! rir_bank_complete "$RIR_BANK"; then
  echo "[build] RIR 뱅크가 없거나 불완전하여 300개를 임시 파일에 재생성합니다."
  mkdir -p "$(dirname "$RIR_BANK")"
  rm -f "$RIR_BUILDING"
  "$VENV_PYTHON" scripts/data/build_rir_bank.py --n 300 --out "$RIR_BUILDING"
  if ! rir_bank_complete "$RIR_BUILDING"; then
    echo "[오류] 새로 생성한 RIR 뱅크 shape/finite 검증 실패" >&2
    rm -f "$RIR_BUILDING"
    exit 1
  fi
  mv -f "$RIR_BUILDING" "$RIR_BANK"
fi
"$VENV_PYTHON" scripts/data/validate_noise_pool.py

echo "=== [5/6] 검증 (pytest) ==="
"$VENV_PYTHON" -m pytest -q

if [ "$START_TRAINING" -eq 0 ]; then
  echo "=== [6/6] 건너뜀 (--no-train) ==="
  echo "부트스트랩 완료 — 데이터/RIR/검증까지 준비됐습니다."
  echo "파인튜닝: scripts/train/run_finetune_pipeline.py --config configs/train_finetune.yaml \\"
  echo "            --set data.digital_primary_path_mode=measured"
  exit 0
fi

echo "=== [6/6] 병렬 학습 시작 (GPU0=base, GPU1=tiny) ==="
# bootstrap 잠금 FD는 백그라운드 학습 프로세스에 상속하지 않는다.
bash scripts/elice/run_parallel_models.sh 8>&-
echo "부트스트랩 완료 — 모니터링: tail -f runs/train_base_corrected.log runs/train_tiny_corrected.log"
