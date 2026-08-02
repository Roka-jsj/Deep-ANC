#!/bin/bash
# 공개 노이즈 데이터셋 다운로드 — Elice Cloud(디스크 128GB)에서 실행 권장.
# 사용: bash scripts/data/download_noise.sh [DNS_샤드수(기본 2)]
#
# 구성:
#   DNS-Challenge noise_fullband (48kHz) — 주 노이즈 풀, 샤드당 ~6GB
#   ESC-50 (44.1kHz → 로더에서 48kHz 리샘플) — 환경음 강건화용 (~600MB)
set -euo pipefail
cd "$(dirname "$0")/../.."

SHARDS="${1:-2}"
ROOT=data/raw/noise
mkdir -p "$ROOT/dns_fullband" "$ROOT/esc50"

DNS_BASE="https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband/noise_fullband"
for i in $(seq 0 $((SHARDS - 1))); do
  n=$(printf "%03d" "$i")
  f="datasets_fullband.noise_fullband.audioset_${n}.tar.bz2"
  if [ -d "$ROOT/dns_fullband/audioset_${n}" ]; then
    echo "[skip] $f (이미 존재)"
    continue
  fi
  echo "[down] $f"
  wget -c -O "$ROOT/$f" "$DNS_BASE/$f"
  mkdir -p "$ROOT/dns_fullband/audioset_${n}"
  tar -xjf "$ROOT/$f" -C "$ROOT/dns_fullband/audioset_${n}" --strip-components=0
  rm -f "$ROOT/$f"
done

if [ ! -d "$ROOT/esc50/ESC-50-master" ]; then
  echo "[down] ESC-50"
  wget -c -O "$ROOT/esc50.zip" "https://github.com/karolpiczak/ESC-50/archive/master.zip"
  unzip -q "$ROOT/esc50.zip" -d "$ROOT/esc50"
  rm -f "$ROOT/esc50.zip"
fi

echo "완료. 다음: python scripts/data/prepare_noise_pool.py"
