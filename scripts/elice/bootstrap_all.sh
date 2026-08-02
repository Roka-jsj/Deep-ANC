#!/bin/bash
# 새 Elice 인스턴스 원샷 부트스트랩 — 환경 + 데이터 + 병렬 학습까지 한 번에.
# 사용 (새 인스턴스의 홈에서):
#   git clone https://github.com/Roka-jsj/Deep-ANC.git && bash Deep-ANC/scripts/elice/bootstrap_all.sh
# 이미 실행한 적이 있으면 완료된 단계는 건너뛴다 (재실행 안전).
#
# 데이터 구성 (48kHz):
#   DNS noise_fullband 2샤드(실환경 소음 ~11GB) + clean_fullband 음성 1샤드(~4.7GB)
#   + ESC-50(환경음) [+ music 은 준비되면 data/raw/noise/music/ 에 넣기만 하면 자동 인식]
# Azure blob 은 연결당 속도제한이 있어 반드시 pget.py(병렬 range)로 받는다.
set -euo pipefail

REPO=~/Deep-ANC
cd "$REPO"
git pull -q origin main || true          # 최신 코드 반영 (기존 clone 재사용 시)
PGET="python3 $REPO/scripts/elice/pget.py"
DNS_BASE="https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband"
ZEN="https://zenodo.org/records"

echo "=== [1/6] 환경 (venv + torch cu121 + 패키지) ==="
if [ ! -f .venv/bin/python ]; then
  bash scripts/elice/setup_env.sh
fi
source .venv/bin/activate

echo "=== [2/6] 데이터 다운로드 (병렬) ==="
mkdir -p data/raw/noise && cd data/raw/noise
declare -A DL=(
  [shard000.tar.bz2]="$DNS_BASE/noise_fullband/datasets_fullband.noise_fullband.audioset_000.tar.bz2"
  [shard001.tar.bz2]="$DNS_BASE/noise_fullband/datasets_fullband.noise_fullband.audioset_001.tar.bz2"
  [speech000.tar.bz2]="$DNS_BASE/clean_fullband/datasets_fullband.clean_fullband.read_speech_000_0.00_3.75.tar.bz2"
)
declare -A DEST=([shard000.tar.bz2]=dns_fullband [shard001.tar.bz2]=dns_fullband [speech000.tar.bz2]=speech)
pids=()
for f in "${!DL[@]}"; do
  d="${DEST[$f]}"
  if [ -d "$d" ] && [ -n "$(find "$d" -name '*.wav' -print -quit 2>/dev/null)" ] && [ ! -f "$f" ]; then
    echo "[skip] $f (이미 해제됨)"; continue
  fi
  [ -f "$f.done" ] && { echo "[skip] $f (다운로드 완료 마커)"; continue; }
  ( $PGET "${DL[$f]}" "$f" 12 && touch "$f.done" ) &
  pids+=($!)
done
if [ ! -d esc50/ESC-50-master ]; then
  ( wget -q -O esc50.zip https://codeload.github.com/karolpiczak/ESC-50/zip/refs/heads/master \
    && unzip -q esc50.zip -d esc50 && rm esc50.zip ) &
  pids+=($!)
fi
# 음악: FMA-small (44.1k mp3 — soundfile/libsndfile 이 직접 디코딩, 로더가 48k 리샘플)
if [ ! -d music/fma_small ]; then
  ( $PGET "https://os.unil.cloud.switch.ch/fma/fma_small.zip" fma_small.zip 8 \
    && mkdir -p music && unzip -q fma_small.zip -d music && rm fma_small.zip ) &
  pids+=($!)
fi
# 지속성 실환경 (DEMAND 48k, 6환경: 주방/세탁기/사무실/복도/지하철/차량)
if [ ! -d demand ]; then
  ( mkdir -p demand && cd demand && \
    for e in DKITCHEN DWASHING OOFFICE OHALLWAY TMETRO TCAR; do
      wget -q -O "${e}_48k.zip" "$ZEN/1227121/files/${e}_48k.zip?download=1" \
        && unzip -q "${e}_48k.zip" && rm "${e}_48k.zip"
    done ) &
  pids+=($!)
fi
# 기계소음: MIMII DG fan (16kHz ⚠ 저역 전용 — QA 리포트에 표기됨)
if [ ! -d machine ]; then
  ( $PGET "$ZEN/6529888/files/fan.zip?download=1" mimii_fan.zip 8 \
    && mkdir -p machine && unzip -q mimii_fan.zip -d machine && rm mimii_fan.zip ) &
  pids+=($!)
fi
for p in "${pids[@]:-}"; do wait "$p"; done

echo "=== [3/6] 샤드 무결성 검사 + 해제 ==="
for f in "${!DEST[@]}"; do
  if [ -f "$f" ]; then
    bzip2 -t "$f" || { echo "[오류] $f 손상 — 재다운로드 필요"; rm -f "$f.done"; exit 1; }
  fi
done
for f in "${!DEST[@]}"; do
  if [ -f "$f" ]; then
    d="${DEST[$f]}"; mkdir -p "$d"
    tar -xjf "$f" -C "$d" && rm -f "$f" "$f.done" &
  fi
done
wait
cd "$REPO"

echo "=== [4/6] manifest + RIR 뱅크 + 데이터셋 QA ==="
python scripts/data/prepare_noise_pool.py
[ -f data/rir_bank/duct_rirs_v1.npz ] || python scripts/data/build_rir_bank.py --n 300
python scripts/data/validate_noise_pool.py   # 학습·추론 적합성 리포트 (치명 시 중단)

echo "=== [5/6] 검증 (pytest) ==="
python -m pytest -q

echo "=== [6/6] 병렬 학습 시작 (GPU0=base, GPU1=tiny) ==="
bash scripts/elice/run_parallel_models.sh
echo "부트스트랩 완료 — 모니터링: tail -f runs/train_base.log runs/train_tiny.log"
