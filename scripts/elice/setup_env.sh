#!/bin/bash
# Elice Cloud (A100, VSCode CUDA 12.8 환경) 초기 셋업.
# 웹 VS Code 터미널에서:  bash scripts/elice/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

# venv 자체를 만드는 최초 한 번만 시스템 python3가 필요하다. 그 이후의 모든
# Python 실행은 인터프리터 혼선을 막기 위해 venv 경로를 명시한다.
python3 -m venv .venv 2>/dev/null || python3 -m venv --without-pip .venv
VENV_PYTHON="$PWD/.venv/bin/python"
SETUP_MARKER="$PWD/.venv/.setup-complete"
rm -f "$SETUP_MARKER"

"$VENV_PYTHON" -m pip -q install -U pip 2>/dev/null || {
  curl -fsS https://bootstrap.pypa.io/get-pip.py | "$VENV_PYTHON"
}

"$VENV_PYTHON" -m pip install -r requirements-train.txt
"$VENV_PYTHON" -m pip install -e .

"$VENV_PYTHON" - <<'EOF'
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

print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA를 사용할 수 없어 Elice 학습 환경 셋업을 완료하지 않습니다")
if torch.cuda.device_count() < 1:
    raise SystemExit("CUDA 장치가 없어 Elice 학습 환경 셋업을 완료하지 않습니다")
torch.empty(1, device="cuda")
for i in range(torch.cuda.device_count()):
    print(f"  GPU{i}:", torch.cuda.get_device_name(i))
EOF
touch "$SETUP_MARKER"
echo "셋업 완료. 다음 단계: bash scripts/data/download_noise.sh && $VENV_PYTHON scripts/data/prepare_noise_pool.py && $VENV_PYTHON scripts/data/build_rir_bank.py"
