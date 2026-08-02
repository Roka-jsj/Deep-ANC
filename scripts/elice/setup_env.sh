#!/bin/bash
# Elice Cloud (A100, VSCode CUDA 12.8 환경) 초기 셋업.
# 웹 VS Code 터미널에서:  bash scripts/elice/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

python3 -m venv .venv 2>/dev/null || python3 -m venv --without-pip .venv
source .venv/bin/activate
python -m pip -q install -U pip 2>/dev/null || {
  curl -sS https://bootstrap.pypa.io/get-pip.py | python
}

pip install -r requirements-train.txt
pip install -e .

python - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU{i}:", torch.cuda.get_device_name(i))
EOF
echo "셋업 완료. 다음 단계: bash scripts/data/download_noise.sh && python scripts/data/prepare_noise_pool.py && python scripts/data/build_rir_bank.py"
