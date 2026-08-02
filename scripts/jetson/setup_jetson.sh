#!/bin/bash
# Jetson AGX Orin venv 셋업 (JetPack 6 / R36.4) — 시스템 변경 없음 (유저 공간만).
# 이미 구축된 .venv 를 재현해야 할 때 사용. 상세 배경: docs/06_deployment_jetson.md
set -euo pipefail
cd "$(dirname "$0")/../.."

# 1) venv (ensurepip 미설치 환경 대응: get-pip 부트스트랩)
python3 -m venv --without-pip --system-site-packages .venv
source .venv/bin/activate
curl -sS https://bootstrap.pypa.io/get-pip.py | python

# 2) 기본 의존성 (torch 제외)
pip install -r requirements-jetson.txt
pip install -e .

# 3) torch — NVIDIA 공식 JP6.1 wheel (jetson-ai-lab 최신은 libcupti 요구로 회피)
pip install --no-cache-dir \
  https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

# 4) wheel 이 요구하는 누락 라이브러리를 pip 로 확보 (apt 불필요)
pip install "nvidia-nvtx-cu12==12.6.77" nvidia-cuda-cupti-cu12 nvidia-cusparselt-cu12

# 5) 인터프리터 시작 시 자동 preload 훅 설치
SITE=.venv/lib/python3.10/site-packages
cat > "$SITE/_deep_anc_libpaths.py" <<'EOF'
"""NVIDIA JP6 torch wheel 이 요구하지만 시스템에 없는 라이브러리를 pip 패키지에서 preload."""
import ctypes, os
from pathlib import Path
_NV = Path(__file__).resolve().parent / "nvidia"
for _rel in ("nvtx/lib/libnvToolsExt.so.1",
             "cuda_cupti/lib/libcupti.so.12",
             "cusparselt/lib/libcusparseLt.so.0"):
    _p = _NV / _rel
    if _p.exists():
        try:
            ctypes.CDLL(str(_p), mode=os.RTLD_GLOBAL)
        except OSError:
            pass
EOF
echo "import _deep_anc_libpaths" > "$SITE/_deep_anc_libs.pth"

# 6) 검증
python - <<'EOF'
import torch, onnxruntime, scipy, numpy
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
print("ort", onnxruntime.__version__, "| numpy", numpy.__version__, "| scipy", scipy.__version__)
EOF
echo "셋업 완료. 다음: pytest -q"
