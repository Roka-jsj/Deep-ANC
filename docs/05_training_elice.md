# 05. Elice Cloud 학습 가이드 (A100)

## 1. 인스턴스 구성 (확정)

| 항목 | 값 |
|---|---|
| 인스턴스 | **G-NAHP-160** — 2× A100 80GB PCIe, 32 vCore, 384GiB RAM (₩3,980/시간) |
| 실행 환경 | **VSCode (CUDA 12.8)** — torch cu121 wheel 과 호환 (드라이버 하위호환) |
| 스토리지 | **128 GiB** (₩19.2/시간) — 온더플라이 합성 설계라 충분. 업그레이드만 가능하므로 작게 시작 |

**비용 팁**
- 스토리지·인스턴스 모두 시간당 과금 — **켜둔 채 방치 금지**.
- 코드 디버깅/소규모 실험은 **G-NAHPM-10 (MIG 1g-10GB, ₩340/시간)** 으로:
  `--set batch_size=4` 만 바꾸면 같은 코드가 그대로 돈다.
- 본 학습(400k step)만 2×A100 으로. 예상 1~2일 → 인스턴스 비용 약 ₩10~20만.

## 2. 초기 셋업 (웹 VS Code 터미널)

```bash
git clone https://github.com/<계정>/Deep_ANC.git && cd Deep_ANC
bash scripts/elice/setup_env.sh          # venv + requirements-train.txt + pip -e .
bash scripts/data/download_noise.sh 2    # DNS 2샤드(~12GB) + ESC-50. 샤드 수 조절 가능
python scripts/data/prepare_noise_pool.py
python scripts/data/build_rir_bank.py --n 300
pytest -q                                # 30+ 테스트로 환경 검증
```

인터넷이 막힌 인스턴스라면: Jetson 에서 `python scripts/data/pack_transfer.py` 로 만든
tar 샤드를 VS Code 탐색기로 업로드 → `for f in transfer/*.tar; do tar -xf "$f"; done`.

## 3. 학습 실행

```bash
# Stage-1 사전학습 (GPU 수 자동 감지 → 2장이면 torchrun DDP)
bash scripts/elice/run_pretrain.sh
# 모니터링
tail -f runs/train_*.log
tensorboard --logdir runs --port 6006     # VS Code 포트포워딩으로 브라우저 접속
# 세션이 끊겨도 nohup 으로 계속 돈다. 재개:
python scripts/train/train.py --config configs/train_pretrain.yaml --resume runs/pretrain_base/ckpt/last.pt
```

주요 하이퍼파라미터는 `configs/train_pretrain.yaml` (batch 24/GPU, AdamW 1e-3,
warmup 5k → cosine, bf16 AMP, 400k step, val 미개선 10회 조기종료).

### Stage-2 폐루프 파인튜닝 (선택 — 시뮬 피드백 동역학 학습)

```bash
python scripts/train/train.py --config configs/train_finetune.yaml \
    --set stage=closed_loop --set "schedule.total_steps=30000"
```

프레임 순차 unroll 이라 step 당 수 배 느리다 — 20k~50k step 권장 (설계 H1).

### 실측 파인튜닝 (덕트 녹음 후)

Jetson 에서 수집한 `data/recorded/` + manifest 를 git/zip 으로 올린 뒤:

```bash
python scripts/train/train.py --config configs/train_finetune.yaml
```

## 4. 결과 회수 → Jetson

```bash
python scripts/train/export_onnx.py --ckpt runs/pretrain_base/ckpt/best.pt --out runs/export/model.onnx
# runs/export/{model.onnx, model.json} + ckpt/best.pt 를 zip 으로 다운로드
# (수십 MB — VS Code 탐색기 우클릭 Download, 또는 GitHub Release 자산으로 업로드)
```

Jetson 쪽 배포는 docs/06 참조.

## 5. 자주 겪는 문제

| 증상 | 조치 |
|---|---|
| CUDA OOM | `--set batch_size=16` (또는 MIG 에선 4) |
| DataLoader 병목 (GPU util 낮음) | `--set num_workers=16` (A100 32vCore 기준) |
| val NMSE 가 0dB 근처 정체 | 정상적 초기 구간 (수만 step 후 하강). 지속되면 lr/데이터 구성 점검 |
| torch 버전 충돌 | requirements-train.txt 는 2.5.1+cu121 고정 — Jetson(2.5.0a0)과 정렬 |
| 세션 종료로 학습 중단 | run_pretrain.sh 는 nohup — 재접속 후 tail 로 확인, 필요시 --resume |
