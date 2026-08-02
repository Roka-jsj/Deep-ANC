# HANDOFF — 세션 인수인계 (다음 AI 에이전트/개발자용)

> **"이어서 진행해줘"를 받았다면**: §2 현재 상태를 훑고 §3 다음 단계를 위에서부터 실행하라.
> 규칙은 [AGENTS.md](AGENTS.md)가 단일 출처. 이 파일은 작업 상태가 바뀔 때마다 갱신할 것.
> 최종 갱신: 2026-08-03 (Elice 학습 준비 단계)

## 1. 프로젝트 한 줄 요약

덕트(사각 아크릴 1.2m) 딥러닝 능동소음제어. 학습=Elice 2×A100, 추론=이 Jetson AGX Orin.
모델 HybridANCNet(tiny 1.16M / base 5.99M), digital-ref 모드 우선. 상세: docs/00, 물리: docs/01.

## 2. 현재 상태 (완료 ✅ / 진행 ⏳ / 대기 ⬜)

### 코드/저장소
- ✅ 저장소 구축 완료: 코드 전체 + 테스트 34종 통과 + 문서 12종. GitHub `Roka-jsj/Deep-ANC` push 완료
- ✅ 19-에이전트 리뷰로 결함 15건(치명 3) 수정 완료 — 지연 이중계상/gitignore/링버퍼 재동기 등
- ⬜ 커밋 이력의 "Co-Authored-By: Claude" 제거 — **사용자가 직접 실행해야 함** (권한 차단):
  `FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f --msg-filter 'sed "/Co-Authored-By: Claude/d"' main && git push -f origin main`
  (새 커밋에는 해당 줄을 넣지 않는다 — AGENTS.md 규칙 5)

### Jetson (이 PC)
- ✅ venv 동작: torch 2.5.0a0 CUDA OK, ORT 1.18.1, 테스트/스모크학습/export 모두 검증
- ✅ 추론 벤치: tiny+ORT CPU P99 1.50ms(실시간 게이트 통과) / base 6.8ms(TRT 필요)
- ⬜ **AB13X USB 오디오 미연결** → 실기 오디오 루프 미검증. 연결 후 docs/02 §3 절차 실행
- ⬜ 레퍼런스 마이크(ch1) 무신호 이력 — 수리 확인 전까지 digital-ref만

### Elice (학습)
- ✅ SSH 접속 확립: `ssh -i ~/.ssh/elice.pem -p 27914 elicer@central-01.tcp.tunnel.elice.io`
  (pem은 이 Jetson의 `~/.ssh/elice.pem`, **절대 커밋 금지**. ControlMaster 소켓 `~/.ssh/cm/` 재사용 권장 —
  터널 첫 접속이 10~20초 걸리고 간헐 실패하므로 재시도 로직 필수)
- ✅ 인스턴스: 2×A100 80GB/32vCore/RAM384G/디스크128G, `~/Deep-ANC`에 clone + venv(torch 2.5.1+cu121) 완료
- ⏳ 노이즈 다운로드: DNS shard000/001(각 5.4GB)을 `~/pget.py`(16-병렬 range, ~7MB/s)로 수신 중
  (일반 wget은 417KB/s로 막혀 있음 — 반드시 pget 사용), ESC-50 수신 중. 로그: `~/pget0.log ~/pget1.log ~/esc50.log`
- ⏳ RIR 뱅크(300개)+pytest 실행 중. 로그: `~/prep.log`
- ⬜ **본학습 미시작** — §3-A가 다음 작업

## 3. 다음 단계 (순서대로)

### A. Elice 본학습 시작 (데이터 수신 완료 후)

```bash
SSH="ssh -o ControlPath=~/.ssh/cm/%r@%h-%p -i ~/.ssh/elice.pem -p 27914 elicer@central-01.tcp.tunnel.elice.io"
# 1) 다운로드 완료 확인 (pget 로그에 DONE, esc50.log에 ESC50_DONE)
$SSH 'tail -1 ~/pget0.log ~/pget1.log ~/esc50.log; tail -2 ~/prep.log'
# 2) 샤드 해제 + manifest 생성
$SSH 'cd ~/Deep-ANC/data/raw/noise && mkdir -p dns_fullband && for f in shard*.tar.bz2; do tar -xjf $f -C dns_fullband && rm $f; done'
$SSH 'cd ~/Deep-ANC && source .venv/bin/activate && python scripts/data/prepare_noise_pool.py'
# 3) 모델 2종 병렬 학습 (사용자 지시: GPU0=base, GPU1=tiny 동시) — 각각 nohup+setsid
$SSH 'cd ~/Deep-ANC && source .venv/bin/activate && \
  CUDA_VISIBLE_DEVICES=0 setsid nohup python scripts/train/train.py --config configs/train_pretrain.yaml \
    --set num_workers=10 > runs/train_base.log 2>&1 < /dev/null & \
  CUDA_VISIBLE_DEVICES=1 setsid nohup python scripts/train/train.py --config configs/train_pretrain.yaml \
    --set model_config=configs/model_tiny.yaml --set ckpt_dir=runs/pretrain_tiny \
    --set batch_size=32 --set num_workers=10 > runs/train_tiny.log 2>&1 < /dev/null & echo STARTED'
# 4) 모니터링 (주기적으로): step 속도·val NMSE 하강 확인, GPU 활용률
$SSH 'tail -3 ~/Deep-ANC/runs/train_base.log ~/Deep-ANC/runs/train_tiny.log; nvidia-smi | head -20'
```

주의: Elice는 시간당 과금 — 학습 완료(조기종료 포함) 확인 즉시 결과 회수 후 사용자에게 인스턴스 중지를 안내할 것.

### B. 학습 완료 후 → Jetson 배포

```bash
# Elice에서: best.pt 존재 확인 후 회수 (scp — Jetson에서 실행)
scp -o ControlPath=~/.ssh/cm/%r@%h-%p -P 27914 elicer@central-01.tcp.tunnel.elice.io:~/Deep-ANC/runs/pretrain_tiny/ckpt/best.pt ~/Deep_ANC/runs/pretrain_tiny/ckpt/
scp ... pretrain_base ...  # 동일
# Jetson에서: export → 벤치 → (하드웨어 연결 후) 실기
cd ~/Deep_ANC && .venv/bin/python scripts/train/export_onnx.py --ckpt runs/pretrain_tiny/ckpt/best.pt --out runs/export/tiny.onnx
.venv/bin/python scripts/bench/measure_inference_latency.py --config configs/runtime.yaml --set engine.type=ort --set engine.onnx=runs/export/tiny.onnx
.venv/bin/python scripts/eval/evaluate_offline.py --ckpt runs/pretrain_tiny/ckpt/best.pt
.venv/bin/python scripts/eval/compare_fxlms.py --ckpt runs/pretrain_tiny/ckpt/best.pt
```

### C. 하드웨어 재연결 후 (사용자 입회 필요 — docs/02 §3 & docs/08 §5 체크리스트)

마이크 진단 → 무음 루프 → `--calibrate` 실효지연 → D_noise 실측 → 실측 녹음 → 파인튜닝 → evaluate_session

## 4. 조심할 것 (이번 세션에서 배운 것)

- Elice 터널 ssh는 느리고 간헐 실패 → ControlMaster + 원격 `setsid nohup ... < /dev/null &` + 로그 폴링 패턴 사용.
  로컬 타임아웃이 나도 원격 작업은 대부분 살아있다 — 재실행 전에 반드시 상태 확인 (중복 실행 주의)
- Azure blob 다운로드는 연결당 제한 → `~/pget.py` (범위 병렬) 필수
- duct.yaml의 에러마이크 위치(1.100)는 잠정값 — 사용자가 확정하면 RIR 뱅크 재생성+짧은 파인튜닝
- 사용자 답변 대기 중인 것 없음 (2026-08-03 현재)
