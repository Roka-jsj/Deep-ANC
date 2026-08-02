# HANDOFF — 세션 인수인계 (다음 AI 에이전트/개발자용)

> **"이어서 진행해줘"를 받았다면**: §2 현재 상태를 훑고 §3 다음 단계를 위에서부터 실행하라.
> 규칙은 [AGENTS.md](AGENTS.md)가 단일 출처. 이 파일은 작업 상태가 바뀔 때마다 갱신할 것.
> 최종 갱신: 2026-08-03 (Elice 1차 인스턴스 삭제됨 — 새 인스턴스 대기 상태)

## 1. 프로젝트 한 줄 요약

덕트(사각 아크릴 1.2m) 딥러닝 능동소음제어. 학습=Elice 2×A100, 추론=이 Jetson AGX Orin.
모델 HybridANCNet(tiny 1.16M=현행 실시간 / base 5.99M=TRT 목표), digital-ref 모드 우선.
상세: docs/00, 물리: docs/01, 구조 지도: docs/10.

## 2. 현재 상태

### 완료 ✅
- 저장소 완성: 코드 + 테스트 30+종 + 문서 13종, GitHub `Roka-jsj/Deep-ANC` push 완료
- 19-에이전트 리뷰(결함 15건) + 5-에이전트 구조 감사(이슈 35건) 모두 수정 반영
- Jetson venv 검증: torch 2.5.0a0 CUDA OK, 스모크학습/ONNX export(ORT 등가 2.4e-8)/벤치 완료
  (tiny+ORT CPU P99 1.50ms → TRT 없이 실시간 게이트 통과)
- 원샷 부트스트랩 완성: `scripts/elice/bootstrap_all.sh` — 환경+데이터 7종 병렬 다운로드
  +QA+테스트+2-GPU 병렬 학습까지 자동 (다운로드는 반드시 `scripts/elice/pget.py` 병렬 range —
  Azure 는 단일 연결 417KB/s 로 제한됨, pget 으로 ~7MB/s 실측)
- 데이터셋 확정(검증된 URL): DNS 소음 2샤드(각 5.4GB) + DNS 음성 1샤드(4.7GB) + ESC-50
  + FMA-small 음악(7.2GB) + DEMAND 48k 6환경 + MIMII fan(16k, 저역 기계음) ≈ 총 26GB
- 데이터 QA 게이트: `scripts/data/validate_noise_pool.py` (부트스트랩에 포함)

### 대기 ⬜ (다음 세션이 할 일 순서)
1. **연구 워크플로 결과 반영** — 비주기·비선형 v2 로드맵(docs/11) 작성 대기 중이었음.
   결과가 없으면 재실행: 전문가 4방향(deep-anc/aperiodic/nonlinear/adaptive) → 설계 → 3렌즈 검증
2. **사용자에게 새 인스턴스 요청** → §3-A 실행
3. 학습 완료 후 → §3-B 회수/배포, 완료 즉시 사용자에게 **인스턴스 중지/삭제 안내** (시간과금!)
4. 하드웨어 재연결 후 → §3-C (USB 오디오 AB13X 미연결 상태, ref 마이크 ch1 무신호 이력)

### 사용자가 직접 해야 하는 것 (권한/자원 소유)
- 새 Elice 인스턴스 생성 (G-NAHP-160 권장, SSH-Only 환경) + 접속 주소/포트 전달
- 기존 커밋 이력의 "Co-Authored-By: Claude" 제거 (권한 차단으로 에이전트 실행 불가):
  `FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f --msg-filter 'sed "/Co-Authored-By: Claude/d"' main && git push -f origin main`
- 덕트 미확정값 확정 시 통보 (에러마이크 X=1.100 잠정 — 확정 시 duct.yaml + RIR 뱅크 재생성)

## 3. 실행 절차

### A. 새 Elice 인스턴스에서 학습 (원샷)

```bash
# Jetson 쪽 SSH 변수 (ControlMaster 포함 — 터널 첫 접속 10~20초·간헐 실패 → 항상 재시도 3회)
SSH="ssh -i ~/.ssh/elice.pem -o StrictHostKeyChecking=accept-new -o BatchMode=yes \
  -o ControlMaster=auto -o ControlPath=~/.ssh/cm/%r@%h-%p -o ControlPersist=600 \
  -p <포트> elicer@<호스트>"
# pem: ~/.ssh/elice.pem (계정 소속 키 — 새 인스턴스에서 바뀌면 사용자에게 재다운로드 요청.
#  저장소 커밋 절대 금지 — .gitignore *.pem)

# 1) 원샷 실행 (원격 setsid nohup — SSH 끊겨도 계속. 약 40-50분 후 학습 자동 시작)
$SSH 'git clone -q https://github.com/Roka-jsj/Deep-ANC.git 2>/dev/null; \
  setsid nohup bash Deep-ANC/scripts/elice/bootstrap_all.sh > ~/bootstrap.log 2>&1 < /dev/null & echo STARTED'
# 2) 진행 폴링 (주기적으로; tail 은 반드시 -n 구문)
$SSH 'tail -n 3 ~/bootstrap.log'
# 3) 학습 시작 후 모니터링: step 증가·val NMSE 하강·GPU 활용률
$SSH 'tail -n 3 ~/Deep-ANC/runs/train_base.log ~/Deep-ANC/runs/train_tiny.log; \
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader'
# 이상 징후: bootstrap.log 에 "[오류]" / 학습 로그 정지 / val 미하강 지속 → HANDOFF §4 참조
```

### B. 학습 완료 후 → Jetson 배포

```bash
mkdir -p ~/Deep_ANC/runs/pretrain_base/ckpt ~/Deep_ANC/runs/pretrain_tiny/ckpt
scp -o ControlPath=~/.ssh/cm/%r@%h-%p -P <포트> \
  elicer@<호스트>:~/Deep-ANC/runs/pretrain_tiny/ckpt/best.pt ~/Deep_ANC/runs/pretrain_tiny/ckpt/
# (base 동일)  → 이 시점에 사용자에게 "인스턴스 중지/삭제" 안내!
cd ~/Deep_ANC && .venv/bin/python scripts/train/export_onnx.py --ckpt runs/pretrain_tiny/ckpt/best.pt --out runs/export/tiny.onnx
.venv/bin/python scripts/bench/measure_inference_latency.py --config configs/runtime.yaml --set engine.type=ort --set engine.onnx=runs/export/tiny.onnx
.venv/bin/python scripts/eval/evaluate_offline.py --ckpt runs/pretrain_tiny/ckpt/best.pt
.venv/bin/python scripts/eval/compare_fxlms.py --ckpt runs/pretrain_tiny/ckpt/best.pt
```

### C. 하드웨어 재연결 후 (사용자 입회) — docs/02 §3 + docs/08 §5 체크리스트

## 4. 조심할 것 (세션에서 배운 것)

- Elice 터널: 로컬 타임아웃이 나도 **원격 작업은 대부분 살아있다** — 재실행 전 반드시 상태 확인
  (중복 실행 = 같은 로그 겹쳐쓰기·불완전 파일). 원격 장기작업은 `setsid nohup … < /dev/null &` 패턴만
- 다운로드 무결성: 해제 전 `bzip2 -t` (부트스트랩 [3/6]에 내장)
- `tail -n 1` (구식 `tail -1` 은 다중 파일에서 GNU 오류)
- Elice 는 **인스턴스 켜진 시간 과금** (SSH 연결 여부 무관). 종료(중지)해도 스토리지는 과금, 삭제만 완전 중지
- Jetson venv 재구성은 `scripts/jetson/setup_jetson.sh` (lib preload 훅 필수), ORT 1.18.1 고정
- S(z)/핸드오프/목표대역은 단일 출처 원칙 (duct.yaml + config.DEFAULT_HANDOFF_SAMPLES) — docs/10 소비 지도 참조
