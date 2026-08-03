# HANDOFF — 세션 인수인계 (다음 AI 에이전트/개발자용)

> **"이어서 진행해줘"를 받았다면**: §0 라이브 상태 → §2 현재 상태 → §3 다음 단계 순으로 실행하라.
> 규칙은 [AGENTS.md](AGENTS.md)가 단일 출처. 이 파일은 작업 상태가 바뀔 때마다 갱신할 것.
> 최종 갱신: 2026-08-04 01:00 KST

## 0. 라이브 상태 (가장 먼저 확인할 것 — 시각은 참고용, 실상태는 아래 명령으로)

- **Elice 인스턴스**: `elicer@central-01.tcp.tunnel.elice.io` **포트 47863**, 2×A100 80GB.
  pem = 이 Jetson의 `~/.ssh/elice.pem` (커밋 금지). 32 vCPU, 디스크 84G 여유.
- **GPU 작업 큐 감독자 2기 가동 중** (이번 세션에 새로 배포):
  - GPU1 감독자 PID **28755**, 큐 `configs/elice/queue_gpu1.yaml`
  - GPU0 감독자 PID **28685**, 큐 `configs/elice/queue_gpu0.yaml`
  - 둘 다 **진입 게이트에서 대기 중**이며 기존 학습/watcher를 전혀 건드리지 않는다.
  - PID는 재기동하면 바뀐다. 권위 있는 소유자는 `runs/.job_queue_gpu{0,1}.lock`의 owner JSON이다.
  - **`tier` 필드는 현재 메타데이터일 뿐 강제되지 않는다.** 감독자는 큐에 적힌 순서대로
    실행한다. Tier-B를 "다른 GPU의 Tier-A ETA 안에서만"으로 제한하려면 별도 구현이 필요하다.
    지금 큐는 순서 자체가 우선순위를 반영하도록 배열해 뒀다.
- **현재 GPU 점유** (감독자가 이어받기 전):
  - GPU0: base `train.py` PID **22554**, 100k 목표. 완료 예상 **8/4 04:45 KST 전후**
  - GPU1: 구 watcher `run_structure_search.sh` PID **24271**. tiny_long_attn 20k 완주 중,
    완료 예상 **8/4 01:16 KST 전후** → 평가 후 watcher 종료 → GPU1 감독자가 인계
- **인계 후 자동 진행 순서**:
  - GPU1: `search_tiny_control` 20k(부재하던 대조군) → `select_structure_winner`(사전 등록 규칙)
  - GPU0: base best/last held-out 평가 → seed 반복 → 회수 번들(SHA-256 목록)

### 상태 확인 (가장 먼저 실행)

```bash
SSH="ssh -i ~/.ssh/elice.pem -o StrictHostKeyChecking=accept-new -o BatchMode=yes \
  -o ControlMaster=auto -o ControlPath=~/.ssh/cm/%r@%h-%p -o ControlPersist=600 \
  -p 47863 elicer@central-01.tcp.tunnel.elice.io"

# 큐 상태 한 번에 (표준 라이브러리만 — 학습 CPU를 뺏지 않는다)
$SSH 'cd ~/Deep-ANC && python3 scripts/elice/queue_status.py'

# 원 학습 로그와 GPU
$SSH 'cd ~/Deep-ANC && grep "^step " runs/train_base_corrected.log | tail -n 1; \
  tail -n 3 runs/structure_search.log; \
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader'
```

`idle_seconds_total`이 이 시스템의 목적함수다. 작업 전환 시 60초를 넘으면 원인을 조사한다.
감독자 로그는 `runs/queue/supervisor_gpu{0,1}.log`, 이벤트는 `runs/queue/events.jsonl`.

### 감독자가 죽었을 때

`queue_status.py`가 `[STALE]`을 표시하면 감독자 자체가 죽은 것이다. 재기동은 안전하다 —
완료된 작업은 결과로 건너뛰고, 진입 게이트를 처음부터 다시 통과한다.

```bash
$SSH 'cd ~/Deep-ANC && bash scripts/elice/run_job_queue.sh 1'   # 또는 0
```

### 큐에 작업을 덧붙이려면

감독자는 **작업 사이마다 큐 YAML을 다시 읽는다.** 재시작 없이 `configs/elice/queue_gpu*.yaml`에
작업을 추가하면 반영된다. 이미 결과가 있는 id는 재실행되지 않는다.

## 1. 프로젝트 한 줄 요약

덕트(사각 아크릴 1.2m) 딥러닝 능동소음제어. 학습=Elice 2×A100, 추론=이 Jetson AGX Orin.
모델 HybridANCNet(tiny 1.16M=현행 실시간 / base 5.99M=TRT 목표), digital-ref 모드 우선.
**절대 목표 2가지: ① 저주파+고주파 노이즈 제거 ② 모든 소리 제거(quiet zone)** — AGENTS.md 참조.
상세: docs/00, 물리: docs/01, 구조 지도: docs/10, 목표 측정: docs/07 §0.

## 2. 현재 상태

### 완료 ✅

- 저장소 골격·문서·자동 테스트, GitHub `Roka-jsj/Deep-ANC` 공개 운영
- 19-에이전트 리뷰(결함 15건) + 5-에이전트 구조 감사(이슈 35건) 반영
- 물리 정합 학습 목표(digital-ref lead 109, P(z) resolver, trusted-band 150–600Hz NMSE)
- 원샷 부트스트랩 `scripts/elice/bootstrap_all.sh` (환경+데이터 6종+RIR+QA+테스트+2GPU 학습)
- 데이터: DNS 16,000 / speech 8,065 / music 7,997 / MIMII 3,600 / ESC-50 2,000 / DEMAND 96
  (약 154.9시간). 손상 FMA MP3 3개는 manifest에서 제외
- recorded group-aware manifest·전수 QA·독립 evaluator·파인튜닝 fail-fast **구현 완료**
- **tiny 100k 완주·로컬 회수**: 최종 val trusted **−19.47** / full −18.27dB,
  best step89,500 trusted **−19.5372dB**. 원격 SHA-256 일치
- **tiny_long 20k 완주·로컬 회수**: best step13,500 trusted **−16.6672dB**
- **Jetson 실측 (2026-08-04)**:
  - `tiny` best.pt → ONNX, ORT 등가 `8.196e-08`, **P99 1.84ms** (게이트 <3ms 통과)
  - `tiny_long` last.pt → ONNX, ORT 등가 `7.567e-09`, **P99 2.24ms** (통과)
  - 즉 tiny_long은 수용영역 2배에 P99 +0.40ms(+22%). 구조 비교의 비용축 확보
- **GPU 작업 큐 감독자 신규 구현·배포** (`src/deep_anc/ops/job_queue.py`):
  기존 프로세스 불가침 4중 진입 게이트, 실패 격리(작업 하나 실패해도 큐 계속),
  사전 등록 승자 선정, 원자적 상태 JSON, 큐 재로드
- **파인튜닝 진입점 완성**: `--state-dir` 배선, `pipeline.lock`, advisory `status.json`,
  exit code 3/4 분리, 상대 config 경로 fail-open 수정.
  검증: `--check-only` → **exit 1 + `runs/` 미생성** (설계대로 NOT READY)
- **전체 회귀 테스트 266개 통과** (세션 시작 기준선 223 → +43)

### 대기 ⬜ (다음 세션이 할 일 순서)

1. **감독자 인계 확인** (01:16~01:25 KST 전후): `queue_status.py`로 GPU1이
   `search_tiny_control`을 시작했는지 확인. `idle_seconds_total`이 60초 미만이어야 정상.
2. **승자 선정 결과 확인** (03:00 KST 전후): `select_structure_winner`의 verdict를 읽는다.
   승자가 정해지면 감독자가 `extension_template`로 **연장 작업을 자동으로 큐에 넣는다**
   (사람이 손댈 필요 없음). 다음 두 경우에는 넣지 않으며 그때만 판단이 필요하다.
   - `winner_ambiguous` — last와 best의 승자가 다름. seed 반복 결과를 먼저 본다.
   - `winner_is_control` — 대조군이 이겼다. tiny 100k 완주본이 이미 있어 연장이 무의미하다.
     이 경우 GPU1이 빈다. 다른 실험을 큐에 덧붙일지 판단한다(큐는 재로드된다).
3. **base 완주 확인** (04:45 KST 전후): GPU0 감독자가 held-out 평가를 자동 실행한다.
   완료되면 checkpoint를 회수하고 **사용자에게 인스턴스 중지/삭제 안내** (시간 과금!).
4. **최우선 하드웨어 게이트**: I²S 입력이 22:39부터 다시 간헐 과클리핑 FAIL이다.
   전원 OFF 물리 접촉 확인 후 §3-C의 무출력 probe 2개가 clip 0으로 반복 PASS하기 전에는
   **어떤 스피커 출력도 실행하지 않는다.**
5. **P/S 실측은 실행했으나 INVALID**: magnitude 형상은 반복되지만 dominant delay가
   P `37.79–64.88ms` / S `37.19–80.31ms`로 이동한다. 동기 경로·타임스탬프 방안을 먼저
   확정한 뒤 같은 gain의 80–1600Hz를 재측정한다. 성공 세션만 골라 게이트를 우회하지 않는다.
6. THD/IMD와 recorded 독립 세션 최소 80개 수집 → measured P(z) open-loop 파인튜닝 →
   recorded G4 → FxLMS와 동일 OFF→ON→OFF 세션 비교.

### 승자 연장 작업의 규약 (감독자가 자동 적용 — 참고용)

`configs/elice/queue_gpu1.yaml`의 `extension_template`에 확정돼 있다. 손으로 만들 일이
생기면 다음 세 가지를 반드시 지킨다.

- **`ckpt_dir`은 새 디렉터리** — 같은 곳에 resume 하면 pilot의 20k `best/last`를 덮어써
  구조 비교 근거가 사라진다.
- **`resume`은 `last.pt`** — `best.pt`로 되감으면 optimizer/scheduler가 후퇴해 예산을 낭비한다.
  대신 pilot의 `best.pt`를 새 ckpt 디렉터리로 **복사**해야 `trainer.py:390`의 `best_metric`
  min() 교정이 동작한다(복사하지 않으면 20k best를 넘기 전까지 `best.pt`가 아예 없다).
- **`seed`는 원 seed +100** — worker RNG는 checkpoint에 저장되지 않고 iterator 생성 시
  `seed + split_offset + worker_id*1009`로 재시드된다(`synth_dataset.py:269`). 그대로 두면
  step 20k–40k가 0–20k와 **같은 데이터를 재생**한다. (worker RNG를 checkpoint에 저장하는
  근본 수정은 별도 후속 항목이다.)
- `run_until_step`은 지정하지 않는다 → `resolve_run_until_step()`이 `total_steps` 폴백.

### 데이터/체크포인트 선택 주의

- 내장 val은 고정 16개(현재 seed에는 DEMAND 0개)라 최종 판정용이 아니다.
  `best.pt`만 맹신하지 말고 `last.pt`도 회수한다.
- 공개 데이터의 파일 단위 split은 speech 화자/책, ESC 원본, MIMII 조건, DEMAND 동시녹음
  채널 같은 상관 그룹이 split을 가로지를 수 있다.
- `secondary_surrogate` checkpoint는 표현 사전학습 전용이라 물리 성능 주장에 쓸 수 없다.
- **로컬 Jetson 오프라인 평가는 제한적이다.** `data/manifests/`와 RIR 뱅크가 없어서
  소스별 표에 `synthetic`만 남고 RIR이 즉석 32개로 대체된다. 즉 **기능 2(모든 소리)는
  로컬에서 측정할 수 없다.** 승자 선정과 소스별 평가는 반드시 Elice에서 돌린다.

### 사용자가 직접 해야 하는 것 (권한/자원 소유)

- **I²S 입력 복구**: 전원 OFF에서 공통 GND/SD/LR·pin17 접촉 확인. 이후 §3-C의 무출력
  probe 2개가 clip 0으로 반복 PASS해야 실측 재개 가능. **이것이 풀리기 전에는 실측 P/S,
  recorded 세션, 파인튜닝이 전부 막혀 있다.** Elice 사전학습 완료가 이 게이트를 대체하지 못한다.
- **Elice 인스턴스 중지/삭제**: 큐가 `drained`가 되고 회수가 끝나면 즉시. 인스턴스 켜진
  시간으로 과금되며, 중지해도 스토리지는 과금되고 삭제만 완전 중지다.
- 파인튜닝 현장 준비: AB13X·두 마이크·두 스피커 고정, 같은 출력게인의 S(cancel→ERR),
  P(noise→ERR), THD/IMD 측정. 실제 소스 독립 세션 최소 80개(1.5–2h/3–4GB),
  권장 160개(3–4h/6–8GB)와 10–15GB 여유 공간.
- 덕트 미확정값 확정 시 통보 (에러마이크 X=1.100 잠정 — 확정 시 duct.yaml + RIR 뱅크 재생성)

### 사용자가 확정한 INMP441 물리 배선 (2026-08-03)

- 두 마이크 공통: VDD 빨강→J30 pin1(3.3V), GND 검정→pin6, SCK 주황→pin12,
  WS 노랑→pin35, 두 SD 공통 갈색→pin38
- 레퍼런스 마이크 L/R 초록→pin17(3.3V, right/ch1), 에러 마이크 L/R 파랑→pin39(GND, left/ch0)
- 공식 핀표와 INMP441 L/R 규약 대조 완료. 안전 주의·근거는 docs/02 §1 참조.
- 현재 pinmux/I²S는 의도된 기존 구성이다. **sudo, Jetson-IO, pinmux, device-tree,
  오디오 데몬, 전원모드 변경은 모두 금지.**

### 현재 I²S·출력 경로·P/S 실측 상태

- APE 입력 `hw:1,1`과 AB13X 출력 `hw:2,0`은 장치로 인식되고 스트림 설정도 수락된다.
- pin17 재연결 직후 5초 probe는 ERR **−46.33dBFS**, REF **−46.64dBFS**, clip 0%로 PASS했다.
- **하지만 22:39 이후 다시 FAIL이다.** 무출력 2초 probe에서 ERR −12.84dBFS/clip 2.474%,
  REF −10.67dBFS/clip 5.029%. 10초 재검사도 FAIL. peak/raw가 정확히 ±1.0/INT32 한계까지
  도달하고 0.1초 구간별 간헐 burst가 있어 시작 과도가 아니다. 장치 점유 프로세스는 없었다.
  **스피커 출력은 전혀 시작하지 않았고 전달맵/direct FxLMS 실측은 안전 중단 상태다.**
- peak 0.005 채널 분리 실측: noise ch0와 cancel ch1 모두 ERR/REF에 도달. tone-bin 상승은
  ch0→ERR/REF +25.79/+26.15dB, ch1→ERR/REF +22.83/+28.34dB. REF 기준 ch1이 ch0보다 약
  8.3dB 강해 실제 앰프 게인/물리 매핑 차이는 별도 확인이 필요하다.
- magnitude 진단은 204–210Hz, 348–351Hz, 458–476Hz, 594–613Hz 부근 피크를 반복 검출해
  1D 예측 공진(210/350/489/629Hz)을 부분 지지한다. 이는 **공진 형상 진단**일 뿐 덕트 식별
  완료, 고정 지연, FxLMS 성능 또는 물리 좌표 확정의 근거가 아니다.
- legacy 300Hz 설정의 과거 "약 2dB 감소"는 현재 하드웨어에서 재검증한 결과가 아니다.
  2026-08-03 재현은 실제 `ON/ADAPT`까지 확인했지만 ON 로그 9개에서 중단됐고 감쇠 중앙값
  +0.11dB, 순간 최대 +0.63dB였다. **2dB 성공 증거가 아니다.**
- 19:14에 읽기 전용 감사 에이전트가 legacy 모듈을 import하면서
  `/home/capston/anc_project/__pycache__/main_realtime_anc.cpython-310.pyc` 한 개를 실수로
  생성했다. 그 경로는 이후 건드리지 않았고 **임의 삭제도 하지 말 것.**
  저장소의 안전 실행기는 `python3 -B`와 `PYTHONDONTWRITEBYTECODE=1`로 재발을 막는다.

## 3. 실행 절차

### A. GPU 작업 큐 (현행 운영 방식)

```bash
$SSH 'cd ~/Deep-ANC && python3 scripts/elice/queue_status.py'                    # 상태
$SSH 'cd ~/Deep-ANC && .venv/bin/python scripts/elice/job_queue.py plan \
  --queue configs/elice/queue_gpu1.yaml'                                          # 예정 순서(GPU 무접촉)
$SSH 'cd ~/Deep-ANC && bash scripts/elice/run_job_queue.sh 1'                     # 기동/재기동
```

> **원격 배포는 반드시 신규 경로만 scp한다.** 원격 워크트리는 HEAD가 아니라 워킹트리
> 스냅샷(dirty 37개)이라 `git pull`은 merge abort되거나, stash/checkout으로 우회하는 순간
> corrected physics 코드가 revert되어 **돌고 있는 학습이 조용히 다른 실험이 된다.**
> 또 실행 중인 bash 스크립트를 scp로 덮어쓰면 (scp가 inode를 truncate) bash가 오프셋 기준
> 지연 읽기를 하므로 **watcher가 즉사한다.** `cp -n`/`cp -rn`으로 no-clobber 설치할 것.

### B. 학습 완료 후 → Jetson 배포

```bash
mkdir -p ~/Deep_ANC/runs/pretrain_base_corrected/ckpt
scp -o ControlPath=~/.ssh/cm/%r@%h-%p -P 47863 \
  elicer@central-01.tcp.tunnel.elice.io:~/Deep-ANC/runs/pretrain_base_corrected/ckpt/best.pt \
  ~/Deep_ANC/runs/pretrain_base_corrected/ckpt/
# 회수 목록과 SHA-256 은 감독자가 runs/queue/handoff.json 에 미리 만들어 둔다
# → 이 시점에 사용자에게 "인스턴스 중지/삭제" 안내!

cd ~/Deep_ANC && .venv/bin/python scripts/train/export_onnx.py \
  --ckpt runs/pretrain_base_corrected/ckpt/best.pt --out runs/export/base.onnx
.venv/bin/python scripts/bench/measure_inference_latency.py --config configs/runtime.yaml \
  --set engine.type=ort --set engine.onnx=runs/export/base.onnx
```

`secondary_surrogate` 결과는 오프라인 표현 평가까지만 한다. 실제 스피커 실행 전에는 같은
게인의 실측 P/S로 파인튜닝하고 runtime `digital_reference_lead_samples=109`를 checkpoint
메타와 맞춘다.

### C. 하드웨어 재연결 후 (사용자 입회) — docs/02 §3 + docs/08 §5

```bash
# 1) 스피커를 전혀 열지 않는 입력 게이트. 현재는 FAIL 상태이므로 여기서 중단.
cd /home/capston/Deep_ANC
.venv/bin/python scripts/bench/check_audio_input.py
.venv/bin/python scripts/bench/check_audio_input.py --require-both

# 2) 둘 다 clip 0으로 반복 PASS한 뒤 네 경로 시간-주파수 지도
.venv/bin/python scripts/bench/measure_duct_transfer_map.py --confirm-volume-minimum

# 3) 전달맵 뒤 저음량 direct FxLMS 진단. legacy S라 결과는 diagnostic-only.
.venv/bin/python scripts/demo/evaluate_fxlms_direct.py \
  --amplitude 0.005 --control-limit 0.005 --confirm-user-present-volume-minimum
```

legacy 재현이 필요하면 원본을 기본값으로 실행하지 않는다. 종료 시 weight를 저장하므로
`--weights-output`을 이 저장소의 ignored `results/`로 우회하고 `python3 -B`를 쓴다.

### D. 파인튜닝 (게이트 통과 후)

```bash
.venv/bin/python scripts/train/run_finetune_pipeline.py \
  --config configs/train_finetune.yaml --set data.digital_primary_path_mode=measured
```

현재는 설계대로 **exit 1 (NOT READY)** 이며 `runs/` 아래에 아무것도 만들지 않는다.
exit code 표와 산출물 경로는 README §6.6 참조.

## 4. 조심할 것 (세션에서 배운 것)

- Elice 터널: 로컬 타임아웃이 나도 **원격 작업은 대부분 살아있다** — 재실행 전 반드시 상태 확인.
  원격 장기작업은 `setsid nohup … < /dev/null &` 패턴만.
- Elice는 **인스턴스 켜진 시간 과금** (SSH 연결 여부 무관). 종료(중지)해도 스토리지는 과금,
  삭제만 완전 중지.
- `tail -n 1` (구식 `tail -1`은 다중 파일에서 GNU 오류)
- 원격에서 pytest를 돌릴 때는 `nice -n 19`. 32 vCPU 중 28개를 두 학습의 DataLoader가 쓴다.
- Jetson venv 재구성은 `scripts/jetson/setup_jetson.sh` (lib preload 훅 필수), ORT 1.18.1 고정
- 입력 raw가 `-1`/0 고정이면 장치가 열려도 유효 오디오가 아니다. 무출력 preflight 실패를
  `--force`로 우회하거나 스피커 출력으로 진단하지 않는다.
- S(z)/핸드오프/목표대역은 단일 출처 원칙 (duct.yaml + config.DEFAULT_HANDOFF_SAMPLES)
- 컨테이너에서 `nvidia-smi --query-compute-apps`는 비어 보일 수 있다. GPU 유휴 판정은
  memory.used와 `/proc/*/environ`의 `CUDA_VISIBLE_DEVICES`까지 함께 봐야 한다.
- `/proc/<pid>/stat`의 starttime은 **마지막 `)` 뒤부터 세어 tail[19]** 다. comm 필드에
  공백이 들어갈 수 있어 단순 split은 틀린다. PID 재사용 판별에 이 값이 필요하다.

### 읽기 전용 참고 구현

- `~/anc_project` — legacy FxLMS(Python). block FxNLMS, S(z)는 `calibrate_s_path.py`로
  오프라인 식별(256/low에서 순수지연 1342, coherence 0.40으로 낮음). 종료 시 기본
  `control_filter_last.npy`를 CWD에 저장하므로 원본에서 기본값 실행 금지.
- `~/FxLMS/realtime_fxlms` — C++ 단일 blocking `snd_pcm_readi`→계산→`writei`, block512,
  S 2048tap, W 512tap, 내부 230+370Hz digital tone reference. FxLMS 부호는 Deep_ANC의
  `e=d+S·y`와 일치하지만 REF ch1은 실제 제어에 쓰지 않는다. APE 입력/USB 출력이 독립 clock인데
  timestamp가 없고 partial write/xrun 뒤에도 계속하므로 **절대 지연·위상 근거로 쓸 수 없다.**
  `run_experiment.sh`는 PCM 100%로 바꾸므로 볼륨 최저 규칙과 충돌해 **실행 금지**다.
  기존 `fxlms_run_log.csv`의 ANC control RMS는 거의 0.299로 ±0.3 hard-limit에 포화됐고
  표시 감쇠는 −1.80~+5.69dB로 요동해 **2dB 성공 증거가 아니다.**
- 두 폴더 모두 앞으로도 읽기 전용. python import 금지(`python3 -B`).
