# 06. Jetson AGX Orin 배포

## 1. 환경 (구축 완료 상태 기록)

| 항목 | 값 |
|---|---|
| 플랫폼 | JetPack 6 / L4T R36.4.4, CUDA 12.6, cuDNN 9.3, RT 커널(의도된 구성 — 변경 금지) |
| Python | 3.10.12 + venv `.venv` (`--without-pip --system-site-packages` + get-pip 부트스트랩) |
| torch | **2.5.0a0+872d972e41.nv24.08** (NVIDIA 공식 JP6.1 wheel) — CUDA(Orin) 동작 확인 |
| onnxruntime | **1.18.1 고정** — 1.19+ 는 Tegra 에서 cpuinfo 크래시 |
| numpy/scipy | 1.26.4 / 1.15.3 (venv 내부 — 시스템 1.21/1.8 과 격리) |

### torch wheel 의 누락 라이브러리 해결 (유저 공간)

NVIDIA JP6 wheel 은 `libnvToolsExt.so.1`, `libcupti.so.12`, `libcusparseLt.so.0` 을
요구하지만 이 시스템의 CUDA 최소 설치에는 없다. **apt(시스템 변경) 대신** pip 패키지
(`nvidia-nvtx-cu12==12.6.77`, `nvidia-cuda-cupti-cu12`, `nvidia-cusparselt-cu12`)로 받고,
`.venv/.../site-packages/_deep_anc_libs.pth` → `_deep_anc_libpaths.py` 가 인터프리터 시작 시
ctypes 로 preload 한다. venv 를 새로 만들면 이 두 파일을 함께 복사할 것.

## 2. 추론 엔진 3종 + FxLMS (실측치 포함)

| 엔진 | 상태 | hop 256 스텝 지연 (Jetson 실측) | 용도 |
|---|---|---|---|
| `torch` | 동작 | tiny ~10ms (P99 11.6ms) | 개발/디버깅 — 실시간 불가 |
| `ort` (CPU) | 동작 | **tiny P99 1.50ms ✓ / base P99 6.8ms ✗** | **현행 배포 기본** (tiny) |
| `trt` (FP16) | 코드 준비됨, 현재 환경 실행 도구 없음 | 미실측 | 별도 사전 구성 환경의 base 목표 경로 |
| `fxlms` | 동작 | ~0.2ms | 베이스라인/폴백 |

게이트: block 256(5.33ms)에서 **P99 < 3.0ms**. 실행:

```bash
.venv/bin/python scripts/bench/measure_inference_latency.py --config configs/runtime.yaml \
    --set engine.type=ort --set engine.onnx=runs/export/model.onnx
```

표의 FxLMS `동작`은 알고리즘/실시간 처리 경로를 뜻한다. 2026-08-03 pin17 복구 뒤 ERR/REF
I²S 입력은 −46dBFS대, clip 0%로 두 채널 probe를 통과했다. 실제 덕트 상쇄는 출력 직전
probe 재통과와 사용자 입회·앰프 볼륨 최저를 모두 만족할 때만 실행한다.

### TensorRT 경로 (현재 Jetson에서는 사용 불가)

TRT 10.3 런타임 라이브러리는 있으나 **python 바인딩·trtexec 이 미설치**다. 이 프로젝트는
apt/sudo를 포함한 Jetson 시스템 변경을 금지하므로 현재 장비에 설치하지 않는다. 따라서 현행
배포는 tiny+ORT를 사용한다. `trtexec`이 사전에 제공된 별도 환경에서만 다음 경로를 검증한다.

```bash
bash scripts/export/build_trt.sh runs/export/model.onnx      # FP16 엔진 빌드
.venv/bin/python scripts/bench/measure_inference_latency.py --config configs/runtime.yaml \
    --set engine.type=trt --set engine.plan=runs/export/model_fp16.plan
```

TRT 없이도 **tiny + ORT CPU 로 실시간 게이트를 통과**하므로 1차 데모는 가능하다.

### 체크포인트와 배포 자격

현재 corrected Stage-1은 `digital_primary_path_mode: secondary_surrogate`로
`P(z)=S(z)`를 사용한 **표현 사전학습**이다. 이 체크포인트는 모델·스트리밍·
ONNX·지연 파이프라인을 검증하는 데는 쓸 수 있지만, 실제 noise speaker→ERR
1차경로 `P(z)`를 대신하지 못한다. 따라서 아래 세 게이트 전에는 덕트
감쇠 수치를 물리 성능으로 인용하지 않는다.

1. 동일 앰프·볼륨·입출력 설정에서 `P(z)`(noise→ERR)와 `S(z)`(cancel→ERR)를 실측
2. `digital_primary_path_mode: measured`로 실측 녹음 파인튜닝
3. 학습에 쓰지 않은 recorded test와 사용자 입회 실기 평가 통과

## 3. 실시간 런타임 구조

```
[PortAudio 콜백, 5.33ms 주기]                [추론 스레드 (CPU 4~7 소프트 고정)]
  마이크 int32→float, DC차단 ──push──▶ in_ring(err/ref_mic/ref_digital)
  NoiseProgram×게이트 → lead FIFO → 소음 ch0
  생성 신호 즉시 ref_digital ──push──▶   hop 대기 → engine.step(ref, err)
  ch1 = out_ring.pop (없으면 무음+카운트) ◀─push── anti-noise
        → NaN방어→리미터(0.2)→FadeGate
  베이스라인/저감dB/워치독 판단          [제어 스레드] 키보드 A/N/R/Q, 1초 통계
```

- 콜백은 **절대 대기하지 않는다** — 추론 지연 시 무음 폴백 + 데드라인 워치독.
- 파이프라인 핸드오프 = 정확히 1 hop → 학습 플랜트의 `handoff_extra_samples: 256` 와 정합.
  실효 지연 검증: `.venv/bin/python -m deep_anc.realtime.run_realtime --config configs/runtime.yaml --calibrate`
  (실측과 다르면 원인을 확인하고 P/S 지연·lead를 함께 재산출한 뒤 파인튜닝).
- `digital_reference_lead_samples: 109`는 자기생성 소음의 실제 재생을 109샘플
  (2.27ms) FIFO로 늦춰 모델에 확정된 선행 신호를 제공한다. 소음 게이트도
  같은 FIFO를 통과하므로 ON/OFF 전환 중에도 `ref[t]=source[t+109]` 정렬을
  유지한다. 현재 corrected Stage-1 학습 설정은 109를 사용한다.
- 배포 템플릿 `configs/runtime.yaml`은 legacy artifact의 자동 오실행을 피하려고 0을
  유지한다. **artifact의 학습 lead와 런타임 lead를 반드시 같게** 맞춘다.
  `reference: mic`은 0 이외의 값을 거부한다.
- 체크포인트에는 resolved 학습 설정·`physics_status`·lead가 남고,
  `export_onnx.py`는 lead를 동반 JSON으로 복사한다. Torch/ORT/TRT 엔진은
  런타임 값과 다르면 **오디오 스트림을 열기 전에 fail-fast**한다. lead
  키가 없는 legacy checkpoint/ONNX JSON은 호환 규약상 0이다.

### 실행

```bash
# 모든 실기 런타임보다 먼저 수행: 스피커를 열지 않는 ERR 입력 게이트
.venv/bin/python scripts/bench/check_audio_input.py
# acoustic-reference/recorded 수집은 두 채널 모두 필요
.venv/bin/python scripts/bench/check_audio_input.py --require-both
# lead=0 artifact의 DL(ORT) — 항상 ANC OFF 시작, A 키로 ON
.venv/bin/python -m deep_anc.realtime.run_realtime --config configs/runtime.yaml \
    --set engine.type=ort --set engine.onnx=runs/export/model.onnx
# +109 artifact 정렬 스모크; surrogate Stage-1은 물리 데모에 사용 금지
.venv/bin/python -m deep_anc.realtime.run_realtime --config configs/runtime.yaml \
    --set digital_reference_lead_samples=109 \
    --set engine.type=ort --set engine.onnx=runs/export/model_lead109.onnx
# FxLMS 회귀 기준선 (기존 시스템과 동일 동작 확인용)
.venv/bin/python -m deep_anc.realtime.run_realtime --config configs/runtime.yaml --set controller=fxlms
# 세션 기록 (npz)
... --run-seconds 60 --record results/demo_0803
```

pin17 복구 뒤 ERR/REF probe는 모두 PASS했다. 매번 `run_realtime`, `evaluate_session`, 경로
보정 또는 legacy FxLMS 직전에 다시 검사하고, 실패하면 진행하지 않는다. pinmux/I²S/RT 커널/
전원모드/오디오 데몬을 바꾸거나 sudo를 사용하지 않는다.

### legacy 300Hz 약 2dB 기준선과 현행 정량 평가

사용자가 과거 다음 설정에서 약 2dB 감소를 확인했다고 전달했다. 이는
`secondary_path.npz` + block512/high + 70ms digital preview를 쓴 **과거 baseline**이며,
현재 복구된 입력에서 아직 재검증한 값이나 현행 256/low 경로의 결과가 아니다.

```bash
cd /home/capston/anc_project
python3 -B main_realtime_anc.py \
  --noise-type tone --frequency 300 --noise-amplitude 0.05 \
  --noise-delay-ms 70 --mu 0.001 --control-limit 0.10 \
  --block-size 512 --latency high
```

위 명령은 역사적 파라미터 기록용이다. 원본 스크립트는 정상 종료 시 기본
`control_filter_last.npy`를 쓰므로 **`~/anc_project`에서 그대로 실행하지 않는다.** 입력 probe가
PASS한 뒤 진단 재현이 필요하면 마지막에 아래 옵션을 추가해 쓰기 경로를 저장소 내부의 ignored
`results/`로 돌린다.

```bash
--weights-output /home/capston/Deep_ANC/results/legacy_fxlms/control_filter_last.npy
```

위 `-B`도 반드시 유지해 읽기 전용 원본에 `__pycache__`를 생성·갱신하지 않는다.

최종 FxLMS 성능은 legacy UI의 순간 reduction이 아니라 현행 경로에서 자동으로 기록한다.

```bash
# 사용자 입회·앰프 볼륨 최저, 입력 probe PASS 후에만
.venv/bin/python scripts/demo/evaluate_session.py \
  --controllers fxlms --scenarios tone300
```

이 프로토콜은 ANC OFF 10초 → ON 30초 → OFF 5초를 같은 세션에 저장한다.

## 4. 안전장치 (전 모드 공통)

> 실시간 실행·지연 보정·실기 평가는 모두 사용자 입회와 앰프 볼륨
> 최저 상태에서만 한다. 런타임은 항상 ANC OFF로 시작한다.

| # | 장치 | 동작 |
|---|---|---|
| 1 | 시작 OFF | `start_on: false` — A 키로만 ON |
| 2 | FadeGate | ANC 20ms / 소음 100ms 페이드 |
| 3 | 출력 방어 | NaN→0, 하드 리미터 ±0.2 |
| 4 | 클립 스트릭 | 연속 20블록 리미터 히트 → 자동 OFF |
| 5 | 발산 워치독 | 에러파워 > 베이스라인×4 가 0.5s 지속 → 자동 OFF |
| 6 | 데드라인 워치독 | out_ring 3 hop 연속 공백 → 자동 OFF (→ FxLMS/tiny 전환 검토) |
| 7 | 입력 클리핑 통계 | 리포트 표기 |
| 8 | 종료 시퀀스 | 양 채널 페이드 후 스트림 종료 |

## 5. 시스템 튜닝 — 정책상 실행하지 않음 (참고 기록)

아래 항목들은 지연/지터 개선 여지가 있으나 **모두 시스템 변경**이라 이 프로젝트에서는
수행하지 않는다 (핀 설정·RT 커널 구성은 의도된 상태 — 절대 변경 금지 정책):
전원모드 MAXN, jetson_clocks, `/etc/security/limits.d` rtprio, pulseaudio/pipewire 중지,
USB autosuspend. 현재 30W 모드·rtprio 0 상태의 실측치가 위 표의 수치다.
유일하게 코드가 하는 것은 추론 스레드의 **소프트 CPU 고정**(`sched_setaffinity`, 권한 불필요)뿐이다.
