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
| `trt` (FP16) | 코드 준비됨 | 미실측 — 바인딩 필요 (아래) | base 실시간의 목표 경로 |
| `fxlms` | 동작 | ~0.2ms | 베이스라인/폴백 |

게이트: block 256(5.33ms)에서 **P99 < 3.0ms**. 실행:

```bash
python scripts/bench/measure_inference_latency.py --config configs/runtime.yaml \
    --set engine.type=ort --set engine.onnx=runs/export/model.onnx
```

### TensorRT 경로 (선택 — 시스템 변경 필요)

TRT 10.3 런타임 라이브러리는 있으나 **python 바인딩·trtexec 이 미설치**다. 설치는
`sudo apt-get install python3-libnvinfer libnvinfer-bin` 이 필요한데, 이는 시스템 변경이므로
**프로젝트 정책상 자동화하지 않는다** — 사용자가 직접 판단·실행할 것. 설치 후:

```bash
bash scripts/export/build_trt.sh runs/export/model.onnx      # FP16 엔진 빌드
python scripts/bench/measure_inference_latency.py --config configs/runtime.yaml \
    --set engine.type=trt --set engine.plan=runs/export/model_fp16.plan
```

TRT 없이도 **tiny + ORT CPU 로 실시간 게이트를 통과**하므로 1차 데모는 가능하다.

## 3. 실시간 런타임 구조

```
[PortAudio 콜백, 5.33ms 주기]                [추론 스레드 (CPU 4~7 소프트 고정)]
  마이크 int32→float, DC차단 ──push──▶ in_ring(err/ref_mic/ref_digital)
  소음 ch0 = NoiseProgram×게이트          hop 대기 → engine.step(ref, err)
  ch1 = out_ring.pop (없으면 무음+카운트) ◀─push── anti-noise
        → NaN방어→리미터(0.2)→FadeGate
  베이스라인/저감dB/워치독 판단          [제어 스레드] 키보드 A/N/R/Q, 1초 통계
```

- 콜백은 **절대 대기하지 않는다** — 추론 지연 시 무음 폴백 + 데드라인 워치독.
- 파이프라인 핸드오프 = 정확히 1 hop → 학습 플랜트의 `handoff_extra_samples: 256` 와 정합.
  실효 지연 검증: `python -m deep_anc.realtime.run_realtime --config configs/runtime.yaml --calibrate`
  (차이가 +512샘플 지터 범위를 벗어나면 duct.yaml 조정 후 파인튜닝).

### 실행

```bash
# DL (ORT), 디지털 레퍼런스, 톤 300Hz 소음 — 항상 ANC OFF 시작, A 키로 ON
python -m deep_anc.realtime.run_realtime --config configs/runtime.yaml \
    --set engine.type=ort --set engine.onnx=runs/export/model.onnx
# FxLMS 회귀 기준선 (기존 시스템과 동일 동작 확인용)
python -m deep_anc.realtime.run_realtime --config configs/runtime.yaml --set controller=fxlms
# 세션 기록 (npz)
... --run-seconds 60 --record results/demo_0803
```

## 4. 안전장치 (전 모드 공통)

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
