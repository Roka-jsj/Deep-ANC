# 02. 하드웨어 구성과 점검 절차

## 1. 장치·채널 맵 (anc_project 에서 실기 검증된 구성)

```
[Jetson AGX Orin]
  입력  hw:APE,1 (ADMAIF2) · S32_LE · 48kHz · 스테레오     sounddevice 예시 idx 5
    ch0 = 에러 마이크   (INMP441, L/R핀 → GND)   덕트 X≈1.10m 벽면
    ch1 = 레퍼런스 마이크(INMP441, L/R핀 → 3.3V)  덕트 X=0.10m 벽면
  출력  AB13X USB Audio (hw:2,0) · S16_LE · 48kHz · 스테레오  idx 24
    ch0 = 소음 스피커   (좌)  덕트 X=0 폐단, 축방향
    ch1 = 상쇄 스피커   (우)  덕트 X=1.05m 상면, side-branch
  앰프  TPA3116D2 (12~24V) — 시작 전 볼륨 최소로!
```

- 장치 해석은 `deep_anc.audio_io.resolve_alsa_portaudio_device` (fxlms_core 이식)가
  `/proc/asound/cards` 의 짧은 ID(`APE`, `Audio`)로 자동 매핑한다.
- 장치 목록 확인: `python -m deep_anc.realtime.run_realtime --list-devices`
- **USB 오디오(AB13X)가 꽂혀 있어야 카드 `Audio` 가 보인다** (2026-08-02 저장소 구축 시점에는
  분리되어 있어 실기 루프 검증이 보류됨 — 연결 후 아래 3절 순서로 점검할 것).

## 2. 시스템 정책 (중요)

Jetson 의 **핀 설정(pinmux/I2S)과 RT 커널 구성은 의도된 것이므로 절대 변경하지 않는다.**
전원모드(30W), pulseaudio/pipewire, RT priority limit 등 시스템 상태도 건드리지 않는다.
이 저장소의 모든 도구는 유저 공간(venv)에서만 동작하도록 만들어졌다.
(성능 튜닝 여지가 있는 항목들은 docs/06 §5 에 "참고"로만 기록)

## 3. 하드웨어 점검 순서 (USB 오디오 연결 후)

```bash
source .venv/bin/activate
# 1) 장치 인식
python -m deep_anc.realtime.run_realtime --list-devices     # APE(hw:1,1), Audio(hw:2,0) 확인
# 2) 마이크 자가진단 (스피커 무음) — 레퍼런스 마이크 ch1 무신호 이력 확인 필수!
python scripts/data/record_duct.py --program silence --seconds 10
#    → "ch1(ref) ... dBFS" 가 -80dBFS 보다 크면 정상. 무신호면 INMP441 배선(L/R=3.3V) 점검
# 3) 무음 전체 루프 (스피커 소리 없음, 3-스레드 검증)
python -m deep_anc.realtime.run_realtime --config configs/runtime.yaml \
    --set noise.type=silence --set engine.type=ort --run-seconds 10
# 4) 실효 지연 측정 (처프 재생 — 볼륨 낮게!)
python -m deep_anc.realtime.run_realtime --config configs/runtime.yaml --calibrate
# 5) I/O 지연 스윕 (선택)
python scripts/bench/measure_io_latency.py
```

### 레퍼런스 마이크(ch1) 이력

2026-08-01 진단(`anc_project/diagnostics/mic_stats.txt`)에서 ch1 이 **사실상 무신호**
(RMS −189dBFS, raw −1/0 반복)로 기록된 바 있다. 이것이 기존 FxLMS 와 본 런타임의
기본 reference 가 `digital` 인 이유다. acoustic-ref(2단계)로 가기 전 반드시 수리·확인할 것.
`record_duct.py` 는 시작 시 자동으로 이를 점검하고 무신호면 중단한다(`--force` 로 강행).

## 4. 2차경로 S(z) 보정

### 현재 자산 (assets/measured/)

| 파일 | delay | fit | coherence | 대역 | 비고 |
|---|---|---|---|---|---|
| `secondary_path_4s.npz` | 1342 (27.96ms) | 2.14dB | 0.40 | 150–600Hz | **채택본** (block 256/low 측정) |
| `secondary_path_legacy_512high.npz` | 2613 (54.4ms) | 1.09dB | 0.27 | 150–600Hz | 구버전 기록용 (block 512/high) |

주의: 기존 anc_project 는 512/high 로 측정된 모델을 256/low 런타임에 쓰고 있었다
(지연 26ms 어긋남 — appendix 참조). 본 저장소는 **256/low 측정본(4s)** 을 채택했다.

### 광대역 재보정 (풀밴드 학습의 선행 게이트)

```bash
# S(z) 재보정: 상쇄 스피커(ch1) → 에러 마이크, 80–8000Hz ESS 스윕
python scripts/data/calibrate_wideband.py --output-channel cancel \
    --out assets/measured/secondary_path_wb.npz
# 반복 일관성 ≥0.9 확인 후 duct.yaml secondary_path.npz 교체 → 파인튜닝
# digital-ref 1차경로 지연 실측: 소음 스피커(ch0) → 에러 마이크
python scripts/data/calibrate_wideband.py --output-channel noise
#    → 출력된 delay 를 duct.yaml digital_reference.d_noise_delay_samples 에 기입
```

## 5. 안전 수칙

1. 모든 실행은 **ANC OFF 로 시작**한다 (A 키로 수동 ON).
2. TPA3116D2 볼륨은 최소에서 시작해 점진적으로 올린다.
3. 런타임 안전장치: 출력 리미터(0.2) / 클립 스트릭 자동 mute / 발산 워치독(+6dB·0.5s)
   / 추론 데드라인 워치독 — 이상 시 자동으로 상쇄 채널이 꺼진다.
4. 스피커에 소리를 내는 스크립트(`record_duct.py`, `calibrate_wideband.py`,
   `measure_io_latency.py`, `evaluate_session.py`, `run_realtime.py`)는 반드시 사람이
   현장에 있을 때 실행한다.
