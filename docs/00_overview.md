# 00. 프로젝트 개요

## 무엇을 만드는가

덕트(사각 아크릴, 1.2m) 안의 소음을 **딥러닝 모델이 실시간으로 상쇄**하는 시스템.
소음 스피커(NS)가 좌측 폐단에서 소음을 방사하면, 레퍼런스 신호를 입력받은 모델이
상쇄 신호 y(n)을 만들어 상쇄 스피커(CS)로 출력하고, 에러 마이크(ERR)가 잔여 소음을 감시한다.

```
            X=0        X=0.1m                X=1.05m   X≈1.1m   X=1.2m
        ┌────┃──────────┃─────────────────────┃─────────┃─────────┐
  [NS]──┨    │        [REF]                 [CS]      [ERR]       ┃ → 개방단
        └────┃──────────┃─────────────────────┃─────────┃─────────┘
   소음 스피커      레퍼런스 마이크        상쇄 스피커   에러 마이크

  Jetson AGX Orin: 마이크 2ch 입력(hw:APE,1) · 스피커 2ch 출력(AB13X USB)
```

기존 FxLMS(적응 필터) 시스템과 동일 하드웨어를 쓰되, 선형 적응 필터로는 불가능한
**비선형 왜곡 보상**과 **복잡한 잡음 패턴의 학습**을 딥러닝으로 달성한다.

## 시스템 전체 그림

```
[학습 — Elice Cloud 2×A100]
  공개 노이즈셋(DNS/ESC-50) + 합성원 ─┐
  덕트 음향 시뮬(RIR 뱅크, duct.yaml) ─┼→ 온더플라이 합성 → HybridANCNet
  측정 2차경로 S(z) (npz) ────────────┘      ↓ e = d + S(G_nl(y)) 손실
                                        best.pt 체크포인트
[배포 — Jetson AGX Orin]
  best.pt → ONNX(정적 스트리밍 그래프) → [ORT CPU | TensorRT FP16]
  3-스레드 런타임: 콜백(5.33ms) ↔ 링버퍼 ↔ 추론 스레드, 안전장치 8종
[검증]
  덕트 실측 녹음 → 파인튜닝 → OFF/ON/OFF 자동 평가 → 밴드별 감쇠 리포트
```

## 3단계 로드맵 (물리 제약 기반 — docs/01 참조)

| 단계 | 모드 | 목표 | 근거 |
|---|---|---|---|
| **1단계 (현재)** | digital-ref | 협대역(공진 210/350Hz) → 광대역 80–800Hz 감쇠. FxLMS 대비 비선형 시나리오 우위 입증 | 상쇄 경로가 소음 경로보다 ~3ms 선행 → 광대역도 인과적 가능 |
| **2단계** | acoustic-ref | 외부(마이크 수음) 소음 중 주기성/준정상 잡음 상쇄 — 진짜 quiet zone 의 시작 | P≈30ms 예측 필요 → LSTM/MHSA 의 주기 학습으로 대응 |
| **3단계** | acoustic-ref 광대역 | I/O 지연 단축(I2S DAC 직결, 96kHz 등 하드웨어 개선) 후 준광대역 확장, 8kHz+ 시도 | 지연이 인과성 예산(2.77ms) 안으로 들어와야 함 |

각 단계는 같은 코드베이스에서 config 변경(`data_sim.yaml reference_mode`,
`duct.yaml` 지연 값)과 파인튜닝만으로 전환된다.

## 저장소 지도

```
Deep_ANC/
├─ configs/          # duct(덕트 실측), model, train, runtime, eval — 모든 파라미터의 단일 출처
├─ src/deep_anc/
│  ├─ dsp/           # 미분가능 S(z), 덕트 시뮬(영상법), 비선형, 필터
│  ├─ models/        # HybridANCNet (TCN/GLSTM/MHSA), 스트리밍/Export 래퍼
│  ├─ losses/        # ANCLoss (NMSE + MR-STFT×W(f))
│  ├─ data/          # 온더플라이 합성, 노이즈풀, 실측 데이터셋, manifest
│  ├─ train/         # Trainer(open/closed-loop, DDP), 체크포인트, 재현성
│  ├─ eval/          # 지표, 플롯, FxLMS 베이스라인
│  ├─ realtime/      # 3-스레드 런타임, 엔진 4종, 링버퍼, 안전장치
│  └─ baselines/     # anc_project fxlms_core.py 사본 (출처 명기)
├─ scripts/          # data / train / eval / elice / jetson / export / bench / demo
├─ tests/            # 30+ 검증 (인과성, 등가성, 물리 재현, 누수)
├─ assets/measured/  # 측정 2차경로 npz (저장소에 포함)
└─ docs/             # 이 문서들
```

## 검증 상태 (2026-08-02, Jetson 실측)

- pytest 30+ 통과: 인과성(미래 무의존), 스트리밍=오프라인 등가(실측 ~3e-8, 테스트 허용 1e-5), GLSTM 이중 경로 등가,
  덕트 시뮬 공진 70/210/350Hz 재현, 데이터 분할 무누수, S(z) torch=scipy 등가
- 학습 스모크: open/closed-loop 각각 Jetson GPU에서 정상 (bf16 AMP)
- ONNX export → ORT 등가성 max err 2.4e-8
- 추론 지연: tiny+ORT CPU P99 **1.50ms** (블록 예산 5.33ms 통과), base+ORT 6.8ms
