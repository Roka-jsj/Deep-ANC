# 04. 모델 아키텍처 — HybridANCNet

GCRN / Transformer / WaveNet / Conv-TasNet 네 구조에서 실시간 ANC 에 필요한 요소만
취사선택해 결합한 **시간영역 인과 회귀 모델**. STFT 를 쓰지 않는다
(창 지연 0, TensorRT 의 DFT 미지원 원천 회피).

## 1. 무엇을 취하고 무엇을 버렸나

| 원천 | 취함 | 버림 | 근거 |
|---|---|---|---|
| Conv-TasNet | 학습형 1D conv 인코더/디코더, TCN 골격 | 마스크 곱 출력, global LayerNorm | ANC 출력은 입력의 마스킹이 아니라 위상반전+예측된 **새 파형** → 직접 회귀. gLN 은 비인과 |
| WaveNet | dilated **causal** depthwise conv, residual, gated activation | 샘플단위 자기회귀 | 샘플 AR 은 48kHz 실시간 불가 — dilation 을 프레임(hop 128) 단위로 |
| GCRN | GLU 게이팅, **GLSTM**(그룹 LSTM 병목) | STFT 복소 스펙트럼 매핑, 주파수축 conv2d | LSTM 의 무한 기억이 주기 잡음 예측의 핵심. STFT 는 지연·TRT 문제 |
| Transformer | **windowed causal MHSA 1층** (KV 캐시 64프레임=170ms), 상대위치 bias | 전역 attention, 대형 FFN | 회전기계 등 반복 패턴 재조회 용도로만 — FFN 은 TCN 이 대체 |

## 2. 구조 (base 기준)

```
입력 [B,2,T] (ch0=ref, ch1=err피드백, T=hop 배수)
 ├ ÷ io_scale (0.02)
 ├ 좌측 256샘플 패딩 → Encoder Conv1d(2→512, k=384, s=128) → GLU → 256ch   ← 룩백 8ms, 룩어헤드 0
 ├ ChannelLN + 1×1
 ├ TCN 반복 ×3 { dilation 1,2,4,8,16 }        각 블록: 1×1(256→512)+PReLU+LN
 │    ↑ 반복2 뒤: GLSTM(그룹2, 그룹당 hid256)     → dwConv k3 ×2(주경로·게이트 σ)
 │    ↑ 반복3 뒤: causal MHSA(head4×64, 윈도64)   → 1×1(512→256) + residual
 ├ Head 1×1(256→512) + PReLU
 ├ Decoder ConvTranspose1d(512→1, k=384, s=128) → 앞 T 샘플 (인과 OLA)
 └ × io_scale → 소프트 리미터 0.2·tanh(y/0.2)                     → 출력 [B,1,T]
```

| 변형 | 파라미터 | 연산 | 수용영역 | 용도 |
|---|---|---|---|---|
| tiny | 1.16M | 0.43 GMAC/s | 0.16s + LSTM | **현행 실시간 기본** (ORT CPU P99 1.5ms 실측) |
| base | 5.99M | 2.25 GMAC/s | 0.50s + LSTM + MHSA 170ms | TRT FP16 배포 목표 (ORT CPU 6.8ms) |
| large | (v2 옵션) | — | — | A100 teacher/distillation — 1차 릴리스 제외 |

파라미터 실측: tests/test_model_shapes.py (tiny 0.9~1.5M, base 5~7M 게이트).

## 3. 인과성과 지연

- 인코더는 과거 384샘플만 참조(좌측 패딩), 디코더 OLA 는 과거 프레임의 꼬리만 합산
  → **알고리즘 지연 0**. 테스트: 미래 입력을 바꿔도 현재 출력 불변(비트 단위 동일).
- 필요한 예측(digital-ref −2.3ms / acoustic-ref −30ms)은 구조가 아니라
  **손실 정렬로 학습**된다: 플랜트가 y 에 1598샘플 지연을 부과하므로, 손실을 낮추려면
  모델이 자동으로 그만큼 미래를 예측해야 한다 (예측 헤드 불필요).

## 4. 스트리밍 상태 (전부 명시적 텐서, 정적 shape)

| 상태 | shape (base) | 내용 |
|---|---|---|
| `st_enc` | [1,2,256] | 인코더 입력 히스토리 (win−hop) |
| `st_i_tcn` ×15 | [1,512,2d] | dilated dwConv 좌측 히스토리 |
| `st_i_lstm_h/c` | [1,512] ×2 | GLSTM 은닉/셀 (2그룹×256 concat) |
| `st_i_attn_k/v` | [1,4,64,64] ×2 | MHSA KV 링버퍼 (concat+slice) |
| `st_i_attn_m` | [1,1,1,64] | KV 슬롯 유효성 마스크 (빈 슬롯 −1e4) — 워밍업 구간도 오프라인과 등가 |
| `st_dec` | [1,1,256] | 디코더 OLA 꼬리 |

스트리밍=오프라인 등가성: max err ≤ 3e-8 (tests). GLSTM 은 학습 시 cuDNN nn.LSTM,
스트리밍/export 시 동일 가중치의 수동 셀 — 등가성 테스트 포함 (설계 H1).

## 5. 학습 손실 (`losses/anc_loss.py`)

```
y → G_nl(랜덤 SEF/drive)  → S(z)(1342+256 지연 + 2048탭 FIR, 섭동 증강) → e = d + S(G_nl(y))
L = NMSE(dB) + 1.0·MR-STFT{256,512,1024,2048}×W(f) + 1e-3·L_pow + 1.0·L_clip(마진 0.18)
```

- NMSE(dB) = 평가지표를 직접 최소화. MR-STFT 는 스펙트럼 균형 담당.
- **W(f) 커리큘럼 A**: 80–1000Hz ×3, 1633Hz(컷오프) 이상 ×0.25, 40Hz 미만 ×0.1.
  풀밴드 커리큘럼 B 는 **광대역 S(z) 재보정 통과 후에만** (설계 C3 게이트).
- 극성 규약: 측정 FIR 에 극성이 포함 — **추가 부호 반전 금지** (e = d + S·y).
- 손실은 FP32 고정 (bf16 은 FFT 미지원).

## 6. ONNX Export 규약 (`scripts/train/export_onnx.py`)

- opset 17, 배치 1, **모든 shape 정적**, 상태 전부 명시 입출력.
- 그래프 입력 x=[1,2,256] — 모델 hop 128 의 2프레임을 **그래프 내부 정적 언롤**
  (LSTM 2스텝 수동 셀, MHSA 2쿼리, KV concat+slice).
- 사용 op: Conv/ConvTranspose/MatMul/Sigmoid/Tanh/Softmax/PReLU/Split/Concat/Slice/
  Transpose/Reshape/LayerNormalization. 금지: LSTM/GRU op, DFT, If/Loop/Scan, 복소 dtype.
- export 후 ORT(CPU) 스트리밍 등가성 자동 검증 (실측 2.4e-8, 허용 1e-4).
