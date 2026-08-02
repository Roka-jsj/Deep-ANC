# 03. 데이터 파이프라인

## 1. 신호 모델 (Deep ANC 방식)

학습 샘플 하나는 (모델 입력 x, 1차경로 소음 d) 쌍이다. 타깃 파형은 없다 —
손실이 미분가능 플랜트로 에러를 직접 계산한다 (docs/04 §5):

```
x[0] = x_ref   (레퍼런스: digital→소스 원신호, acoustic→P_ref·n)
x[1] = err_in  (에러 피드백 근사: d 를 512~1024샘플 랜덤 지연 — 캡처+블록 지연 모사)
d    = 1차경로 소음 (digital→P_err·n 을 D_noise 만큼 지연, acoustic→P_err·n)
손실:  e = d + S(G_nl(y)),  y = model(x)
```

지연 물리(모드별 D_noise, S 지연+핸드오프)는 docs/01 §3 이 근거다.

## 2. 소음원 구성 (`data_sim.yaml source_mix_ratio`)

| 소스 | 비율 | 이유 |
|---|---|---|
| 합성원 (`synthetic_signals.py`) | 45% | 물리적으로 상쇄 가능한 주기성/준정상 잡음이 1차 목표. 톤+고조파, AM/FM 기계음, 협대역, 처프, 멀티톤. 덕트 공진(70~629Hz) 근방 f0 를 40% 확률로 가중 추첨 |
| DNS-Challenge noise_fullband | 45% | 48kHz 네이티브 실환경 소음 대량 확보 |
| ESC-50 | ≤10% | 환경음 다양성 — 비정상(non-stationary) 신호라 상쇄 불가 성분이 많아 **강건화용 소량**만 (설계 검증 L1) |

다운로드(`scripts/data/download_noise.sh`)는 Elice 에서 실행 권장 (Jetson 디스크 38GB 여유).
manifest 없이도 학습은 합성원 100% 로 자동 폴백된다 (스모크 테스트용).

## 3. 덕트 음향 시뮬 (RIR 뱅크)

- `dsp/duct_sim.py` — 1D 영상법, closed(폐단)–open(개방단) 경계. `duct.yaml` 기하 사용.
  검증: 이론 공진 70/210/350Hz 재현 (tests/test_duct_sim.py).
- 한계: 평면파 모델 — 컷오프(1633Hz) 이상 고차 모드는 미모델링(저역통과 근사만).
- **도메인 랜덤화 뱅크**: 반사계수/감쇠/위치 ±1cm/저역통과 컷오프를 랜덤화한 변형 300개.
  시뮬-실측 갭을 흡수한다. RIR 변형도 train/val/test 로 분할(누수 방지).

```bash
python scripts/data/build_rir_bank.py --n 300     # duct.yaml 변경 시 재실행 필수
```

## 4. 증강

| 증강 | 범위 | 목적 |
|---|---|---|
| 레벨 | −35 ~ −10 dBFS | 마이크 레벨 다양성 |
| 스피커 비선형 | SEF η∈{0.1,0.5,1,10} + drive g∈[1,4] + 하드클립 5% | Zhang&Wang(2021) 프로토콜 — DL 의 비선형 보상 학습 |
| 플랜트 섭동 | 지연 지터 [0,+512], 게인 ±3dB, 틸트 ±2dB/oct, 올패스 위상 | S(z) coherence 0.40 의 낮은 신뢰도 대응 |
| 마이크 자기잡음 | SNR 5~30dB | INMP441 잡음 바닥 모사 |
| 채널 dropout | ref 15% / err 15% (동시 금지) | ref-only / err-only 운용 폴백 학습 |
| 피드백 지연 | err_in 에 512~1024샘플 랜덤 | 실배치 캡처+블록 지연 모사 (설계 H3) |

## 5. 실측 수집 → 파인튜닝

```bash
# 세션 수집 (ANC OFF, ch1 상쇄 스피커 무음 유지)
python scripts/data/record_duct.py --program tone --frequency 300 --seconds 60
python scripts/data/record_duct.py --program band --seconds 120
python scripts/data/record_duct.py --program nonlinear --seconds 60
python scripts/data/record_duct.py --program silence --seconds 30    # 암소음
# manifest 생성 (세션 단위 8:1:1 분할)
python scripts/data/make_recorded_manifest.py
# 파인튜닝 (실측:합성 = 7:3 혼합)
python scripts/train/train.py --config configs/train_finetune.yaml
```

세션 구조: `data/recorded/<시각_프로그램>/{mics.wav(2ch PCM_32), source.wav, session.json}`.
ANC OFF 녹음이므로 **에러 마이크 신호가 곧 d(t)** 다. digital-ref 파인튜닝은 source.wav 를,
acoustic-ref 는 ref 마이크 채널을 x_ref 로 쓴다 (`recorded_dataset.py`).

## 6. manifest 스키마와 분할 규칙

JSONL 한 줄 = 파일/세션 하나:

```json
{"path": "...", "duration_s": 12.3, "sample_rate": 48000, "tag": "dns_fullband", "split": "train"}
```

누수 방지 3원칙 (tests/test_dataset.py 가 검사):
1. 노이즈 풀은 **원본 파일 단위** 90/5/5 분할 (세그먼트 단위 금지)
2. RIR 뱅크 변형도 **변형 단위** 분할
3. 실측 데이터는 **세션 단위** 분할

## 7. 데이터셋 적합성 — 학습 × 실시간 추론 정합 분석

배포된 모델이 보는 신호는 "덕트 마이크가 들은 소리"다. 학습 분포가 이것과 정합해야 한다.

### 소스별 적합성 매트릭스

| 소스 | 원 SR | 학습 적합성 | 추론(배포) 정합성 | 비고 |
|---|---|---|---|---|
| DNS noise_fullband | **48k 네이티브** | ◎ 실환경 소음 대량 | ◎ 팬/기계/환경 소음 = 덕트 실전 분포 | 주력 |
| DNS clean_fullband(speech) | **48k 네이티브** | ◎ 대화 제거 목표의 핵심 | ◎ digital-ref 데모(음성 재생→상쇄)와 직결 | ⚠ acoustic-ref 에선 비주기라 불가(물리) |
| ESC-50 | 44.1k→48k 리샘플 | ○ 다양성 (≤10% 제한) | △ 비정상 이벤트음 — 강건화용 | 22.05k 이상 무성분(문제 없음 — 나이퀴스트 밖) |
| 합성원 | 48k 생성 | ◎ 주기성/공진 정조준 | ◎ 덕트 공진(70~629Hz) 가중 생성 | 무한 |
| music (예정) | 44.1/48k | ○ 음악 제거 목표 | ○ digital-ref 데모용 | mp3 는 soundfile(libsndfile≥1.1)로 디코딩 |

### 파이프라인의 정합 장치 (코드 근거)

| 배포 현실 | 학습 쪽 대응 |
|---|---|
| INMP441 레벨 편차 | 소스 RMS 정규화 후 레벨 랜덤화 −35~−10 dBFS (`synth_dataset`) |
| 마이크 자기잡음 | SNR 5~30dB 가우시안 부가 |
| DC/저역 험 (런타임은 DCBlocker) | dc_hum 증강 20% + 손실 W(f)<40Hz ×0.1 |
| 44.1k 소스 | 로더에서 resample_poly 48k (`NoisePool`) |
| 5초 미만 클립 | 타일링 반복 (`NoisePool.sample_segment`) |
| 스피커/앰프 비선형 | SEF/drive/하드클립 증강 (플랜트 통과 전) |
| 덕트 음향 | RIR 뱅크 300변형 (도메인 랜덤화) + 측정 S(z) |

### 자동 QA 게이트

다운로드 직후 `python scripts/data/validate_noise_pool.py` (부트스트랩 [4/6]에 포함)가
태그별 표본 150개를 실검사한다: 샘플레이트 분포/읽기 실패율/클리핑/무음 비율/
**덕트 제어대역(<1.6kHz) 에너지 비율**. 결과는 `data/manifests/dataset_qa.md` 리포트로 남고,
치명(태그 전체 읽기 불가)이면 학습 시작 전에 중단된다.

## 8. 저장 정책

온더플라이 합성이므로 학습쌍을 디스크에 굳히지 않는다 — 원본 노이즈(30~50GB)만 저장.
Elice 128GiB 스토리지에 여유 있게 들어가며, 업로드가 필요할 땐
`python scripts/data/pack_transfer.py` 로 2GB tar 샤드를 만든다.
