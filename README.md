# Deep_ANC — 덕트 딥러닝 능동소음제어

사각 아크릴 덕트 실험 리그에서 **딥러닝 기반 능동소음제어(Deep ANC)** 를 구현하는 캡스톤 프로젝트.
기존 FxLMS 시스템(`~/anc_project`, 읽기전용)과 동일한 하드웨어를 사용하며,
학습은 **Elice Cloud A100**, 실시간 추론은 **NVIDIA Jetson AGX Orin** 에서 수행한다.

## 목표

1. **비선형·고주파 잡음 제거** — 스피커/앰프 포화 등 FxLMS 가 다루지 못하는 비선형성을 학습으로 극복
2. **Quiet zone** — 특정 소음원이 아닌 임의 소음의 제거 (단계적 접근, 아래 물리 한계 참조)

## 핵심 설계 요약

| 항목 | 값 |
|---|---|
| 모델 | `HybridANCNet` — Conv-TasNet 인코더/디코더 + WaveNet TCN + GCRN GLSTM + causal MHSA (시간영역, 48kHz) |
| 변형 | tiny 1.16M (Jetson 즉시 실시간) / base 5.99M (TRT 배포용) |
| 학습 | 미분가능 플랜트 `e = d + S(G_nl(y))` 에러신호 직접 최소화 (측정된 2차경로 사용) |
| 지연 규약 | 캘리브레이션 지연 1342 + 스레드 핸드오프 256 샘플, 지터 증강 [0, +512] |
| 실시간 | 3-스레드(콜백/추론/제어), 블록 256 (5.33ms), 안전장치 8종 |
| 실측 성능(Jetson) | tiny+ORT CPU **P99 1.50ms** (예산 5.33ms 통과) / base+ORT 6.8ms (TRT 필요) |

## 물리 한계 (반드시 읽을 것 → [docs/01_physics_limits.md](docs/01_physics_limits.md))

- **digital reference** (소음을 Jetson이 직접 생성): 상쇄 경로가 소음 경로보다 ~3ms 먼저 도달 →
  **광대역 상쇄가 인과적으로 가능**. 1차 릴리스 기본 모드.
- **acoustic reference** (외부 소음을 마이크로 수음): I/O 지연 ≈28ms ≫ 음향 전파 →
  **주기성/협대역 잡음만 예측 상쇄 가능**. 2단계 목표.
- 덕트 평면파 컷오프 **1,633Hz** — 그 이상은 고차 모드로 단일 스피커 제어가 물리적으로 제한됨.
  현실적 1차 목표 대역은 **80–800Hz** (덕트 구조 문서 기준).

## 빠른 시작

```bash
# ── Jetson (이 저장소가 있는 곳) ──────────────────────────────
python3 -m venv --without-pip --system-site-packages .venv    # 상세: docs/06
source .venv/bin/activate
pip install -r requirements-jetson.txt && pip install -e .
# torch 는 NVIDIA Jetson wheel 로 별도 설치 (docs/06_deployment_jetson.md)

pytest                                        # 검증 (30+ 테스트)
python scripts/data/build_rir_bank.py         # 덕트 RIR 뱅크 생성

# ── Elice Cloud (2×A100, VSCode CUDA 12.8) ──────────────────
git clone <이 저장소>
bash scripts/elice/setup_env.sh
bash scripts/data/download_noise.sh && python scripts/data/prepare_noise_pool.py
python scripts/data/build_rir_bank.py
bash scripts/elice/run_pretrain.sh            # GPU 수 자동 감지 (DDP)

# ── 학습 결과 → Jetson 배포 ─────────────────────────────────
python scripts/train/export_onnx.py --ckpt runs/pretrain_base/ckpt/best.pt --out runs/export/model.onnx
python scripts/bench/measure_inference_latency.py --config configs/runtime.yaml --set engine.type=ort --set engine.onnx=runs/export/model.onnx
python -m deep_anc.realtime.run_realtime --config configs/runtime.yaml --set engine.type=ort  # ⚠ 볼륨 낮추고 시작
```

## 문서 지도

| 문서 | 내용 |
|---|---|
| [00_overview.md](docs/00_overview.md) | 전체 그림, 3단계 로드맵, 저장소 지도 |
| [01_physics_limits.md](docs/01_physics_limits.md) | 지연 물리(2-모드), 컷오프, 정직한 기대치 |
| [02_hardware_setup.md](docs/02_hardware_setup.md) | 배선/채널맵, 마이크 점검, 2차경로 보정 |
| [03_data_pipeline.md](docs/03_data_pipeline.md) | 합성 데이터, RIR 뱅크, 실측 수집 프로토콜 |
| [04_model_architecture.md](docs/04_model_architecture.md) | 레이어 스펙, 스트리밍 상태, ONNX 규약 |
| [05_training_elice.md](docs/05_training_elice.md) | Elice A100 학습 절차, 비용 팁 |
| [06_deployment_jetson.md](docs/06_deployment_jetson.md) | Jetson 설치, 엔진 3종, 지연 실측치 |
| [07_evaluation_protocol.md](docs/07_evaluation_protocol.md) | 지표, 시나리오, FxLMS 비교 |
| [08_dev_workflow.md](docs/08_dev_workflow.md) | Jetson↔Elice git 워크플로, 프로젝트 정책 |
| [09_duct_structure.md](docs/09_duct_structure.md) | 덕트 실물 구조 (실측 문서) |
| [appendix_legacy_fxlms.md](docs/appendix_legacy_fxlms.md) | 기존 anc_project 분석·인벤토리 |

## 프로젝트 정책

- `~/anc_project` 는 **읽기 전용** — 수정 금지, 검증된 코드/측정 자산의 복사만 허용
- Jetson **시스템 불가침** — 핀 설정·RT 커널·전원모드·오디오 데몬 등 시스템 변경 금지 (의도된 구성)
- 안전 우선 — 실시간 실행은 항상 ANC OFF 로 시작, TPA3116D2 볼륨은 낮은 상태에서 점진 상향

## 라이선스

MIT — [LICENSE](LICENSE)
