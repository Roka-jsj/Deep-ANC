# 08. 개발 워크플로와 프로젝트 정책

## 1. 저장소 정책

| 정책 | 내용 |
|---|---|
| **anc_project 읽기전용** | `~/anc_project`(기존 FxLMS)는 절대 수정 금지. 검증된 코드(fxlms_core)와 측정 자산(npz)의 **복사만** 허용 — 출처를 주석으로 명기 |
| **Jetson 시스템 불가침** | 핀 설정(pinmux/I2S)·RT 커널·전원모드·오디오 데몬·apt 설치 등 **시스템 변경 금지** (의도된 실험 구성). 모든 도구는 venv/유저 공간에서만 |
| 대용량 산출물 | `data/`, `runs/`, `*.pt`, `*.onnx`, `*.plan` 은 .gitignore — 가중치는 GitHub Release 자산으로 |
| 커밋 자산 예외 | `assets/measured/*.npz` (수십 KB 측정 자산)는 저장소에 포함 |

## 2. Jetson ↔ Elice 동기화 (git 허브 모델)

```
[Jetson]  코드 개발·실기 검증 → git push
                                   ↓
[GitHub]  단일 동기화 지점 (원격 저장소)
                                   ↓ git pull/clone
[Elice]   학습 실행 → 작은 산출물(설정, metrics, 로그 요약)만 커밋
          가중치/onnx 는 zip 다운로드 또는 GitHub Release 업로드
```

- 브랜치: `main` 은 항상 실행 가능 상태 유지. 실험은 `exp/<이름>` 브랜치.
- 실측 녹음(data/recorded)은 크기가 작으면 zip 으로 Elice 에 직접 업로드,
  크면 `pack_transfer.py` 샤드.

## 3. 재현성

- 모든 실행은 config yaml 이 단일 출처 — CLI 는 `--set key=value` 오버라이드만.
- `train/reproducibility.py` 가 run 디렉토리에 config 스냅샷·git rev·pip freeze 자동 기록.
- 체크포인트는 모델+옵티마이저+스케줄러+step+RNG 를 포함해 `--resume` 완전 재개.
- 시드: 학습 `seed`(+rank), 데이터 분할 고정 시드, val 배치 고정 시드(1234/999).
- 버전 고정: Elice torch 2.5.1+cu121 ↔ Jetson 2.5.0a0(JP6.1) 정렬, ORT 1.18.1(Jetson).

## 4. 코드 품질 게이트

```bash
pytest -q                 # 커밋 전 통과 필수 (30+)
```

핵심 불변식 (테스트가 강제):
1. 모델 인과성 — 미래 입력 무의존 (비트 단위)
2. 스트리밍 = 오프라인 등가 (≤1e-5), GLSTM nn.LSTM = 수동 셀 등가
3. S(z) torch = scipy 등가, 극성 규약(e = d + S·y, 추가 반전 금지)
4. 덕트 시뮬이 이론 공진(70/210/350Hz) 재현
5. 데이터 분할 무누수 (파일/세션/RIR 변형 단위)

## 5. 단계별 진행 체크리스트

- [ ] USB 오디오(AB13X) 연결 → docs/02 §3 하드웨어 점검 1~5
- [ ] 레퍼런스 마이크 ch1 수리 확인 (acoustic-ref 선결)
- [ ] `--calibrate` 실효 지연 확인 (핸드오프 정합)
- [ ] D_noise 실측 (`calibrate_wideband.py --output-channel noise`) → duct.yaml 기입
- [ ] Elice 사전학습 → export → Jetson ORT 벤치(게이트 P99<3ms) → 실기 데모
- [ ] 실측 녹음 수집 → 파인튜닝 → evaluate_session 리포트
- [ ] (선택) S(z) 광대역 재보정 → 커리큘럼 B / (선택) TRT 바인딩 설치(사용자 판단) → base 배포
- [ ] 덕트 문서 미확정 항목 확정 시 duct.yaml 갱신 (ERR 위치 등) + RIR 뱅크 재생성
