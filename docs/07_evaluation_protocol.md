# 07. 평가 프로토콜

## 1. 지표

| 지표 | 정의 | 좋은 방향 |
|---|---|---|
| NMSE(dB) | 10·log₁₀(Σe²/Σd²) | 음수 ↓ |
| 감쇠(attenuation, dB) | −NMSE = 10·log₁₀(P_d/P_e) | 양수 ↑ |
| 옥타브밴드 감쇠 | 중심 125~8000Hz, 경계 f/√2~f√2 (버터워스 4차) | — |
| 세그먼트 분포 | 1s 세그먼트 감쇠의 중앙값 / 최악 10% | — |
| 실시간 건전성 | step P99(ms), deadline miss, xrun | ↓ |

**신뢰 표기**: S(z) 보정 유효대역(현재 150–600Hz) 밖의 밴드 수치는 `trusted=False`(*)로
표기한다 — 광대역 재보정(docs/02 §4) 후 유효대역을 갱신할 것 (설계 L2).

## 2. 시나리오 (configs/eval.yaml — 오프라인/실기 공통)

| 이름 | 소음 | 목적 |
|---|---|---|
| S1 `tone300` | 300Hz 톤 | FxLMS 대비 동등성 검증 |
| S2 `multitone` | 120+300+750Hz | 다중 협대역 |
| S3 `band` | 80–1000Hz 대역잡음 | digital-ref 광대역 능력 |
| S4 `nonlinear` | 210Hz+3·5차 고조파+소프트클립 | **DL 비선형 우위 입증 (핵심)** |
| S5 `file` | 실측 소음 WAV 루프 | 실전 데모 — eval.yaml 미등록: 실행 시 `--set noise.type=file --set noise.file=<wav>` 로 run_realtime 에 직접 지정 |

## 3. 오프라인 평가 (하드웨어 불필요)

```bash
# 테스트 split 종합 평가 → runs/<exp>/eval/{metrics.md, psd.png, spec.png, band.png}
python scripts/eval/evaluate_offline.py --ckpt runs/pretrain_base/ckpt/best.pt
# 동일 시나리오·동일 S(z) 에서 DL vs FxLMS 표
python scripts/eval/compare_fxlms.py --ckpt runs/pretrain_base/ckpt/best.pt
```

무학습 체크포인트 기준값 (파이프라인 검증, 2026-08-02): FxLMS 는 tone300 +88dB(이상 조건)
/ band +2.1dB / nonlinear +8.8dB, 무학습 DL 은 전부 0dB 부근 — 학습 후 이 표가 채워져야 한다.

## 4. 실기 평가 (덕트, 사용자 입회)

```bash
python scripts/demo/evaluate_session.py --controllers fxlms dl --scenarios tone300 multitone band nonlinear
```

프로토콜: 시나리오마다 **OFF 10s(베이스라인) → ON 30s → OFF 5s**, 게이트 램프 ±1~2s 는
분석에서 제외. 산출: `results/eval_report_<시각>.md` (전대역/밴드별 감쇠, miss/xrun) +
세션 원시 npz. FxLMS 와 DL 은 **같은 세션 묶음에서 연속 측정**해 조건을 통일한다.

## 5. 비교 공정성 원칙

1. 동일 소음 프로그램·레벨·볼륨 (시나리오 설정 공유)
2. 동일 S(z) 자산 (FxLMS 도 secondary_path_4s.npz)
3. FxLMS 수렴 시간을 인정 — ON 구간 후반부로 평가 (오프라인은 후반 1/3)
4. 실기에서는 마이크 캘리브레이션이 없어도 감쇠(비율)는 유효 — 절대 SPL 주장은 하지 않는다

## 6. 리포트 양식

`results/eval_report_*.md` 표 + 다음 플롯(오프라인 도구 재사용):
ANC OFF/ON 스펙트로그램, PSD 오버레이(off/FxLMS/DL), 옥타브밴드 막대(신뢰 회색 표기).
캡스톤 보고서에는 시나리오 표 + 밴드 막대 + 물리 한계 요약(docs/01 §5)을 함께 실을 것.
