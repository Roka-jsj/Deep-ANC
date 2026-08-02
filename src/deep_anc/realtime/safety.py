"""안전장치 — anc_project 실기 검증 패턴 계승 + DL 특화 워치독 2종.

전 모드(dl/fxlms) 공통 적용 (docs/06):
  1. 시작 시 ANC OFF          2. FadeGate 페이드 온/오프
  3. NaN 방어 + 출력 리미터    4. 클립 스트릭 자동 mute
  5. 발산 워치독(신규)         6. 추론 데드라인 워치독(신규)
  7. 입력 클리핑 통계          8. 종료 페이드 시퀀스
"""

from __future__ import annotations

import numpy as np


class FadeGate:
    """0↔1 선형 페이드 게이트 (anc_project/main_realtime_anc.py 이식)."""

    def __init__(self, fade_samples: int, initial: float = 0.0) -> None:
        self.fade_samples = max(0, int(fade_samples))
        self.current = float(initial)
        self.target = float(initial)
        self.remaining = 0

    def set_target(self, target: float) -> None:
        target = float(np.clip(target, 0.0, 1.0))
        if target == self.target:
            return
        self.target = target
        self.remaining = self.fade_samples
        if self.fade_samples == 0:
            self.current = self.target

    def process(self, frames: int) -> np.ndarray:
        if frames <= 0:
            return np.empty(0, dtype=np.float32)
        if self.remaining <= 0 or self.current == self.target:
            self.current = self.target
            self.remaining = 0
            return np.full(frames, self.current, dtype=np.float32)
        count = min(frames, self.remaining)
        start = self.current
        ramp = start + (self.target - start) * (
            np.arange(1, count + 1, dtype=np.float64) / self.remaining
        )
        output = np.empty(frames, dtype=np.float32)
        output[:count] = ramp.astype(np.float32)
        self.current = float(ramp[-1])
        self.remaining -= count
        if count < frames:
            output[count:] = np.float32(self.target)
            self.current = self.target
            self.remaining = 0
        return output


class PowerEMA:
    """지수이동평균 파워 미터 (anc_project 이식)."""

    def __init__(self, sample_rate: int, time_constant: float = 0.5) -> None:
        self.sample_rate = sample_rate
        self.time_constant = max(1.0e-3, float(time_constant))
        self.value = 0.0
        self.initialized = False

    def update(self, block: np.ndarray) -> float:
        values = np.asarray(block, dtype=np.float64)
        power = float(np.mean(values * values)) if values.size else 0.0
        alpha = float(np.exp(-values.size / (self.sample_rate * self.time_constant)))
        if not self.initialized:
            self.value = power
            self.initialized = True
        else:
            self.value = alpha * self.value + (1.0 - alpha) * power
        return self.value


class SafetySupervisor:
    """콜백 안에서 매 블록 호출되는 안전 감시자 — 상태와 자동 mute 판단."""

    def __init__(self, cfg: dict, sample_rate: int, block_size: int) -> None:
        self.control_limit = float(cfg.get("control_limit", 0.2))
        self.clip_streak_mute = int(cfg.get("clip_streak_mute", 20))
        self.divergence_ratio = float(cfg.get("divergence_ratio", 4.0))
        self.divergence_hold_blocks = max(
            1, int(float(cfg.get("divergence_hold_s", 0.5)) * sample_rate / block_size)
        )
        self.deadline_miss_mute = int(cfg.get("deadline_miss_mute", 3))

        self.clip_streak = 0
        self.divergence_streak = 0
        self.deadline_streak = 0
        self.messages: list[str] = []

    def limit_output(self, y: np.ndarray) -> tuple[np.ndarray, float]:
        """NaN 방어 + 하드 리미터. 반환: (제한된 신호, 클립 비율)."""
        y = np.nan_to_num(np.asarray(y, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        clipped = np.abs(y) > self.control_limit
        frac = float(np.mean(clipped)) if clipped.size else 0.0
        return np.clip(y, -self.control_limit, self.control_limit), frac

    def check_block(
        self,
        anc_on: bool,
        clip_fraction: float,
        error_power: float,
        baseline_power: float,
        had_data: bool,
    ) -> bool:
        """블록마다 호출. True 를 반환하면 ANC 를 즉시 OFF 해야 한다."""
        mute = False

        # 4) 클립 스트릭
        self.clip_streak = self.clip_streak + 1 if clip_fraction > 0.0 else 0
        if anc_on and self.clip_streak >= self.clip_streak_mute:
            self.messages.append(
                "안전 mute: 상쇄 출력이 연속으로 리미터에 걸렸습니다 — 볼륨/모델 확인"
            )
            self.clip_streak = 0
            mute = True

        # 5) 발산 워치독: 에러 파워가 베이스라인 ×ratio 지속
        if anc_on and baseline_power > 0.0 and error_power > self.divergence_ratio * baseline_power:
            self.divergence_streak += 1
            if self.divergence_streak >= self.divergence_hold_blocks:
                self.messages.append(
                    f"발산 워치독: 에러 파워가 베이스라인 대비 "
                    f"{10*np.log10(error_power/max(baseline_power,1e-30)):.1f}dB 상승 — 자동 OFF"
                )
                self.divergence_streak = 0
                mute = True
        else:
            self.divergence_streak = 0

        # 6) 추론 데드라인 워치독: out_ring 연속 언더런
        if anc_on and not had_data:
            self.deadline_streak += 1
            if self.deadline_streak >= self.deadline_miss_mute:
                self.messages.append(
                    "데드라인 워치독: 추론이 콜백을 따라가지 못합니다 — DL 출력 mute "
                    "(엔진을 trt 로 바꾸거나 tiny 모델/폴백을 사용하세요)"
                )
                self.deadline_streak = 0
                mute = True
        elif had_data:
            self.deadline_streak = 0

        return mute

    def drain_messages(self) -> list[str]:
        out, self.messages = self.messages, []
        return out
