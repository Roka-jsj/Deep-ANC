"""digital-reference 학습용 1차경로 P(z) 선택과 단위 정합.

실측 primary FIR은 noise 출력부터 error mic 입력까지의 gain/FIR/순수지연을 모두
포함한다. 합성 1D RIR은 절대 장치 gain이 없으므로 실측 S(z)와 직접 결합해 물리
성능을 주장할 수 없다. 이를 숨기지 않도록 모드를 명시적으로 분리한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import _resolve_path, default_d_noise_delay
from ..dsp.secondary_path import SecondaryPathData, load_secondary_path


@dataclass(frozen=True)
class DigitalPrimaryPath:
    """순수지연과 compact FIR로 분리된 digital noise→ERR 경로."""

    fir: np.ndarray
    delay_samples: int
    mode: str
    source_path: str
    is_surrogate: bool


def resolve_digital_primary_path(
    data_cfg: dict,
    duct_cfg: dict,
    sample_rate: int,
    secondary_path: SecondaryPathData,
) -> tuple[DigitalPrimaryPath | None, int]:
    """설정에서 P(z)를 해석한다.

    반환 두 번째 값은 D_noise 총 순수지연이다. `rir_surrogate`에서는 기존 p_err
    RIR의 음향 onset을 고려해 dataset이 추가지연을 따로 계산한다. 나머지 두 모드는
    반환된 FIR과 총지연을 그대로 한 번만 적용한다.
    """

    mode = str(data_cfg.get("digital_primary_path_mode", "rir_surrogate"))
    allowed = {"rir_surrogate", "secondary_surrogate", "measured"}
    if mode not in allowed:
        raise ValueError(
            f"digital_primary_path_mode={mode!r}; 허용값은 {sorted(allowed)}"
        )

    digital_cfg = duct_cfg.get("digital_reference", {})
    configured_delay = digital_cfg.get("d_noise_delay_samples")
    fallback_delay = (
        int(configured_delay)
        if configured_delay is not None
        else default_d_noise_delay(
            duct_cfg, int(sample_rate), int(secondary_path.delay_samples)
        )
    )

    if mode == "rir_surrogate":
        return None, fallback_delay

    if mode == "secondary_surrogate":
        # 실측 P가 없을 때에만 쓰는 표현 사전학습용 모드. S의 장치 gain/FIR을
        # 빌리되 D_noise 지연을 적용해 P/S scale infeasibility를 피한다.
        return (
            DigitalPrimaryPath(
                fir=np.ascontiguousarray(secondary_path.fir, dtype=np.float32),
                delay_samples=fallback_delay,
                mode=mode,
                source_path=secondary_path.source_path,
                is_surrogate=True,
            ),
            fallback_delay,
        )

    path = digital_cfg.get("primary_path_npz")
    if not path:
        raise ValueError(
            "digital_primary_path_mode='measured'에는 "
            "duct.digital_reference.primary_path_npz가 필요합니다. "
            "noise→ERR ESS 측정 파일을 지정하세요."
        )
    primary = load_secondary_path(_resolve_path(path))
    if int(primary.sample_rate) != int(sample_rate):
        raise ValueError(
            f"P(z) sample rate {primary.sample_rate} != 학습 sample rate {sample_rate}"
        )
    if configured_delay is not None and int(configured_delay) != int(primary.delay_samples):
        raise ValueError(
            "digital_reference.d_noise_delay_samples와 primary_path_npz의 delay가 "
            f"다릅니다: {configured_delay} != {primary.delay_samples}"
        )
    return (
        DigitalPrimaryPath(
            fir=np.ascontiguousarray(primary.fir, dtype=np.float32),
            delay_samples=int(primary.delay_samples),
            mode=mode,
            source_path=primary.source_path,
            is_surrogate=False,
        ),
        int(primary.delay_samples),
    )
