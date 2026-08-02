"""FxLMS 오프라인 베이스라인 — DL 과 동일 시나리오·동일 S(z) 비교용.

anc_project 의 FxLMSController(사본)를 블록 단위로 구동해 e(n) 궤적을 만든다.
물리 규약: e = d + S·y (부호 반전 금지 — fxlms_core docstring).
"""

from __future__ import annotations

import numpy as np

from ..baselines.fxlms_core import FxLMSController
from ..dsp.filters import StreamingFIR


def run_fxlms_offline(
    x_ref: np.ndarray,
    d: np.ndarray,
    s_fir: np.ndarray,
    s_delay: int,
    block: int = 256,
    control_len: int = 256,
    mu: float = 0.05,
    leakage: float = 1.0e-6,
) -> dict:
    """반환: e(에러 궤적), y(제어 출력), 최종 1/3 구간 감쇠 dB."""
    controller = FxLMSController(
        s_fir,
        secondary_delay_samples=s_delay,
        control_len=control_len,
        mu=mu,
        leakage=leakage,
    )
    plant = StreamingFIR(s_fir, s_delay)

    n = min(x_ref.size, d.size)
    n -= n % block
    e_out = np.zeros(n, dtype=np.float32)
    y_out = np.zeros(n, dtype=np.float32)

    for start in range(0, n, block):
        sl = slice(start, start + block)
        y = controller.generate_block(x_ref[sl])
        e = d[sl] + plant.process(y)
        controller.adapt_block(e, enabled=True)
        e_out[sl] = e
        y_out[sl] = y

    tail = slice(2 * n // 3, n)
    att_db = 10.0 * np.log10(
        (np.mean(d[tail] ** 2) + 1e-20) / (np.mean(e_out[tail] ** 2) + 1e-20)
    )
    return {"e": e_out, "y": y_out, "attenuation_db_tail": float(att_db)}
