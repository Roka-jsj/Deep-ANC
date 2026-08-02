"""1D 덕트 음향 시뮬레이터 (영상법, closed–open 경계).

configs/duct.yaml 의 기하(docs/09_duct_structure.md rev.3 확정값)로부터
소음스피커/상쇄스피커/마이크 간 임펄스 응답(IR)을 생성한다.

모델링 범위와 한계 (정직하게):
  - 평면파 1D 모델 — 컷오프(1633Hz) 이상의 고차 모드는 모델링하지 않는다.
    고주파 성분은 per-reflection 저역통과 근사로 감쇠만 반영한다.
  - side-branch 상쇄 스피커의 국소 방사 특성은 IR 스케일로 흡수한다(학습은
    측정된 S(z)를 사용하므로 시뮬 S는 참고용).
  - 검증: closed–open 공진(70/210/350/489/629Hz)이 재현되는지
    tests/test_duct_sim.py 에서 확인한다.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def effective_length(duct_cfg: dict) -> float:
    """개방단 보정 포함 유효 길이 L_eff = L_int + 0.61·r_eq."""
    d = duct_cfg["duct"]
    a, b = (float(v) for v in d["cross_section_m"])
    area = a * b
    r_eq = float(np.sqrt(area / np.pi))
    return float(d["interior_length_m"]) + float(d.get("end_correction_factor", 0.61)) * r_eq


def image_source_ir(
    x_src: float,
    x_rcv: float,
    length_eff: float,
    sample_rate: int,
    speed_of_sound: float = 343.0,
    r_closed: float = 0.8,
    r_open: float = -0.45,
    max_order: int = 60,
    ir_len: int = 8192,
    alpha_per_m: float = 0.12,
    lowpass_hz: float | None = 4000.0,
) -> np.ndarray:
    """closed(x=0)–open(x=L_eff) 1D 관의 x_src → x_rcv 임펄스 응답.

    영상법: 소스 이미지 위치
      family A: s = x_src + 2mL, 게인 (r_c·r_o)^|m|
      family B: s = -x_src + 2mL,
                 m<=0 → r_c^{|m|+1}·r_o^{|m|},  m>=1 → r_o^{m}·r_c^{m-1}
    각 도달은 진폭 gain·exp(-α·d) 로, 분수 지연은 선형 보간으로 배치한다.
    """
    L = float(length_eff)
    ir = np.zeros(ir_len, dtype=np.float64)

    def place(distance: float, gain: float) -> None:
        if abs(gain) < 1.0e-6:
            return
        delay = distance / speed_of_sound * sample_rate
        idx = int(np.floor(delay))
        frac = delay - idx
        amp = gain * np.exp(-alpha_per_m * distance)
        if 0 <= idx < ir_len - 1:
            ir[idx] += amp * (1.0 - frac)
            ir[idx + 1] += amp * frac

    for m in range(-max_order, max_order + 1):
        # family A
        s = x_src + 2.0 * m * L
        gain_a = (r_closed * r_open) ** abs(m)
        place(abs(x_rcv - s), gain_a)
        # family B
        s = -x_src + 2.0 * m * L
        if m <= 0:
            gain_b = (r_closed ** (abs(m) + 1)) * (r_open ** abs(m))
        else:
            gain_b = (r_open ** m) * (r_closed ** (m - 1))
        place(abs(x_rcv - s), gain_b)

    if lowpass_hz is not None and lowpass_hz < sample_rate / 2:
        sos = signal.butter(2, lowpass_hz, btype="lowpass", fs=sample_rate, output="sos")
        ir = signal.sosfilt(sos, ir)

    return ir.astype(np.float32)


def duct_paths(duct_cfg: dict, sample_rate: int, ir_len: int = 8192, **overrides) -> dict[str, np.ndarray]:
    """주요 경로 IR 일괄 생성.

    반환 키:
      p_ref : 소음스피커 → 레퍼런스 마이크 (acoustic-ref 모드 입력 경로)
      p_err : 소음스피커 → 에러 마이크 위치 (primary path)
      f_fb  : 상쇄스피커 → 레퍼런스 마이크 (피드백 경로, acoustic-ref 2단계용)
      s_ac  : 상쇄스피커 → 에러 마이크 (참고용 — 학습 플랜트는 측정 S(z) 사용)
    """
    pos = duct_cfg["positions_m"]
    refl = duct_cfg.get("reflection", {})
    L = effective_length(duct_cfg)
    c = float(duct_cfg["duct"]["speed_of_sound_mps"])

    kwargs = dict(
        length_eff=L,
        sample_rate=sample_rate,
        speed_of_sound=c,
        r_closed=float(refl.get("closed_end", 0.8)),
        r_open=float(refl.get("open_end", -0.45)),
        lowpass_hz=refl.get("per_reflection_lowpass_hz", 4000.0),
        ir_len=ir_len,
    )
    kwargs.update(overrides)

    def path(a: str, b: str) -> np.ndarray:
        if pos.get(a) is None or pos.get(b) is None:
            raise ValueError(f"duct.yaml positions_m.{a}/{b} 가 필요합니다")
        return image_source_ir(float(pos[a]), float(pos[b]), **kwargs)

    return {
        "p_ref": path("noise_speaker", "reference_mic"),
        "p_err": path("noise_speaker", "error_mic"),
        "f_fb": path("cancel_speaker", "reference_mic"),
        "s_ac": path("cancel_speaker", "error_mic"),
    }


def transfer_magnitude(ir: np.ndarray, sample_rate: int, nfft: int = 32768) -> tuple[np.ndarray, np.ndarray]:
    """IR의 전달함수 크기 (주파수축, |H|)."""
    H = np.fft.rfft(ir, nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
    return freqs, np.abs(H)


def find_resonances(ir: np.ndarray, sample_rate: int, fmax: float = 800.0) -> np.ndarray:
    """전달함수 피크(공진) 주파수 추출 — closed–open 이론값과 비교 검증용."""
    freqs, mag = transfer_magnitude(ir, sample_rate)
    band = freqs <= fmax
    freqs, mag = freqs[band], mag[band]
    peaks, _ = signal.find_peaks(mag, prominence=np.max(mag) * 0.05)
    return freqs[peaks]


def build_rir_bank(
    duct_cfg: dict,
    sample_rate: int,
    n_variants: int = 300,
    seed: int = 20260802,
    ir_len: int = 8192,
) -> dict[str, np.ndarray]:
    """도메인 랜덤화 RIR 뱅크 — 시뮬-실측 갭 흡수.

    변형 축: 반사계수, 감쇠, 마이크/스피커 위치 소폭 지터(±1cm), 저역통과 컷오프.
    반환: {"p_ref": [N, L], "p_err": [N, L], "f_fb": [N, L], "meta": ...}
    """
    rng = np.random.default_rng(seed)
    pos0 = duct_cfg["positions_m"]
    banks: dict[str, list[np.ndarray]] = {"p_ref": [], "p_err": [], "f_fb": []}

    for _ in range(n_variants):
        cfg = {
            "duct": dict(duct_cfg["duct"]),
            "positions_m": {
                k: (None if v is None else float(v) + float(rng.uniform(-0.01, 0.01)))
                for k, v in pos0.items()
            },
            "reflection": {
                "closed_end": float(rng.uniform(0.6, 0.95)),
                "open_end": float(rng.uniform(-0.6, -0.25)),
                "per_reflection_lowpass_hz": float(rng.uniform(3000.0, 6000.0)),
            },
        }
        paths = duct_paths(cfg, sample_rate, ir_len=ir_len, alpha_per_m=float(rng.uniform(0.05, 0.25)))
        for key in banks:
            banks[key].append(paths[key])

    return {key: np.stack(vals).astype(np.float32) for key, vals in banks.items()}
