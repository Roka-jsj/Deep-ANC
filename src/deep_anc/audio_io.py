"""오디오 장치 해석·PCM 변환 유틸.

anc_project/fxlms_core.py 에서 실기 검증된 함수를 그대로 이식했다 (출처 명기).
Jetson AGX Orin: 입력 hw:APE,1 (S32_LE), 출력 AB13X USB (S16_LE).
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np

_INT32_SCALE = np.float32(1.0 / 2147483648.0)
_INT16_MAX = np.float32(32767.0)


def alsa_card_index(card_id: str) -> int:
    """ALSA 짧은 카드 ID('APE', 'Audio')를 카드 번호로 해석."""
    cards_path = Path("/proc/asound/cards")
    if not cards_path.exists():
        raise RuntimeError("/proc/asound/cards 를 읽을 수 없습니다")

    wanted = card_id.strip()
    pattern = re.compile(r"^\s*(\d+)\s+\[([^\]]+)\]")
    for line in cards_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if match and match.group(2).strip() == wanted:
            return int(match.group(1))

    raise RuntimeError(f"ALSA 카드 ID '{wanted}' 를 찾지 못했습니다")


def format_sounddevice_devices() -> str:
    """PortAudio/sounddevice 장치 목록을 표로 반환."""
    import sounddevice as sd

    lines = []
    for index, device in enumerate(sd.query_devices()):
        lines.append(
            f"{index:3d}: in={int(device['max_input_channels']):2d}, "
            f"out={int(device['max_output_channels']):2d}, "
            f"rate={float(device['default_samplerate']):8.1f} | {device['name']}"
        )
    return "\n".join(lines)


def resolve_alsa_portaudio_device(
    card_id: str,
    pcm_device: int,
    direction: str,
    required_channels: int,
    override_index: int | None = None,
) -> int:
    """ALSA 카드 ID/디바이스 번호를 sounddevice 장치 인덱스로 매핑."""
    import sounddevice as sd

    direction = direction.lower().strip()
    if direction not in {"input", "output"}:
        raise ValueError("direction 은 'input' 또는 'output'")

    devices = sd.query_devices()
    capability_key = "max_input_channels" if direction == "input" else "max_output_channels"

    if override_index is not None:
        if not 0 <= override_index < len(devices):
            raise RuntimeError(f"잘못된 sounddevice 인덱스: {override_index}")
        if int(devices[override_index][capability_key]) < required_channels:
            raise RuntimeError(
                f"장치 {override_index} 는 {direction} {required_channels}채널을 지원하지 않습니다"
            )
        return int(override_index)

    card_number = alsa_card_index(card_id)
    token = f"hw:{card_number},{int(pcm_device)}"
    matches: list[int] = []

    for index, device in enumerate(devices):
        name = str(device["name"])
        if token in name and int(device[capability_key]) >= required_channels:
            matches.append(index)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        for index in matches:
            if f"({token})" in str(devices[index]["name"]):
                return index
        return matches[0]

    raise RuntimeError(
        f"ALSA {card_id}, device {pcm_device} 를 PortAudio {direction} 장치로 매핑하지 "
        f"못했습니다.\n장치 목록:\n{format_sounddevice_devices()}"
    )


def pcm_int32_to_float32(samples: np.ndarray) -> np.ndarray:
    """S32_LE PCM → 정규화 float32 (채널 유지)."""
    return np.asarray(samples, dtype=np.int32).astype(np.float32) * _INT32_SCALE


def float32_to_pcm_int16(samples: np.ndarray) -> np.ndarray:
    """정규화 float32 → 클리핑 후 S16_LE PCM."""
    values = np.asarray(samples, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.rint(np.clip(values, -1.0, 1.0) * _INT16_MAX).astype(np.int16)


def rms_dbfs(samples: np.ndarray, floor_db: float = -200.0) -> float:
    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0:
        return floor_db
    power = float(np.mean(values * values))
    if not np.isfinite(power) or power <= 0.0:
        return floor_db
    return max(floor_db, 10.0 * float(np.log10(power)))
