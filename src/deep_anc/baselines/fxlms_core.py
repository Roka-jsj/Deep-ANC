#!/usr/bin/env python3
"""FxLMS core and small audio-device utilities for Jetson AGX Orin ANC.

The controller uses a causal block FxNLMS update:

    y[n] = W(z) x[n]
    e[n] = d[n] + S(z) y[n]
    w <- w - mu * X_f.T @ e / ||X_f||^2

where X_f is the reference signal filtered by the measured secondary path.
The minus sign is intentional.  No additional sign inversion should be applied
at the cancellation-speaker output; the measured secondary-path model already
contains the acoustic/electrical polarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
from scipy import signal

_FLOAT32_ONE = np.array([1.0], dtype=np.float32)
_INT32_SCALE = np.float32(1.0 / 2147483648.0)
_INT16_MAX = np.float32(32767.0)


@dataclass(frozen=True)
class SecondaryPathModel:
    """Measured cancellation-speaker -> error-microphone model."""

    fir: np.ndarray
    delay_samples: int
    sample_rate: int
    dc_block_r: float = 0.995
    fit_improvement_db: float = float("nan")
    coherence_median: float = float("nan")
    calibration_block_size: int = 0
    calibration_latency: str = "unknown"
    source_path: str = ""

    @property
    def total_length(self) -> int:
        return self.delay_samples + int(self.fir.size)

    @property
    def delay_ms(self) -> float:
        return 1000.0 * self.delay_samples / self.sample_rate


def _npz_scalar(data: Any, key: str, default: Any) -> Any:
    if key not in data:
        return default
    value = data[key]
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(-1)[0].item()
    return value


def load_secondary_path(path: str | Path) -> SecondaryPathModel:
    """Load the new NPZ model, or a legacy NPY impulse response.

    New calibration files store a compact FIR and a separate pure delay.  A
    legacy NPY file is also accepted.  Long leading zeros in a legacy array are
    separated into an explicit delay to keep real-time filtering affordable.
    """

    model_path = Path(path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Secondary-path model not found: {model_path}")

    if model_path.suffix.lower() == ".npz":
        with np.load(model_path, allow_pickle=False) as data:
            if "fir" not in data:
                raise ValueError(f"{model_path} does not contain a 'fir' array")
            fir = np.asarray(data["fir"], dtype=np.float32).reshape(-1)
            delay_samples = int(_npz_scalar(data, "delay_samples", 0))
            sample_rate = int(_npz_scalar(data, "sample_rate", 48000))
            dc_block_r = float(_npz_scalar(data, "dc_block_r", 0.995))
            fit_db = float(_npz_scalar(data, "fit_improvement_db", float("nan")))
            coherence = float(_npz_scalar(data, "coherence_median", float("nan")))
            block_size = int(_npz_scalar(data, "calibration_block_size", 0))
            latency = str(_npz_scalar(data, "calibration_latency", "unknown"))
    else:
        expanded = np.asarray(np.load(model_path, allow_pickle=False), dtype=np.float32).reshape(-1)
        if expanded.size == 0:
            raise ValueError(f"Empty secondary-path array: {model_path}")

        peak = float(np.max(np.abs(expanded)))
        if peak <= 0.0 or not np.isfinite(peak):
            raise ValueError(f"Invalid secondary-path array: {model_path}")

        # Keep a short pre-roll before the first meaningful coefficient.
        indices = np.flatnonzero(np.abs(expanded) >= peak * 1.0e-5)
        first = int(indices[0]) if indices.size else 0
        delay_samples = max(0, first - 32)
        fir = expanded[delay_samples:].copy()
        sample_rate = 48000
        dc_block_r = 0.995
        fit_db = float("nan")
        coherence = float("nan")
        block_size = 0
        latency = "legacy-npy"

    if fir.ndim != 1 or fir.size < 1:
        raise ValueError("Secondary-path FIR must be a non-empty 1-D array")
    if not np.all(np.isfinite(fir)):
        raise ValueError("Secondary-path FIR contains NaN or infinity")
    if float(np.max(np.abs(fir))) <= 0.0:
        raise ValueError("Secondary-path FIR is all zeros")
    if delay_samples < 0:
        raise ValueError("Secondary-path delay cannot be negative")
    if sample_rate <= 0:
        raise ValueError("Invalid model sample rate")
    if not 0.0 < dc_block_r < 1.0:
        raise ValueError("dc_block_r must be between 0 and 1")

    return SecondaryPathModel(
        fir=np.ascontiguousarray(fir, dtype=np.float32),
        delay_samples=delay_samples,
        sample_rate=sample_rate,
        dc_block_r=dc_block_r,
        fit_improvement_db=fit_db,
        coherence_median=coherence,
        calibration_block_size=block_size,
        calibration_latency=latency,
        source_path=str(model_path),
    )


def alsa_card_index(card_id: str) -> int:
    """Resolve an ALSA short card ID such as 'APE' or 'Audio'."""

    cards_path = Path("/proc/asound/cards")
    if not cards_path.exists():
        raise RuntimeError("/proc/asound/cards is not available")

    wanted = card_id.strip()
    pattern = re.compile(r"^\s*(\d+)\s+\[([^\]]+)\]")
    for line in cards_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if match and match.group(2).strip() == wanted:
            return int(match.group(1))

    raise RuntimeError(f"ALSA card ID '{wanted}' was not found in /proc/asound/cards")


def format_sounddevice_devices() -> str:
    """Return PortAudio/sounddevice devices as a readable table."""

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
    """Map an ALSA card ID/device number to a sounddevice device index."""

    import sounddevice as sd

    direction = direction.lower().strip()
    if direction not in {"input", "output"}:
        raise ValueError("direction must be 'input' or 'output'")

    devices = sd.query_devices()
    capability_key = "max_input_channels" if direction == "input" else "max_output_channels"

    if override_index is not None:
        if not 0 <= override_index < len(devices):
            raise RuntimeError(f"Invalid sounddevice index: {override_index}")
        if int(devices[override_index][capability_key]) < required_channels:
            raise RuntimeError(
                f"Device {override_index} does not provide {required_channels} {direction} channels"
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
        # Prefer the direct ALSA hardware entry if more than one name contains
        # the same token.
        for index in matches:
            if f"({token})" in str(devices[index]["name"]):
                return index
        return matches[0]

    raise RuntimeError(
        f"Could not map ALSA {card_id}, device {pcm_device} to a PortAudio "
        f"{direction} device.\nAvailable devices:\n{format_sounddevice_devices()}"
    )


def pcm_int32_to_float32(samples: np.ndarray) -> np.ndarray:
    """Convert S32_LE PCM to normalized float32 without changing channels."""

    return np.asarray(samples, dtype=np.int32).astype(np.float32) * _INT32_SCALE


def float32_to_pcm_int16(samples: np.ndarray) -> np.ndarray:
    """Clip normalized audio and convert it to S16_LE PCM."""

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


class DCBlocker:
    """Streaming first-order DC blocker: (1 - z^-1) / (1 - r z^-1)."""

    def __init__(self, r: float = 0.995) -> None:
        if not 0.0 < r < 1.0:
            raise ValueError("r must be between 0 and 1")
        self.r = float(r)
        self._b = np.array([1.0, -1.0], dtype=np.float32)
        self._a = np.array([1.0, -self.r], dtype=np.float32)
        self._zi = np.zeros(1, dtype=np.float32)

    def reset(self) -> None:
        self._zi.fill(0.0)

    def process(self, block: np.ndarray) -> np.ndarray:
        values = np.asarray(block, dtype=np.float32).reshape(-1)
        output, self._zi = signal.lfilter(self._b, self._a, values, zi=self._zi)
        return np.asarray(output, dtype=np.float32)


class SampleDelay:
    """A causal, streaming integer-sample delay."""

    def __init__(self, delay_samples: int) -> None:
        if delay_samples < 0:
            raise ValueError("delay_samples cannot be negative")
        self.delay_samples = int(delay_samples)
        self._state = np.zeros(self.delay_samples, dtype=np.float32)

    def reset(self) -> None:
        self._state.fill(0.0)

    def process(self, block: np.ndarray) -> np.ndarray:
        values = np.asarray(block, dtype=np.float32).reshape(-1)
        if self.delay_samples == 0:
            return values.copy()

        joined = np.concatenate((self._state, values))
        output = joined[: values.size].copy()
        self._state = joined[values.size : values.size + self.delay_samples].copy()
        return output


@dataclass(frozen=True)
class AdaptationResult:
    adapted: bool
    filtered_reference_power: float
    gradient_norm: float
    weight_norm: float
    weight_limited: bool
    reason: str


class FxLMSController:
    """Causal block FxNLMS controller with persistent FIR histories."""

    def __init__(
        self,
        s_hat: np.ndarray,
        secondary_delay_samples: int = 0,
        control_len: int = 256,
        mu: float = 0.05,
        leakage: float = 1.0e-6,
        normalization_epsilon: float = 1.0e-12,
        weight_norm_limit: float = 20.0,
    ) -> None:
        secondary = np.asarray(s_hat, dtype=np.float32).reshape(-1)
        if secondary.size < 1 or not np.all(np.isfinite(secondary)):
            raise ValueError("s_hat must be a finite, non-empty 1-D array")
        if float(np.max(np.abs(secondary))) <= 0.0:
            raise ValueError("s_hat cannot be all zeros")
        if control_len < 1:
            raise ValueError("control_len must be positive")
        if not 0.0 < mu <= 2.0:
            raise ValueError("mu should be in the interval (0, 2]")
        if not 0.0 <= leakage < 1.0:
            raise ValueError("leakage must be in [0, 1)")
        if normalization_epsilon <= 0.0:
            raise ValueError("normalization_epsilon must be positive")
        if weight_norm_limit <= 0.0:
            raise ValueError("weight_norm_limit must be positive")

        self.s_hat = np.ascontiguousarray(secondary, dtype=np.float32)
        self.secondary_delay_samples = int(secondary_delay_samples)
        self.control_len = int(control_len)
        self.mu = float(mu)
        self.leakage = float(leakage)
        self.normalization_epsilon = float(normalization_epsilon)
        self.weight_norm_limit = float(weight_norm_limit)

        self.w = np.zeros(self.control_len, dtype=np.float32)
        self._x_history = np.zeros(max(0, self.control_len - 1), dtype=np.float32)
        self._xf_history = np.zeros(max(0, self.control_len - 1), dtype=np.float32)
        self._secondary_delay = SampleDelay(self.secondary_delay_samples)
        self._secondary_state = np.zeros(max(0, self.s_hat.size - 1), dtype=np.float32)

        self._pending_xf_matrix: np.ndarray | None = None
        self._pending_frames = 0
        self.update_count = 0

    def reset(self, reset_histories: bool = False) -> None:
        """Zero adaptive weights; optionally clear all signal histories."""

        self.w.fill(0.0)
        self._pending_xf_matrix = None
        self._pending_frames = 0
        self.update_count = 0
        if reset_histories:
            self._x_history.fill(0.0)
            self._xf_history.fill(0.0)
            self._secondary_delay.reset()
            self._secondary_state.fill(0.0)

    def set_mu(self, mu: float) -> None:
        if not 0.0 < mu <= 2.0:
            raise ValueError("mu should be in the interval (0, 2]")
        self.mu = float(mu)

    @staticmethod
    def _tap_matrix(
        block: np.ndarray,
        history: np.ndarray,
        tap_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(block, dtype=np.float32).reshape(-1)
        if values.size == 0:
            return np.empty((0, tap_count), dtype=np.float32), history.copy()

        if tap_count == 1:
            return values.reshape(-1, 1).copy(), np.empty(0, dtype=np.float32)

        extended = np.concatenate((history, values))
        windows = np.lib.stride_tricks.sliding_window_view(extended, tap_count)
        # Tap 0 is x[n], tap 1 is x[n-1], ...
        matrix = np.ascontiguousarray(windows[:, ::-1], dtype=np.float32)
        new_history = extended[-(tap_count - 1) :].copy()
        return matrix, new_history

    def generate_block(self, reference_block: np.ndarray) -> np.ndarray:
        """Generate the cancellation-speaker block and prepare filtered-x data."""

        reference = np.asarray(reference_block, dtype=np.float32).reshape(-1)
        if reference.size == 0:
            self._pending_xf_matrix = np.empty((0, self.control_len), dtype=np.float32)
            self._pending_frames = 0
            return np.empty(0, dtype=np.float32)
        if not np.all(np.isfinite(reference)):
            reference = np.nan_to_num(reference, nan=0.0, posinf=0.0, neginf=0.0)

        x_matrix, self._x_history = self._tap_matrix(
            reference,
            self._x_history,
            self.control_len,
        )
        output = np.asarray(x_matrix @ self.w, dtype=np.float32)

        delayed_reference = self._secondary_delay.process(reference)
        filtered_reference, self._secondary_state = signal.lfilter(
            self.s_hat,
            _FLOAT32_ONE,
            delayed_reference,
            zi=self._secondary_state,
        )
        filtered_reference = np.asarray(filtered_reference, dtype=np.float32)

        xf_matrix, self._xf_history = self._tap_matrix(
            filtered_reference,
            self._xf_history,
            self.control_len,
        )
        self._pending_xf_matrix = xf_matrix
        self._pending_frames = reference.size
        return output

    def adapt_block(self, error_block: np.ndarray, enabled: bool = True) -> AdaptationResult:
        """Update coefficients from the error microphone for the pending block."""

        if self._pending_xf_matrix is None:
            raise RuntimeError("generate_block() must be called before adapt_block()")

        error = np.asarray(error_block, dtype=np.float32).reshape(-1)
        xf_matrix = self._pending_xf_matrix
        expected_frames = self._pending_frames
        self._pending_xf_matrix = None
        self._pending_frames = 0

        if error.size != expected_frames:
            raise ValueError(
                f"Error block has {error.size} frames, expected {expected_frames}"
            )

        if error.size == 0:
            return AdaptationResult(False, 0.0, 0.0, float(np.linalg.norm(self.w)), False, "empty")
        if not enabled:
            return AdaptationResult(
                False,
                float(np.sum(xf_matrix * xf_matrix)),
                0.0,
                float(np.linalg.norm(self.w)),
                False,
                "disabled",
            )
        if not np.all(np.isfinite(error)):
            return AdaptationResult(
                False,
                float(np.sum(xf_matrix * xf_matrix)),
                0.0,
                float(np.linalg.norm(self.w)),
                False,
                "non-finite error",
            )

        filtered_power = float(np.sum(xf_matrix * xf_matrix, dtype=np.float64))
        if not np.isfinite(filtered_power) or filtered_power <= self.normalization_epsilon:
            return AdaptationResult(
                False,
                filtered_power,
                0.0,
                float(np.linalg.norm(self.w)),
                False,
                "filtered reference too small",
            )

        gradient = np.asarray(xf_matrix.T @ error, dtype=np.float32)
        gradient_norm = float(np.linalg.norm(gradient))
        if not np.all(np.isfinite(gradient)):
            return AdaptationResult(
                False,
                filtered_power,
                gradient_norm,
                float(np.linalg.norm(self.w)),
                False,
                "non-finite gradient",
            )

        if self.leakage > 0.0:
            self.w *= np.float32(1.0 - self.leakage)

        # Physical acoustic summation is e = d + S*y, hence gradient descent
        # uses a minus sign here.  Do not negate y again in the output code.
        step = np.float32(self.mu / (filtered_power + self.normalization_epsilon))
        self.w -= step * gradient

        if not np.all(np.isfinite(self.w)):
            self.w.fill(0.0)
            raise FloatingPointError("FxLMS weights became non-finite and were reset")

        weight_norm = float(np.linalg.norm(self.w))
        weight_limited = False
        if weight_norm > self.weight_norm_limit:
            self.w *= np.float32(self.weight_norm_limit / weight_norm)
            weight_norm = self.weight_norm_limit
            weight_limited = True

        self.update_count += 1
        return AdaptationResult(
            True,
            filtered_power,
            gradient_norm,
            weight_norm,
            weight_limited,
            "ok",
        )

    def process_block(
        self,
        reference_block: np.ndarray,
        error_block: np.ndarray,
        adapt: bool = True,
    ) -> np.ndarray:
        """Convenience wrapper for generate_block() followed by adapt_block()."""

        output = self.generate_block(reference_block)
        self.adapt_block(error_block, enabled=adapt)
        return output

    def save_weights(self, path: str | Path) -> Path:
        output_path = Path(path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, self.w.astype(np.float32))
        return output_path

    def load_weights(self, path: str | Path) -> None:
        values = np.asarray(np.load(Path(path).expanduser(), allow_pickle=False), dtype=np.float32).reshape(-1)
        if values.size != self.control_len:
            raise ValueError(
                f"Weight file length {values.size} does not match control_len {self.control_len}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("Weight file contains NaN or infinity")
        self.w[:] = values


class _StreamingPath:
    """Small helper used only by the offline self-test."""

    def __init__(self, fir: np.ndarray, delay: int) -> None:
        self.fir = np.asarray(fir, dtype=np.float32)
        self.delay = SampleDelay(delay)
        self.zi = np.zeros(max(0, self.fir.size - 1), dtype=np.float32)

    def process(self, block: np.ndarray) -> np.ndarray:
        delayed = self.delay.process(block)
        output, self.zi = signal.lfilter(self.fir, _FLOAT32_ONE, delayed, zi=self.zi)
        return np.asarray(output, dtype=np.float32)


def _self_test() -> int:
    rng = np.random.default_rng(20260801)
    sample_rate = 8000
    block_size = 128
    sample_count = sample_rate * 8

    raw = rng.standard_normal(sample_count).astype(np.float32)
    sos = signal.butter(4, [80.0, 1200.0], btype="bandpass", fs=sample_rate, output="sos")
    reference = signal.sosfilt(sos, raw).astype(np.float32)
    reference *= np.float32(0.08 / (np.std(reference) + 1.0e-12))

    primary_fir = np.array([0.70, 0.22, -0.08, 0.04], dtype=np.float32)
    secondary_fir = np.array([0.52, 0.18, -0.09, 0.03], dtype=np.float32)
    primary = _StreamingPath(primary_fir, delay=48)
    secondary = _StreamingPath(secondary_fir, delay=20)

    controller = FxLMSController(
        secondary_fir,
        secondary_delay_samples=20,
        control_len=64,
        mu=0.45,
        leakage=1.0e-6,
        weight_norm_limit=10.0,
    )

    errors: list[np.ndarray] = []
    disturbances: list[np.ndarray] = []

    for start in range(0, sample_count, block_size):
        x_block = reference[start : start + block_size]
        y_block = controller.generate_block(x_block)
        disturbance = primary.process(x_block)
        error = disturbance + secondary.process(y_block)
        controller.adapt_block(error, enabled=True)
        disturbances.append(disturbance)
        errors.append(error)

    disturbance_all = np.concatenate(disturbances)
    error_all = np.concatenate(errors)
    final_slice = slice(sample_count - 2 * sample_rate, sample_count)
    reduction_db = 10.0 * np.log10(
        (np.mean(disturbance_all[final_slice] ** 2) + 1.0e-20)
        / (np.mean(error_all[final_slice] ** 2) + 1.0e-20)
    )

    print("FxLMS offline self-test")
    print(f"  final error reduction : {reduction_db:7.2f} dB")
    print(f"  coefficient updates   : {controller.update_count}")
    print(f"  final weight norm     : {np.linalg.norm(controller.w):.6f}")

    if not np.isfinite(reduction_db) or reduction_db < 6.0:
        print("  RESULT                : FAIL")
        return 1

    print("  RESULT                : PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
