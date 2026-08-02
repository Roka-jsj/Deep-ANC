"""실시간 Deep ANC 런타임 — 3-스레드 구조 (콜백 / 추론 / 제어).

  python -m deep_anc.realtime.run_realtime --config configs/runtime.yaml
  python -m deep_anc.realtime.run_realtime --config configs/runtime.yaml --set controller=fxlms
  python -m deep_anc.realtime.run_realtime --config configs/runtime.yaml --calibrate

구조 (docs/06):
  [콜백]   입력 변환/DC차단 → in_ring, 소음(ch0) 생성, out_ring→리미터/게이트→ch1 출력
  [추론]   in_ring 에서 hop 단위 소비 → engine.step → out_ring  (콜백은 절대 대기 안 함)
  [제어]   키보드, 1초 통계, 워치독 메시지
파이프라인 핸드오프 지연 = 1 hop — 학습 플랜트의 handoff_extra_samples 와 정합 [C1].
시작은 항상 ANC OFF. 시스템(전원모드/RT우선순위 등)은 건드리지 않는다 — 프로젝트 정책.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from ..audio_io import (
    float32_to_pcm_int16,
    format_sounddevice_devices,
    pcm_int32_to_float32,
    resolve_alsa_portaudio_device,
)
from ..config import apply_overrides, load_runtime_config
from ..dsp.filters import DCBlocker
from .engines import build_engine
from .noise_gen import NoiseProgram
from .ring_buffer import SPSCRing
from .safety import FadeGate, PowerEMA, SafetySupervisor
from .ui import KeyboardController, RuntimeState


def power_to_db(power: float, floor_db: float = -200.0) -> float:
    if not np.isfinite(power) or power <= 0.0:
        return floor_db
    return max(floor_db, 10.0 * float(np.log10(power)))


class RealtimeANC:
    """프로그래밍 API — evaluate_session 등에서 재사용. CLI 는 main() 참조."""

    def __init__(self, cfg: dict, record_seconds: float = 0.0) -> None:
        import sounddevice as sd

        self.sd = sd
        self.cfg = cfg
        hw = cfg["hardware"]["audio"]
        self.fs = int(hw["sample_rate"])
        self.block = int(hw["block_size"])
        self.hop = int(cfg.get("hop", self.block))
        if self.hop != self.block:
            raise ValueError("현재 구현은 hop == block_size 를 요구합니다")
        ch = cfg["hardware"]["channels"]
        self.ch_err, self.ch_ref = int(ch["error_mic"]), int(ch["reference_mic"])
        self.ch_noise, self.ch_cancel = int(ch["noise_out"]), int(ch["cancel_out"])
        self.reference = str(cfg.get("reference", "digital"))

        self.in_dev = resolve_alsa_portaudio_device(hw["input"]["card"], hw["input"]["pcm"], "input", 2)
        self.out_dev = resolve_alsa_portaudio_device(hw["output"]["card"], hw["output"]["pcm"], "output", 2)

        self.engine = build_engine(cfg)
        self.program = NoiseProgram(cfg.get("noise", {}), self.fs)

        dc_r = float(cfg["hardware"].get("dc_blocker_r", 0.995))
        self.err_dc, self.ref_dc = DCBlocker(dc_r), DCBlocker(dc_r)

        safety_cfg = cfg.get("safety", {})
        self.safety = SafetySupervisor(safety_cfg, self.fs, self.block)
        fade = int(float(safety_cfg.get("fade_ms", 20.0)) * self.fs / 1000.0)
        self.state = RuntimeState(start_on=bool(cfg.get("start_on", False)))
        self.anc_gate = FadeGate(fade, initial=1.0 if self.state.anc_enabled else 0.0)
        self.noise_gate = FadeGate(max(fade, int(0.1 * self.fs)), initial=0.0)
        self.noise_gate.set_target(1.0)

        self.in_ring = SPSCRing(3, self.hop * 64)      # err, ref_mic, ref_digital
        self.out_ring = SPSCRing(1, self.hop * 64)

        self.err_meter = PowerEMA(self.fs, 0.4)
        self.ctrl_meter = PowerEMA(self.fs, 0.4)
        self.baseline_power = 0.0
        self.baseline_init = False
        self.step_times_ms: list[float] = []
        self.xruns = 0
        self._last_anc = self.state.anc_enabled

        self.record_len = int(record_seconds * self.fs)
        self.rec_pos = 0
        if self.record_len > 0:
            self.rec = {
                "err": np.zeros(self.record_len, dtype=np.float32),
                "ref": np.zeros(self.record_len, dtype=np.float32),
                "source": np.zeros(self.record_len, dtype=np.float32),
                "control": np.zeros(self.record_len, dtype=np.float32),
                "anc_gain": np.zeros(self.record_len, dtype=np.float32),
            }
        else:
            self.rec = None

        self._infer_thread: threading.Thread | None = None
        self._stream = None

    # ---------- 콜백 (PortAudio 스레드) ----------

    def _callback(self, indata, outdata, frames, _time_info, status) -> None:
        try:
            if status:
                self.xruns += 1

            mics = pcm_int32_to_float32(indata[:, :2])
            err = self.err_dc.process(mics[:, self.ch_err])
            ref_mic = self.ref_dc.process(mics[:, self.ch_ref])

            noise_gain = self.noise_gate.process(frames)
            self.noise_gate.set_target(1.0 if self.state.noise_enabled else 0.0)
            source = self.program.generate(frames) * noise_gain

            self.in_ring.push(np.stack([err, ref_mic, source]))

            y_blk, had_data = self.out_ring.pop_or_silence(frames)
            y_lim, clip_frac = self.safety.limit_output(y_blk[0])

            if self.state.anc_enabled != self._last_anc:
                self.anc_gate.set_target(1.0 if self.state.anc_enabled else 0.0)
                self._last_anc = self.state.anc_enabled
            gain = self.anc_gate.process(frames)
            control = y_lim * gain

            out = np.zeros((frames, 2), dtype=np.float32)
            out[:, self.ch_noise] = source
            out[:, self.ch_cancel] = control
            outdata[:] = float32_to_pcm_int16(out)

            err_power = self.err_meter.update(err)
            ctrl_power = self.ctrl_meter.update(control)

            # 베이스라인: ANC 게이트가 닫혀 있고 소음이 켜진 구간의 에러 파워
            if float(np.max(gain)) <= 0.001 and float(np.min(noise_gain)) >= 0.999:
                alpha = float(np.exp(-frames / (self.fs * 1.0)))
                if not self.baseline_init:
                    self.baseline_power = err_power
                    self.baseline_init = True
                else:
                    self.baseline_power = alpha * self.baseline_power + (1 - alpha) * err_power

            mute = self.safety.check_block(
                self.state.anc_enabled, clip_frac, err_power, self.baseline_power,
                had_data or not self.state.anc_enabled,
            )
            if mute:
                self.state.anc_enabled = False
            for msg in self.safety.drain_messages():
                self.state.messages.put(msg)

            if self.rec is not None and self.rec_pos < self.record_len:
                n = min(frames, self.record_len - self.rec_pos)
                sl = slice(self.rec_pos, self.rec_pos + n)
                self.rec["err"][sl] = err[:n]
                self.rec["ref"][sl] = ref_mic[:n]
                self.rec["source"][sl] = source[:n]
                self.rec["control"][sl] = control[:n]
                self.rec["anc_gain"][sl] = gain[:n]
                self.rec_pos += n

            reduction = float("nan")
            if self.baseline_init and err_power > 0:
                reduction = 10.0 * np.log10((self.baseline_power + 1e-30) / (err_power + 1e-30))
            self.state.latest_stats = {
                "anc": self.state.anc_enabled,
                "err_dbfs": power_to_db(err_power),
                "ctrl_dbfs": power_to_db(ctrl_power),
                "reduction_db": reduction,
                "underruns": self.out_ring.underruns,
                "xruns": self.xruns,
                "step_ms": float(np.mean(self.step_times_ms[-50:])) if self.step_times_ms else 0.0,
            }
        except BaseException as exc:      # 콜백 예외 → 안전 정지
            outdata.fill(0)
            self.state.fatal_error = exc
            self.state.quit_event.set()
            raise self.sd.CallbackAbort from exc

    # ---------- 추론 스레드 ----------

    def _inference_loop(self) -> None:
        affinity = self.cfg.get("engine", {}).get("cpu_affinity")
        if affinity:
            try:
                os.sched_setaffinity(0, set(int(c) for c in affinity))
            except OSError:
                pass
        while not self.state.quit_event.is_set():
            if self.state.reset_event.is_set():
                self.engine.reset()
                self.in_ring.reset()
                self.out_ring.reset()
                self.state.reset_event.clear()
            if not self.in_ring.wait_for(self.hop, timeout=0.1):
                continue
            blk = self.in_ring.pop(self.hop)
            if blk is None:
                continue
            err, ref_mic, ref_digital = blk
            ref = ref_digital if self.reference == "digital" else ref_mic
            t0 = time.perf_counter()
            try:
                y = self.engine.step(ref.copy(), err.copy())
            except Exception as exc:
                self.state.messages.put(f"엔진 오류: {exc!r} — 무음 출력")
                y = np.zeros(self.hop, dtype=np.float32)
            dt = (time.perf_counter() - t0) * 1000.0
            self.step_times_ms.append(dt)
            if len(self.step_times_ms) > 10000:
                del self.step_times_ms[:5000]
            self.out_ring.push(y.reshape(1, -1))

    # ---------- 실행 ----------

    def start(self) -> None:
        self._infer_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="anc-inference"
        )
        self._infer_thread.start()
        self._stream = self.sd.Stream(
            samplerate=self.fs,
            blocksize=self.block,
            device=(self.in_dev, self.out_dev),
            channels=(2, 2),
            dtype=("int32", "int16"),
            latency=("low", "low"),
            callback=self._callback,
            prime_output_buffers_using_stream_callback=True,
        )
        self._stream.start()

    def stop(self) -> None:
        # 종료 페이드 시퀀스 (안전장치 8)
        self.state.anc_enabled = False
        self.state.noise_enabled = False
        time.sleep(0.2)
        self.state.quit_event.set()
        if self._stream is not None:
            try:
                self._stream.abort()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
        if self._infer_thread is not None:
            self._infer_thread.join(timeout=1.0)

    def session_data(self) -> dict[str, np.ndarray]:
        if self.rec is None:
            return {}
        n = self.rec_pos
        return {k: v[:n].copy() for k, v in self.rec.items()}


def run_cli(cfg: dict, run_seconds: float, record_path: str | None) -> int:
    anc = RealtimeANC(cfg, record_seconds=run_seconds if record_path else 0.0)
    keyboard = KeyboardController(anc.state)

    engine_desc = cfg.get("controller", "dl")
    if engine_desc == "dl":
        engine_desc = f"dl/{cfg.get('engine', {}).get('type', 'torch')}"
    print("=" * 72)
    print(f"Deep ANC 실시간 런타임 | 컨트롤러: {engine_desc} | reference: {cfg.get('reference')}")
    print(f"블록 {anc.block} ({1000*anc.block/anc.fs:.2f}ms) @ {anc.fs}Hz | 시작: ANC OFF")
    print(KeyboardController.help_text())
    print("주의: TPA3116D2 볼륨을 낮춘 상태에서 시작하세요.")
    print("=" * 72)

    anc.start()
    keyboard.start()
    started = time.monotonic()
    next_report = started
    try:
        while not anc.state.quit_event.is_set():
            now = time.monotonic()
            if run_seconds > 0 and now - started >= run_seconds:
                break
            while True:
                try:
                    print(f"\n[명령] {anc.state.messages.get_nowait()}")
                except Exception:
                    break
            if now >= next_report:
                s = anc.state.latest_stats
                if s:
                    red = s.get("reduction_db", float("nan"))
                    red_txt = "  n/a" if not np.isfinite(red) else f"{red:6.2f}"
                    print(
                        f"[{'ON ' if s['anc'] else 'OFF'}] e={s['err_dbfs']:7.2f} dBFS | "
                        f"ctrl={s['ctrl_dbfs']:7.2f} | 저감={red_txt} dB | "
                        f"step={s['step_ms']:5.2f}ms | miss={s['underruns']} | xrun={s['xruns']}"
                    )
                next_report = now + 1.0
            if anc.state.fatal_error is not None:
                raise RuntimeError("오디오 콜백 실패") from anc.state.fatal_error
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        keyboard.stop()
        anc.stop()

    if record_path:
        data = anc.session_data()
        if data:
            out = Path(record_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out.with_suffix(".npz"), fs=anc.fs, **data)
            print(f"세션 저장: {out.with_suffix('.npz')}")
    print("종료 — 양 채널 무음.")
    return 0


def run_calibrate(cfg: dict) -> int:
    """--calibrate: 3-스레드 경로 그대로의 실효 상쇄경로 지연 실측 [C1].

    추론 엔진 자리에 '처프 재생 엔진'을 넣어 out_ring→콜백→스피커→에러마이크
    왕복 지연을 상호상관으로 측정하고, 학습에 쓰는 지연(캘리브레이션+핸드오프)과
    비교해 어긋남을 리포트한다.
    """
    from scipy import signal as sp_signal

    from ..dsp.secondary_path import load_secondary_path

    fs = int(cfg["hardware"]["audio"]["sample_rate"])
    seconds = 6.0
    t = np.arange(int(seconds * fs)) / fs
    chirp = (
        0.05 * sp_signal.chirp(t, 100.0, seconds, 2000.0, method="logarithmic")
    ).astype(np.float32)
    fade = int(0.05 * fs)
    chirp[:fade] *= np.linspace(0, 1, fade)
    chirp[-fade:] *= np.linspace(1, 0, fade)

    class ChirpEngine:
        def __init__(self, hop: int) -> None:
            self.hop = hop
            self.pos = 0

        def reset(self) -> None:
            self.pos = 0

        def step(self, ref, err):
            out = np.zeros(self.hop, dtype=np.float32)
            n = min(self.hop, chirp.size - self.pos)
            if n > 0:
                out[:n] = chirp[self.pos : self.pos + n]
                self.pos += n
            return out

    cfg = dict(cfg)
    cfg["noise"] = {"type": "silence"}
    anc = RealtimeANC(cfg, record_seconds=seconds + 2.0)
    anc.engine = ChirpEngine(anc.hop)
    anc.state.anc_enabled = True          # 게이트를 열어 처프를 내보낸다
    anc.anc_gate.set_target(1.0)
    print(f"실효 지연 측정: 처프 {seconds:.0f}s 재생 (상쇄 스피커 ch1) ...")
    anc.start()
    time.sleep(seconds + 1.5)
    anc.stop()

    data = anc.session_data()
    err = data["err"].astype(np.float64)
    ctrl = data["control"].astype(np.float64)
    if np.max(np.abs(ctrl)) < 1e-6:
        print("[실패] 출력이 재생되지 않았습니다", file=sys.stderr)
        return 1
    corr = sp_signal.fftconvolve(err, ctrl[::-1], mode="full")
    lag = int(np.argmax(np.abs(corr))) - (ctrl.size - 1)
    sp = load_secondary_path(cfg["secondary_path"])
    handoff = int(cfg["duct"]["secondary_path"].get("handoff_extra_samples", 256))
    expected = sp.delay_samples + handoff
    print(f"측정 실효 지연 : {lag}샘플 ({1000*lag/fs:.2f}ms)")
    print(f"학습 가정 지연 : {expected}샘플 (캘리브레이션 {sp.delay_samples} + 핸드오프 {handoff})")
    print(f"차이           : {lag - expected:+d}샘플 ({1000*(lag-expected)/fs:+.2f}ms)")
    if abs(lag - expected) > 512:
        print(
            "→ 차이가 지터 증강 범위(+512)를 벗어납니다. duct.yaml 의 "
            "handoff_extra_samples 조정 또는 재캘리브레이션 후 파인튜닝을 권장합니다."
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/runtime.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--run-seconds", type=float, default=None)
    parser.add_argument("--record", default=None, help="세션 npz 저장 경로")
    parser.add_argument("--calibrate", action="store_true", help="실효 지연 측정 모드")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        print(format_sounddevice_devices())
        return 0

    cfg = load_runtime_config(args.config, args.overrides)
    if args.calibrate:
        return run_calibrate(cfg)
    run_seconds = args.run_seconds if args.run_seconds is not None else float(cfg.get("run_seconds", 0.0))
    record = args.record or cfg.get("record")
    return run_cli(cfg, run_seconds, record)


if __name__ == "__main__":
    raise SystemExit(main())
