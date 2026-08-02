"""온더플라이 합성 학습 데이터셋.

신호 모델 (Deep ANC 방식):
    x_ref(t) : 모델 입력 ch0 (레퍼런스)
    d(t)     : 에러 마이크 위치의 1차경로 소음 (타깃 아님 — 손실에서 e=d+S·y)
    err_in   : 모델 입력 ch1 (에러 피드백 근사; 캡처+블록 지연 랜덤화 [H3])

레퍼런스 모드별 지연 물리 [설계 교차검증 C2]:
  digital  : x_ref = n(t) (Jetson 이 소음을 직접 생성).
             d = P_err · n(t − D_noise),  D_noise = 실측(권장) 또는
             기하 추정 s_delay − t_ac(CS→ERR) + t_ac(NS→ERR).
             출력버퍼 지연이 소음·상쇄 경로에 공통 → 광대역 상쇄가 인과적으로 가능.
  acoustic : x_ref = P_ref · n(t), d = P_err · n(t).
             S(z) 실측 지연(≈28ms)이 그대로 예측 부담이 됨 → 주기성/협대역 한정.

S(z) 플랜트와 핸드오프(+256)는 손실 모듈(anc_loss)에서 적용된다 — 여기서는
d 경로만 만든다 (소음 ch0 은 콜백에서 직접 생성되므로 핸드오프가 없다 [C1]).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ..config import default_d_noise_delay, duct_distance_samples
from ..dsp.duct_sim import build_rir_bank
from ..dsp.filters import fft_filter
from ..dsp.secondary_path import load_secondary_path
from .noise_pool import NoisePool
from .synthetic_signals import SyntheticNoise


def _delay_np(x: np.ndarray, delay: int) -> np.ndarray:
    if delay <= 0:
        return x.copy()
    out = np.zeros_like(x)
    out[delay:] = x[: x.size - delay]
    return out


class SynthANCDataset(IterableDataset):
    """무한 IterableDataset — 매 아이템 (소음원, RIR 변형, 레벨, 지연)을 랜덤 추첨."""

    def __init__(
        self,
        data_cfg: dict,
        duct_cfg: dict,
        split: str = "train",
        seed: int = 20260802,
        rir_bank: dict[str, np.ndarray] | None = None,
    ) -> None:
        super().__init__()
        self.data_cfg = data_cfg
        self.duct_cfg = duct_cfg
        self.split = split
        self.seed = int(seed)
        self.fs = int(data_cfg["sample_rate"])
        # 세그먼트를 런타임 블록(256 = 모델 hop 128×2)의 배수로 내림 — 모델 입력 요건
        raw_segment = int(round(float(data_cfg["segment_seconds"]) * self.fs))
        self.segment = max(256, (raw_segment // 256) * 256)
        self.reference_mode = str(data_cfg.get("reference_mode", "digital"))
        if self.reference_mode not in ("digital", "acoustic"):
            raise ValueError(f"reference_mode: {self.reference_mode}")

        # RIR 뱅크: 파일이 있으면 로드, 없으면 소규모 즉석 생성 (스모크 테스트용)
        if rir_bank is not None:
            self.rirs = rir_bank
        else:
            bank_path = data_cfg.get("rir_bank")
            try:
                with np.load(bank_path) as z:
                    self.rirs = {k: z[k] for k in ("p_ref", "p_err", "f_fb")}
            except (TypeError, FileNotFoundError, OSError):
                print(
                    f"[synth_dataset] RIR 뱅크({bank_path})가 없어 즉석 생성(32개)합니다 — "
                    "본 학습 전에는 scripts/data/build_rir_bank.py 를 실행하세요."
                )
                self.rirs = build_rir_bank(duct_cfg, self.fs, n_variants=32, seed=self.seed)

        n_var = self.rirs["p_err"].shape[0]
        # RIR 변형도 split 단위 분할 (누수 방지)
        idx = np.arange(n_var)
        rng = np.random.default_rng(20260801)
        rng.shuffle(idx)
        n_val = max(1, int(n_var * 0.05))
        n_test = max(1, int(n_var * 0.05))
        self.rir_indices = {
            "val": idx[:n_val],
            "test": idx[n_val : n_val + n_test],
            "train": idx[n_val + n_test :],
        }[split]

        # 노이즈 풀 (manifest 가 없으면 합성원 100%로 폴백)
        self.mix_ratio = dict(data_cfg.get("source_mix_ratio", {"synthetic": 1.0}))
        manifest_dir = data_cfg.get("noise_manifest_dir")
        self.pools: dict[str, list] = {}
        for tag in ("dns_fullband", "esc50"):
            if self.mix_ratio.get(tag, 0.0) <= 0.0:
                continue
            manifest = f"{manifest_dir}/{tag}.jsonl"
            try:
                self.pools[tag] = [manifest]
            except Exception:
                pass
        self._pool_objs: dict[str, NoisePool] = {}

        # digital-ref 1차경로 순수지연.
        # 규약: d_noise_delay_samples(실측/기본값)는 "디지털 출력→에러마이크 총 순수지연"이다.
        # p_err RIR(영상법)에는 음향 전파 온셋 t_ac(NS→ERR)가 이미 포함되어 있으므로,
        # RIR 과 결합할 때는 전기/버퍼 성분만 추가한다 — 이중 계상 방지 (리뷰 확정 결함 #1).
        sp = load_secondary_path(duct_cfg["secondary_path"]["npz"])
        d_noise = duct_cfg.get("digital_reference", {}).get("d_noise_delay_samples")
        if d_noise is None:
            d_noise = default_d_noise_delay(duct_cfg, self.fs, sp.delay_samples)
        self.d_noise_total = int(d_noise)
        t_ns_err = duct_distance_samples(duct_cfg, "noise_speaker", "error_mic", self.fs)
        self.d_noise_delay = max(0, self.d_noise_total - t_ns_err)

        self.level_range = tuple(data_cfg.get("level_dbfs", [-35, -10]))
        self.snr_range = tuple(data_cfg.get("snr_mic_noise_db", [5, 30]))
        fb = data_cfg.get("closed_loop", {}).get("feedback_delay_samples", [512, 1024])
        self.feedback_delay_range = (int(fb[0]), int(fb[1]))

    # ---------- 내부 ----------

    def _pool(self, tag: str, rng: np.random.Generator) -> NoisePool | None:
        if tag not in self.pools:
            return None
        if tag not in self._pool_objs:
            try:
                self._pool_objs[tag] = NoisePool(
                    self.pools[tag], self.split, self.fs, seed=int(rng.integers(1 << 31))
                )
            except (FileNotFoundError, ValueError):
                print(f"[synth_dataset] {tag} manifest 없음 — 합성원으로 대체합니다")
                self.pools.pop(tag)
                return None
        return self._pool_objs[tag]

    def _sample_source(self, rng: np.random.Generator, synth: SyntheticNoise) -> np.ndarray:
        tags = list(self.mix_ratio.keys())
        probs = np.array([self.mix_ratio[t] for t in tags], dtype=np.float64)
        probs = probs / probs.sum()
        tag = str(rng.choice(tags, p=probs))
        if tag != "synthetic":
            pool = self._pool(tag, rng)
            if pool is not None:
                seg = pool.sample_segment(self.segment)
                rms = float(np.sqrt(np.mean(seg**2)) + 1e-9)
                return seg / rms
        return synth.generate(self.segment)

    def _make_item(self, rng: np.random.Generator, synth: SyntheticNoise) -> dict:
        n = self._sample_source(rng, synth)

        # 레벨 랜덤화
        level_db = float(rng.uniform(*self.level_range))
        n = n * (10.0 ** (level_db / 20.0))

        # RIR 변형 추첨
        ridx = int(rng.choice(self.rir_indices))
        p_ref = self.rirs["p_ref"][ridx]
        p_err = self.rirs["p_err"][ridx]

        if self.reference_mode == "digital":
            x_ref = n.copy()
            d = _delay_np(fft_filter(n, p_err), self.d_noise_delay)
        else:
            x_ref = fft_filter(n, p_ref)
            d = fft_filter(n, p_err)

        # 에러 피드백 입력 (open-loop 근사: d 를 캡처+블록 지연 후 공급) [H3]
        fb_delay = int(rng.integers(*self.feedback_delay_range))
        err_in = _delay_np(d, fb_delay)

        # 마이크 자기잡음
        snr_db = float(rng.uniform(*self.snr_range))
        for sig in (x_ref, err_in):
            p_sig = float(np.mean(sig**2) + 1e-12)
            p_noise = p_sig / (10.0 ** (snr_db / 10.0))
            sig += rng.standard_normal(sig.size).astype(np.float32) * np.sqrt(p_noise)

        # 채널 dropout — ref-only / err-only 운용 대비 (동시 제거는 금지)
        u = rng.random()
        if u < 0.15:
            err_in = np.zeros_like(err_in)
        elif u < 0.30:
            x_ref = np.zeros_like(x_ref)

        x = np.stack([x_ref, err_in]).astype(np.float32)   # [2, T]
        return {
            "x": torch.from_numpy(x),
            "d": torch.from_numpy(d.astype(np.float32)).unsqueeze(0),  # [1, T]
        }

    # ---------- IterableDataset ----------

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        split_offset = {"train": 0, "val": 7919, "test": 15859}[self.split]
        rng = np.random.default_rng(self.seed + split_offset + worker_id * 1009)
        synth = SyntheticNoise(self.fs, seed=int(rng.integers(1 << 31)))
        while True:
            yield self._make_item(rng, synth)


def make_eval_batch(
    dataset: SynthANCDataset, n_items: int, seed: int = 12345
) -> dict[str, torch.Tensor]:
    """고정 시드 검증 배치 — 학습 중 val NMSE 추적용 (매번 동일 데이터)."""
    rng = np.random.default_rng(seed)
    synth = SyntheticNoise(dataset.fs, seed=seed)
    items = [dataset._make_item(rng, synth) for _ in range(n_items)]
    return {
        "x": torch.stack([it["x"] for it in items]),
        "d": torch.stack([it["d"] for it in items]),
    }
