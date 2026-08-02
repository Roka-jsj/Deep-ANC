"""학습 루프 — Stage-1 open-loop / Stage-2 closed-loop, 단일 GPU·DDP(torchrun) 겸용.

실행 (Elice A100):
  단일:  python scripts/train/train.py --config configs/train_pretrain.yaml
  2-GPU: torchrun --nproc_per_node=2 scripts/train/train.py --config configs/train_pretrain.yaml
MIG(1g-10GB) 디버깅은 batch_size 를 4 로 낮추면 동일 코드로 동작한다.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from ..config import DEFAULT_HANDOFF_SAMPLES, REPO_ROOT
from ..data.recorded_dataset import RecordedANCDataset
from ..data.synth_dataset import SynthANCDataset, make_eval_batch
from ..dsp.nonlinear import RandomNonlinear
from ..dsp.secondary_path import DifferentiableSecondaryPath, load_secondary_path
from ..losses import ANCLoss
from ..models import build_model
from .checkpoint import load_checkpoint, save_checkpoint
from .reproducibility import set_seed, snapshot_run


def _ddp_env() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank, world, local_rank


class MixedIterator:
    """합성/실측 데이터셋 혼합 (recorded_ratio 확률로 실측 배치 샘플)."""

    def __init__(self, synth_iter, recorded_iter, recorded_ratio: float, seed: int) -> None:
        import numpy as np

        self.synth = synth_iter
        self.recorded = recorded_iter
        self.ratio = float(recorded_ratio)
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        return self

    def __next__(self):
        if self.recorded is not None and self.rng.random() < self.ratio:
            return next(self.recorded)
        return next(self.synth)


class Trainer:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.rank, self.world, self.local_rank = _ddp_env()
        self.is_main = self.rank == 0
        if self.world > 1 and not dist.is_initialized():
            dist.init_process_group("nccl")
            torch.cuda.set_device(self.local_rank)
        self.device = torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        )

        seed = int(cfg.get("seed", 0)) + self.rank
        set_seed(seed)

        self.stage = str(cfg.get("stage", "open_loop"))
        if self.stage == "closed_loop" and self.world > 1:
            # 폐루프는 언랩된 모듈의 streaming_step 을 직접 호출하므로 DDP 그래디언트
            # 동기화가 동작하지 않는다 (리뷰 확정 결함 #3). 20~50k step 단기 학습이므로
            # 단일 GPU 로 실행할 것.
            raise RuntimeError(
                "closed_loop 스테이지는 단일 GPU 전용입니다 — torchrun 없이 실행하세요."
            )
        self.fs = int(cfg["data"]["sample_rate"])
        self.run_dir = Path(cfg["ckpt_dir"])
        if not self.run_dir.is_absolute():
            self.run_dir = REPO_ROOT / self.run_dir

        # ----- 모델 -----
        self.model = build_model(cfg["model"]).to(self.device)
        if self.world > 1:
            self.model = DistributedDataParallel(self.model, device_ids=[self.local_rank])

        # ----- 플랜트 + 손실 -----
        duct = cfg["duct"]
        sp = load_secondary_path(REPO_ROOT / duct["secondary_path"]["npz"])
        if sp.sample_rate != self.fs:
            raise ValueError(
                f"S(z) npz 샘플레이트 {sp.sample_rate} ≠ 데이터 {self.fs} — "
                "duct.yaml secondary_path.npz 를 확인하세요 (감사 M7)"
            )
        pp = cfg["data"].get("plant_perturbation", {})
        plant = DifferentiableSecondaryPath(
            sp,
            handoff_extra_samples=int(
                duct["secondary_path"].get("handoff_extra_samples", DEFAULT_HANDOFF_SAMPLES)
            ),
            delay_jitter_range=tuple(pp.get("delay_jitter_range", [0, 0])),
            gain_db_range=tuple(pp.get("gain_db", [0.0, 0.0])),
            tilt_db_per_octave_range=tuple(pp.get("gain_tilt_db_per_octave", [0.0, 0.0])),
            allpass_perturb=bool(pp.get("allpass_perturb", False)),
            seed=seed + 17,
        ).to(self.device)
        nl_cfg = cfg["data"].get("nonlinear", {})
        nonlinear = RandomNonlinear(
            nl_cfg.get("sef_eta_choices", [10.0]),
            tuple(nl_cfg.get("drive_range", [1.0, 1.0])),
            hardclip_prob=float(nl_cfg.get("hardclip_prob", 0.0)),
            seed=seed + 29,
        )
        acoustics = duct.get("acoustics", {})
        cutoff = float(acoustics.get("plane_wave_cutoff_hz", 1633.0))
        target_band = tuple(acoustics.get("realistic_target_band_hz", [80.0, 1000.0]))
        self.criterion = ANCLoss(
            plant, cfg["loss"], self.fs, nonlinear=nonlinear,
            cutoff_hz=cutoff, target_band_hz=target_band,
        ).to(self.device)

        # 헤드룸 정합: 손실 클립 마진 < 모델 소프트리미터 한계 (감사 L8)
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        clip_margin = float(cfg["loss"].get("clip_margin", 0.18))
        if clip_margin >= raw_model.limit:
            raise ValueError(
                f"loss.clip_margin({clip_margin}) 은 모델 limiter.limit({raw_model.limit}) "
                "보다 작아야 합니다"
            )

        # ----- 데이터 -----
        # 비-synthetic 태그의 manifest 존재를 명시 검사 — 조용한 합성 폴백으로
        # 학습 분포가 바뀌는 사고 방지 (감사 M1). 부재 태그는 배너로 알린다.
        if self.is_main:
            mix = cfg["data"].get("source_mix_ratio", {})
            mdir = Path(cfg["data"].get("noise_manifest_dir", "data/manifests"))
            if not mdir.is_absolute():
                mdir = REPO_ROOT / mdir
            missing = [
                t for t, r in mix.items()
                if t != "synthetic" and float(r) > 0 and not (mdir / f"{t}.jsonl").exists()
            ]
            if missing:
                total_missing = sum(float(mix[t]) for t in missing)
                print("=" * 70)
                print(f"[trainer 경고] manifest 부재 태그 {missing} (비율 합 {total_missing:.0%})")
                print("  → 해당 비율은 합성원으로 대체됩니다. 의도가 아니면 학습을 중단하고")
                print("    scripts/data/prepare_noise_pool.py 를 실행하세요.")
                print("=" * 70)

        synth_train = SynthANCDataset(cfg["data"], duct, split="train", seed=seed)
        loader = DataLoader(
            synth_train,
            batch_size=int(cfg["batch_size"]),
            num_workers=int(cfg.get("num_workers", 4)),
            prefetch_factor=int(cfg.get("prefetch_factor", 2)) if cfg.get("num_workers", 4) else None,
            pin_memory=self.device.type == "cuda",
            persistent_workers=bool(cfg.get("num_workers", 4)),
        )
        self.train_iter = iter(loader)

        recorded_manifest = cfg.get("recorded_manifest")
        if recorded_manifest and not Path(recorded_manifest).is_absolute():
            recorded_manifest = str(REPO_ROOT / recorded_manifest)
        if recorded_manifest and Path(recorded_manifest).exists():
            rec = RecordedANCDataset(recorded_manifest, cfg["data"], split="train", seed=seed + 5)
            rec_loader = DataLoader(
                rec, batch_size=int(cfg["batch_size"]), num_workers=2,
                pin_memory=self.device.type == "cuda",
            )
            self.train_iter = MixedIterator(
                self.train_iter, iter(rec_loader), float(cfg.get("recorded_ratio", 0.5)), seed
            )
        elif recorded_manifest and self.is_main:
            print(f"[trainer] recorded_manifest({recorded_manifest}) 없음 — 합성 데이터만 사용")

        val_ds = SynthANCDataset(cfg["data"], duct, split="val", seed=1234)
        self.val_batch = make_eval_batch(val_ds, n_items=min(16, int(cfg["batch_size"])))

        # ----- 옵티마이저/스케줄 -----
        opt_cfg = cfg["optimizer"]
        opt_name = str(opt_cfg.get("name", "adamw")).lower()
        if opt_name != "adamw":
            raise ValueError(f"지원하지 않는 optimizer.name: {opt_name} (adamw 만 구현됨)")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(opt_cfg["lr"]),
            weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )
        sch = cfg["schedule"]
        warmup = int(sch.get("warmup_steps", 0))
        total = int(sch["total_steps"])
        min_ratio = float(sch.get("min_lr", 1e-5)) / float(opt_cfg["lr"])

        def lr_lambda(step: int) -> float:
            if warmup > 0 and step < warmup:
                return (step + 1) / warmup
            progress = min(1.0, (step - warmup) / max(1, total - warmup))
            return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.total_steps = total
        self.grad_clip = float(cfg.get("grad_clip", 0.0))
        self.amp_dtype = torch.bfloat16 if cfg.get("amp") == "bf16" else None

        if bool(cfg.get("freeze_encoder", False)):
            raw = self.model.module if hasattr(self.model, "module") else self.model
            for p in raw.encoder.parameters():
                p.requires_grad_(False)

        # ----- 상태 -----
        self.step = 0
        self.best_metric = float("inf")
        self.writer = None
        if self.is_main:
            snapshot_run(self.run_dir, {k: v for k, v in cfg.items() if k not in ("model", "data", "duct")})
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(str(self.run_dir / "tb"))
            except ImportError:
                print("[trainer] tensorboard 미설치 — 파일 로그만 기록합니다")
        self.loss_log = open(self.run_dir / "loss_log.txt", "a", encoding="utf-8") if self.is_main else None

        # init_ckpt(파인튜닝) → resume(재개) 순서로 적용 (상대경로는 저장소 루트 기준)
        def _abs(p):
            return str(p) if p is None or Path(p).is_absolute() else str(REPO_ROOT / p)

        init_ckpt = _abs(cfg.get("init_ckpt"))
        if cfg.get("init_ckpt") and not Path(init_ckpt).exists() and self.is_main:
            print(f"[trainer] 경고: init_ckpt({init_ckpt})가 없어 무시합니다")
        if init_ckpt and Path(init_ckpt).exists():
            state = load_checkpoint(init_ckpt, self.model, restore_rng=False, map_location="cpu")
            if self.is_main:
                print(f"[trainer] init_ckpt 로드: {init_ckpt} (step {state.get('step')})")
        resume = _abs(cfg.get("resume"))
        if resume and Path(resume).exists():
            state = load_checkpoint(resume, self.model, self.optimizer, self.scheduler, map_location="cpu")
            self.step = int(state.get("step", 0))
            self.best_metric = float(state.get("best_metric", float("inf")))
            if self.is_main:
                print(f"[trainer] 재개: step {self.step}, best {self.best_metric:.3f}")

    # ---------- 스텝 ----------

    def _forward_loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        x = batch["x"].to(self.device, non_blocking=True)
        d = batch["d"].to(self.device, non_blocking=True)
        if self.stage == "closed_loop":
            return self._closed_loop_forward(x, d)
        y = self.model(x)
        return self.criterion(y, d)

    def _closed_loop_forward(self, x: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Stage-2: 프레임 순차 unroll — 시뮬레이션 e 를 err 입력으로 되먹임 [H1/H3].

        피드백 지연(fb_delay ≥ 512)이 그룹 크기(hop×unroll_group)보다 크므로
        각 그룹 시작 시점까지 계산된 e 프리픽스만으로 인과적 되먹임이 가능하다.
        """
        cl = self.cfg["data"].get("closed_loop", {})
        group = int(cl.get("unroll_group_frames", 4))
        fb_lo, fb_hi = (int(v) for v in cl.get("feedback_delay_samples", [512, 1024]))
        warmup_s = float(cl.get("warmup_seconds", 0.25))

        raw = self.model.module if hasattr(self.model, "module") else self.model
        hop = raw.hop
        chunk = hop * group
        T = x.shape[-1] - (x.shape[-1] % chunk)
        x, d = x[..., :T], d[..., :T]

        import numpy as np

        fb_delay = int(np.random.default_rng(self.step).integers(fb_lo, fb_hi + 1))
        fb_delay = max(fb_delay, chunk)            # 인과성 보장

        states = raw.init_states(x.shape[0], self.device)
        y_parts: list[torch.Tensor] = []
        e_hist = torch.zeros_like(d)

        # 되먹임 경로와 최종 손실이 동일한 플랜트 섭동·비선형을 쓰도록 한 번만 샘플링
        # (리뷰 확정 결함 #6 — 되먹임만 공칭 선형이면 배포 분포와 어긋난다)
        plant = self.criterion.plant
        perturb = plant.sample_perturbation() if self.criterion.training else {"jitter": 0}
        nl = self.criterion.nonlinear
        nl_params = nl.sample(x.shape[0]) if (self.criterion.training and nl is not None) else None

        for start in range(0, T, chunk):
            sl = slice(start, start + chunk)
            err_in = e_hist[..., max(0, start - fb_delay) : max(0, start - fb_delay) + chunk]
            if err_in.shape[-1] < chunk:
                err_in = torch.nn.functional.pad(err_in, (chunk - err_in.shape[-1], 0))
            x_blk = torch.cat([x[:, :1, sl], err_in], dim=1)
            y_blk, states = raw.streaming_step(x_blk, states)
            y_parts.append(y_blk.float())          # 플랜트 FFT 는 FP32 필요 (bf16 미지원)
            # e 프리픽스 갱신 — 프리픽스 전체 재컨볼브 O(T²/chunk)는 알려진 성능 한계
            # (fb_delay ≥ chunk 라 인과성은 보장). 최적화 시 스트리밍 FIR 상태로 대체 가능.
            y_so_far = torch.cat(y_parts, dim=-1)
            y_nl = nl.apply_torch(y_so_far, nl_params) if nl_params is not None else y_so_far
            s_y = plant(y_nl, perturb)
            e_hist[..., : y_so_far.shape[-1]] = d[..., : y_so_far.shape[-1]] + s_y

        y = torch.cat(y_parts, dim=-1)
        skip = int(warmup_s * self.fs)
        # 절단은 손실 내부에서 플랜트 적용 "후"에 수행 (결함 #2/#5)
        return self.criterion(y, d, loss_start_sample=skip, perturb=perturb, nl_params=nl_params)

    def _validate(self) -> float:
        self.model.eval()
        self.criterion.eval()
        with torch.no_grad():
            x = self.val_batch["x"].to(self.device)
            d = self.val_batch["d"].to(self.device)
            raw = self.model.module if hasattr(self.model, "module") else self.model
            y = raw(x)
            _, metrics = self.criterion(y, d)
        self.model.train()
        self.criterion.train()
        return metrics["nmse_db"]

    # ---------- 메인 루프 ----------

    def train(self) -> None:
        cfg = self.cfg
        eval_every = int(cfg.get("eval_every", 2000))
        log_every = int(cfg.get("log_every", 100))
        patience = int(cfg.get("early_stop_patience", 0))
        bad_evals = 0
        self.model.train()
        self.criterion.train()
        t0 = time.time()

        while self.step < self.total_steps:
            batch = next(self.train_iter)
            self.optimizer.zero_grad(set_to_none=True)

            if self.amp_dtype is not None and self.device.type == "cuda":
                with torch.autocast("cuda", dtype=self.amp_dtype):
                    loss, metrics = self._forward_loss(batch)
            else:
                loss, metrics = self._forward_loss(batch)

            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.step += 1

            if self.is_main and self.step % log_every == 0:
                lr = self.scheduler.get_last_lr()[0]
                sps = log_every / max(1e-9, time.time() - t0)
                t0 = time.time()
                line = (
                    f"step {self.step:7d} | loss {metrics['loss']:8.3f} | "
                    f"nmse {metrics['nmse_db']:7.2f} dB | lr {lr:.2e} | {sps:5.2f} it/s"
                )
                print(line, flush=True)
                if self.loss_log:
                    self.loss_log.write(line + "\n")
                    self.loss_log.flush()
                if self.writer:
                    for k, v in metrics.items():
                        self.writer.add_scalar(f"train/{k}", v, self.step)
                    self.writer.add_scalar("train/lr", lr, self.step)

            if self.step % eval_every == 0:
                val_nmse = self._validate()
                stop_flag = torch.zeros(1, device=self.device)
                if self.is_main:
                    print(f"[eval] step {self.step}: val NMSE {val_nmse:.2f} dB", flush=True)
                    if self.writer:
                        self.writer.add_scalar("val/nmse_db", val_nmse, self.step)
                    save_checkpoint(
                        self.run_dir / "ckpt" / "last.pt",
                        self.model, self.optimizer, self.scheduler,
                        self.step, self.best_metric, cfg_snapshot(cfg),
                    )
                    if val_nmse < self.best_metric:
                        self.best_metric = val_nmse
                        bad_evals = 0
                        save_checkpoint(
                            self.run_dir / "ckpt" / "best.pt",
                            self.model, self.optimizer, self.scheduler,
                            self.step, self.best_metric, cfg_snapshot(cfg),
                        )
                        print(f"[eval] best 갱신 → {val_nmse:.2f} dB", flush=True)
                    else:
                        bad_evals += 1
                        if patience and bad_evals >= patience:
                            print(f"[eval] {patience}회 연속 미개선 — 조기 종료", flush=True)
                            stop_flag.fill_(1.0)
                # 조기종료 결정을 전 랭크에 전파 — rank0 만 break 하면 나머지가 행업된다 (#7)
                if self.world > 1:
                    dist.broadcast(stop_flag, src=0)
                if float(stop_flag.item()) > 0:
                    break

        if self.is_main:
            save_checkpoint(
                self.run_dir / "ckpt" / "last.pt",
                self.model, self.optimizer, self.scheduler,
                self.step, self.best_metric, cfg_snapshot(cfg),
            )
            print(f"학습 종료: step {self.step}, best val NMSE {self.best_metric:.2f} dB")
        if self.world > 1:
            dist.destroy_process_group()


def cfg_snapshot(cfg: dict) -> dict:
    """체크포인트에 저장할 설정 (모델 재구성에 필요한 부분 포함)."""
    return {
        "model": cfg["model"],
        "stage": cfg.get("stage"),
        "loss": cfg.get("loss"),
        "sample_rate": cfg["data"]["sample_rate"],
    }
