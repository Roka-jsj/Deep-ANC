# 10. 저장소 구조 전체 지도 (Structure Map)

> 근거: 2026-08-03 3종 감사(데이터흐름·문서 사실성·운영/HANDOFF) 통합 + 코드 직접 재검증.
> 이 문서는 "어떤 설정 키를 어느 코드가 소비하는가"의 단일 참조점이다.

## 1. 전체 데이터흐름

### 1.1 학습 (Elice, open-loop / Stage-1)

```
configs/train_*.yaml ──load_train_config(config.py:68)──▶ cfg{model,data,duct 병합}
                                                             │
소음원 ──────────────────────────────────────────────────────┤
  SyntheticNoise(synthetic_signals.py, 덕트공진 가중)   30%    │
  NoisePool(noise_pool.py ← data/manifests/<tag>.jsonl)      │  dns_fullband 30% + speech 15%
       └ 48kHz 리샘플(resample_poly), RMS=1 정규화            │  + music 15%(폴백) + esc50 10%
                                                             ▼
  SynthANCDataset(data/synth_dataset.py)              n(t) [T=1.5s→71,936샘플(256 배수 내림)]
  ├ RIR 뱅크: data/rir_bank/duct_rirs_v1.npz (build_rir_bank.py 300변형; 분할 5/5/90% 하드코딩)
  │   duct.yaml positions_m/reflection/duct.* ──dsp/duct_sim.py 영상법──▶ p_ref / p_err / f_fb
  ├ digital-ref: x_ref = n,  d = delay(p_err*n, D_noise − t_ac(NS→ERR))   [이중계상 방지]
  │   D_noise = duct.digital_reference.d_noise_delay_samples(null → 기하추정 1489 = 1342−7+154)
  ├ err_in = delay(d, fb∈[512,1024]) + 마이크잡음(snr_mic_noise_db) + 채널드롭아웃(0.15/0.15 하드코딩)
  ▼
x=[B,2,T] ──HybridANCNet(models/hybrid_anc.py: /io_scale(0.02 묵시) → enc(win384,hop128)
            → TCN/GLSTM/MHSA → dec → ×io_scale → 0.2·tanh 소프트리미터)──▶ y=[B,1,T]
  ▼
ANCLoss(losses/anc_loss.py, FP32 강제)
  y → RandomNonlinear(dsp/nonlinear.py: drive[1,4] → SEF η{0.1,0.5,1,10} → hardclip 5%)
    → DifferentiableSecondaryPath(dsp/secondary_path.py: 지연 1342+핸드오프256+지터[0,512]
       → FIR2048 + 게인/틸트/올패스 섭동)
  e = d + S(G(y)) → NMSE(dB) + λ·MRSTFT×W(f)[curriculum_a: 80–1000Hz ×3, >1633Hz ×0.25]
                    + λ_pow·|y|² + λ_clip·relu(|y|−0.18)²
  ▼
Trainer(train/trainer.py): AdamW+cosine, bf16 autocast(손실 FP32), DDP, val NMSE → best.pt

[Stage-2 closed-loop] 단일 GPU 전용(DDP 금지, trainer.py:75). chunk = hop×unroll_group = 512샘플
  순차 unroll, e 프리픽스를 fb_delay(≥chunk) 지연 후 err 채널로 되먹임.
  손실 절단(warmup 0.25s)은 플랜트 적용 후(anc_loss.py loss_start_sample).
```

### 1.2 배포 (Jetson)

```
best.pt ──export_onnx.py(블록256, 상태 명시 I/O, ORT 등가 검증)──▶ model.onnx + model.json
        ──scripts/export/build_trt.sh──▶ model_fp16.plan (+메타 json 복사)
runtime.yaml ──load_runtime_config(config.py:86)──▶ RealtimeANC(realtime/run_realtime.py, 3-스레드)
  [콜백]  int32 입력 → DCBlocker → in_ring / NoiseProgram(noise.*) → ch0
          out_ring.pop_latest → SafetySupervisor.limit(0.2) → FadeGate → ch1(int16)
  [추론]  ref(digital=소스|mic) + err → engine.step(hop=256, ==block_size 강제) → out_ring
          (1 hop 핸드오프 = 학습 handoff_extra_samples=256 과 정합 [C1])
  [엔진]  engines.py: torch(ckpt 내장 cfg 재구성)|ort|trt|fxlms(runtime.secondary_path+fxlms.*)
  [안전]  safety.*: 클립스트릭/발산/데드라인 워치독 → 자동 mute
```

### 1.3 실측·보정 루프 / 평가

```
record_duct.py(hardware.yaml) → data/recorded/<세션> → make_recorded_manifest.py
  → RecordedANCDataset(d=err mic, x_ref=source.wav) → MixedIterator(recorded_ratio)
calibrate_wideband.py(ESS) → --output-channel cancel: S(z) 재보정 npz
                           → --output-channel noise:  D_noise 실측 → duct.yaml 수기 기입
run_realtime --calibrate: 3-스레드 실효지연 측정 vs (1342+256) 대조 (run_realtime.py:391)

evaluate_offline.py(test split, 무섭동 S) / compare_fxlms.py(동일 S(z))
/ evaluate_session.py(실기 OFF→ON→OFF, eval.yaml scenarios/protocol)
```

## 2. 설정 소비 지도

범례: ✅=코드 소비, ☠=죽은 키(어떤 코드도 읽지 않음 — grep 재확인 완료), ⚠=주의.

### duct.yaml
| 키 | 상태 | 소비 지점 |
|---|---|---|
| duct.interior_length_m / cross_section_m / end_correction_factor / speed_of_sound_mps | ✅ | dsp/duct_sim.py, config.duct_distance_samples |
| duct.shape / wall_thickness_m / total_length_m / boundary, positions_m.opening, acoustics.axial_resonances_hz, acoustics.realistic_target_band_hz | ☠ | 문서용(테스트도 하드코딩값 사용) |
| positions_m.{noise_speaker,reference_mic,cancel_speaker,error_mic} | ✅ | duct_sim.duct_paths, duct_distance_samples, validate_duct |
| acoustics.plane_wave_cutoff_hz | ✅ | trainer.py:112 → ANCLoss cutoff |
| reflection.* | ✅ | duct_sim(단, build_rir_bank 는 자체 랜덤값으로 대체) |
| secondary_path.npz | ✅ | trainer.py:94, synth_dataset.py:110, evaluate_offline.py:52, compare_fxlms.py:65 |
| secondary_path.handoff_extra_samples | ⚠ | trainer.py:98(기본0)·evaluate_offline:54(0)·compare_fxlms:66(0) vs run_realtime.py:392(기본**256**) — 기본값 불일치 |
| secondary_path.delay_jitter_range | ☠ | 실소비는 data_sim.plant_perturbation 쪽(trainer.py:99)뿐 |
| digital_reference.d_noise_delay_samples | ✅ | synth_dataset.py:111, compare_fxlms.py:44, validate_duct |
| calibration.input_ref_rms | ☠ | 모델 io_scale(hybrid_anc.py:42 묵시 기본 0.02)과 미연결 |

### data_sim.yaml
| 키 | 상태 | 소비 지점 |
|---|---|---|
| sample_rate / segment_seconds / reference_mode | ✅ | synth·recorded_dataset, trainer.fs (세그먼트는 256 배수 내림) |
| source_mix_ratio.* | ⚠ | synth_dataset.py:97 — 키=manifest 태그, **없으면 조용히 synthetic 폴백**(:134) |
| noise_manifest_dir / rir_bank | ⚠ | synth_dataset.py:98/70 — **CWD 기준 상대경로**, RIR 부재 시 32개 즉석 폴백 |
| level_dbfs / snr_mic_noise_db | ✅ | synth_dataset.py:118-119 |
| dc_hum_prob | ☠ | 미구현 |
| nonlinear.* / plant_perturbation.* | ✅ | trainer.py:105-110 / :95-102 |
| closed_loop.{feedback_delay_samples,warmup_seconds,unroll_group_frames} | ✅ | trainer.py:225-228, synth/recorded_dataset |
| closed_loop.chunk_seconds | ☠ | 실제 chunk = hop×unroll_group=512샘플(trainer.py:232) |
| split.* | ☠ | RIR 5/5% 하드코딩(synth_dataset:86-87), manifest 분할은 prepare_noise_pool.py:46 {0.9,0.05}·make_recorded_manifest.py:45 {0.8,0.1} 하드코딩 |

### model_base/tiny.yaml
hop/win/in_channels/encoder/tcn/glstm/attention/limiter.limit → models/hybrid_anc.py:32-92 ✅.
name → export 메타. **sample_rate ☠**(models/·export_onnx 모두 미소비). **io_scale 은 yaml에 없는 묵시 키**(코드 기본 0.02).

### train_pretrain/finetune.yaml
model/data/duct_config → config.py:77-79 ✅. batch_size/num_workers/prefetch → trainer:119-126 ✅.
optimizer.{lr,weight_decay,betas} → trainer:149 ✅ (**name ☠** — AdamW 하드코딩).
schedule/amp/grad_clip/loss.*/eval_every/early_stop/ckpt_dir/resume/seed ✅.
init_ckpt/recorded_manifest/recorded_ratio/freeze_encoder(파인튜닝) → trainer:129-201 ✅.

### runtime.yaml
hop(==block_size 강제, run_realtime:58-60), reference, controller, engine.{type,ckpt,onnx,plan,cpu_affinity},
fxlms.*, safety.*, noise.*, record/run_seconds/start_on ✅.
**engine.model_config ☠**(TorchEngine 은 ckpt 내장 state["cfg"]["model"] 사용, engines.py:31).
**secondary_path ⚠** — duct.yaml 의 npz 와 이중 정의(engines.py:227 + run_realtime.py:391 이 runtime 쪽 소비).

### hardware_jetson.yaml
sample_rate/block_size, input/output.card+pcm, channels.*, dc_blocker_r ✅.
**audio.latency, input/output.{channels,dtype} ☠** — run_realtime.py:239-241·record_duct.py:127-129 등이
("int32","int16"), (2,2), ("low","low") 하드코딩.

### eval.yaml
octave_bands_hz/trusted_band_hz/scenarios/protocol ✅. **report_dir ☠**(evaluate_session.py:91,97 이 REPO_ROOT/results 하드코딩).

## 3. 모듈 의존 관계 (src/deep_anc)

```
config.py (REPO_ROOT, load/merge/validate)  ←─ 모든 스크립트의 진입 관문
   ▲
data/    synth_dataset ─▶ synthetic_signals, noise_pool(─▶manifest), dsp/{duct_sim,filters,secondary_path}, config
         recorded_dataset ─▶ manifest
dsp/     duct_sim, filters, nonlinear, secondary_path (독립적 하위층)
models/  hybrid_anc ─▶ tcn_blocks, glstm, attention, streaming
losses/  anc_loss ─▶ dsp/{nonlinear,secondary_path}
train/   trainer ─▶ data/*, dsp/*, losses, models, checkpoint, reproducibility, config
realtime/ run_realtime ─▶ engines(─▶models, baselines/fxlms_core), ring_buffer, safety, noise_gen, ui, audio_io, config
eval/    metrics, plots, fxlms_baseline  ←─ scripts/eval·demo 가 사용
baselines/ fxlms_core (레거시 FxLMS — anc_project 유산)
```
상향 의존 없음(dsp/data → train → scripts 단방향). 경로 해석은 config._resolve_path(REPO_ROOT 폴백)가 원칙이나
synth_dataset·noise_pool 은 원시 상대경로 사용(§5 이슈 M2).

## 4. 실행 경로별 진입점

| 경로 | 진입점 | 소비 설정 |
|---|---|---|
| 학습(사전) | `scripts/train/train.py --config configs/train_pretrain.yaml` (DDP: torchrun) | train_pretrain → model/data/duct 병합 |
| 학습(원샷, Elice) | `scripts/elice/bootstrap_all.sh` → `run_parallel_models.sh`(GPU0=base, GPU1=tiny, `mkdir -p runs` 포함) | 〃 |
| 파인튜닝 | `train.py --config configs/train_finetune.yaml` (+recorded_manifest) | train_finetune |
| 데이터 준비 | `scripts/data/{download_noise.sh, prepare_noise_pool.py, build_rir_bank.py}` | data_sim, duct |
| 실측/보정 | `scripts/data/{record_duct.py, make_recorded_manifest.py, calibrate_wideband.py}` | hardware, duct |
| 내보내기 | `scripts/train/export_onnx.py` → `scripts/export/build_trt.sh` | (ckpt 내장 cfg) |
| 실시간 | `python -m deep_anc.realtime.run_realtime --config configs/runtime.yaml` (--calibrate/--list-devices) | runtime → hardware/duct 병합 |
| 벤치 | `scripts/bench/measure_{inference,io}_latency.py` | runtime, hardware |
| 평가 | `scripts/eval/{evaluate_offline,compare_fxlms}.py --ckpt …`, `scripts/demo/evaluate_session.py` | eval, duct, data_sim |

---

## 부록: 감사 이슈 반영 상태 (2026-08-03)

위 지도를 만든 3종 감사에서 확정된 이슈 35건(HIGH 2/MED 12/LOW 21)은 같은 날 커밋에서
일괄 반영되었다 — 죽은 키 정리/주석화, S(z)·핸드오프·목표대역 단일 출처화, CWD 상대경로 제거,
manifest 부재 배너, dc_hum 구현, HANDOFF 재작성 등. 상세 내역은 해당 커밋 메시지 참조.
이 문서의 "☠ 죽은 키" 표기 중 일부는 반영 후 해소되었으므로, 설정을 바꿀 때는 이 지도로
소비 지점을 찾은 뒤 코드로 재확인할 것.

