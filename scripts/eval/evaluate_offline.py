#!/usr/bin/env python3
"""오프라인 평가 — 테스트 split 합성 데이터에서 모델 성능 일괄 산출.

  python scripts/eval/evaluate_offline.py --ckpt runs/pretrain_base/ckpt/best.pt
산출: runs/<exp>/eval/{metrics.md, psd_*.png, spec_*.png, band_*.png}
플랜트는 섭동 없는 결정적 S(z)(핸드오프 포함)로 계산한다.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.config import REPO_ROOT, load_yaml                     # noqa: E402
from deep_anc.data.synth_dataset import SynthANCDataset, make_eval_batch  # noqa: E402
from deep_anc.dsp.secondary_path import (                            # noqa: E402
    DifferentiableSecondaryPath,
    load_secondary_path,
)
from deep_anc.eval.metrics import nmse_db, octave_band_attenuation   # noqa: E402
from deep_anc.eval.plots import band_bar, psd_overlay, spectrogram_pair  # noqa: E402
from deep_anc.models import build_model                              # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-config", default="configs/data_sim.yaml")
    parser.add_argument("--duct-config", default="configs/duct.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--n-items", type=int, default=32)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    data_cfg = load_yaml(args.data_config)
    duct_cfg = load_yaml(args.duct_config)
    eval_cfg = load_yaml(args.eval_config)

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_model(state["cfg"]["model"])
    model.load_state_dict(state["model"])
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    fs = int(data_cfg["sample_rate"])
    sp = load_secondary_path(REPO_ROOT / duct_cfg["secondary_path"]["npz"])
    plant = DifferentiableSecondaryPath(
        sp, handoff_extra_samples=int(duct_cfg["secondary_path"].get("handoff_extra_samples", 0))
    ).to(device)

    ds = SynthANCDataset(data_cfg, duct_cfg, split="test", seed=999)
    batch = make_eval_batch(ds, n_items=args.n_items, seed=999)

    with torch.no_grad():
        x = batch["x"].to(device)
        d = batch["d"].to(device)
        y = model(x)
        e = d + plant(y.float(), {"jitter": 0})

    d_np = d.squeeze(1).cpu().numpy()
    e_np = e.squeeze(1).cpu().numpy()

    out_dir = Path(args.out) if args.out else Path(args.ckpt).parent.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    bands = eval_cfg.get("octave_bands_hz", [125, 250, 500, 1000, 2000, 4000, 8000])
    trusted = tuple(eval_cfg.get("trusted_band_hz", [150, 600]))

    per_item = [nmse_db(d_np[i], e_np[i]) for i in range(d_np.shape[0])]
    overall = float(np.mean(per_item))
    d_cat, e_cat = d_np.reshape(-1), e_np.reshape(-1)
    band_att = octave_band_attenuation(d_cat, e_cat, fs, bands, trusted)

    lines = [
        f"# 오프라인 평가 — {Path(args.ckpt).name}",
        "",
        f"- 테스트 아이템: {len(per_item)}개 (reference_mode={data_cfg.get('reference_mode')})",
        f"- **평균 NMSE: {overall:.2f} dB** (감쇠 {-overall:.2f} dB)",
        f"- 아이템 분포: 중앙값 {np.median(per_item):.2f} dB / 최악 {np.max(per_item):.2f} dB",
        "",
        "| 밴드(Hz) | 감쇠(dB) | 신뢰 |",
        "|---|---|---|",
    ]
    for b in band_att:
        mark = "O" if b["trusted"] else "낮음*"
        lines.append(f"| {b['center_hz']:.0f} | {b['attenuation_db']:+.2f} | {mark} |")
    lines += ["", "*: S(z) 보정 유효대역 밖 — 광대역 재보정 전에는 참고용."]
    (out_dir / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    spectrogram_pair(d_np[0], e_np[0], fs, out_dir / "spec_item0.png", "ANC OFF vs ON (시뮬)")
    psd_overlay({"d (OFF)": d_cat, "e (ON)": e_cat}, fs, out_dir / "psd.png", "PSD 비교")
    band_bar(band_att, out_dir / "band.png", "옥타브밴드 감쇠")

    print("\n".join(lines))
    print(f"\n산출물: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
