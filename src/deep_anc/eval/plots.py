"""평가 플롯 — ANC OFF/ON 스펙트로그램·PSD 비교, 밴드 감쇠 막대."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def spectrogram_pair(
    d: np.ndarray, e: np.ndarray, fs: int, out_path: str | Path, title: str = ""
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, sig, name in ((axes[0], d, "ANC OFF (d)"), (axes[1], e, "ANC ON (e)")):
        f, t, S = signal.spectrogram(sig, fs=fs, nperseg=1024, noverlap=768)
        ax.pcolormesh(t, f, 10 * np.log10(S + 1e-14), shading="gouraud", cmap="magma")
        ax.set_title(name)
        ax.set_xlabel("시간 (s)")
        ax.set_ylim(0, 4000)
    axes[0].set_ylabel("주파수 (Hz)")
    fig.suptitle(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def psd_overlay(
    signals: dict[str, np.ndarray], fs: int, out_path: str | Path, title: str = ""
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, sig in signals.items():
        f, p = signal.welch(sig, fs=fs, nperseg=4096)
        ax.semilogx(f, 10 * np.log10(p + 1e-16), label=name)
    ax.set_xlim(20, fs / 2)
    ax.set_xlabel("주파수 (Hz)")
    ax.set_ylabel("PSD (dB/Hz)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def band_bar(
    band_results: list[dict], out_path: str | Path, title: str = ""
) -> Path:
    centers = [b["center_hz"] for b in band_results]
    values = [b["attenuation_db"] for b in band_results]
    colors = ["tab:blue" if b.get("trusted", True) else "tab:gray" for b in band_results]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(centers)), values, color=colors)
    ax.set_xticks(range(len(centers)))
    ax.set_xticklabels([f"{c:.0f}" for c in centers])
    ax.set_xlabel("옥타브밴드 중심 (Hz) — 회색 = S(z) 유효대역 밖(신뢰 낮음)")
    ax.set_ylabel("감쇠 (dB)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
