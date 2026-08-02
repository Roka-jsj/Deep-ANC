"""비선형 증강 검증 — 범위·선형 극한·미분가능성."""

import numpy as np
import torch

from deep_anc.dsp.nonlinear import RandomNonlinear, drive_np, sef_np, sef_torch


def test_sef_linear_limit():
    x = np.linspace(-0.5, 0.5, 101).astype(np.float32)
    assert np.allclose(sef_np(x, 10.0), x)        # η=10 → 선형
    y = sef_np(x, 0.1)
    assert np.max(np.abs(y)) <= 0.1 + 1e-6        # 포화 한계


def test_drive_monotone():
    x = np.linspace(-1, 1, 201).astype(np.float32)
    y = drive_np(x, 4.0)
    assert np.all(np.diff(y) >= 0)


def test_random_nonlinear_torch_grad():
    rnl = RandomNonlinear([0.1, 0.5, 1.0, 10.0], (1.0, 4.0), hardclip_prob=0.5, seed=3)
    params = rnl.sample(4)
    y = (torch.randn(4, 1, 1024) * 0.1).requires_grad_()
    out = rnl.apply_torch(y, params)
    out.mean().backward()
    assert torch.isfinite(y.grad).all()
    assert out.shape == y.shape


def test_sef_torch_batch_eta():
    y = torch.randn(3, 1, 64) * 0.5
    eta = torch.tensor([0.1, 1.0, 10.0]).view(-1, 1, 1)
    out = sef_torch(y, eta)
    assert out.shape == y.shape
    assert torch.allclose(out[2], y[2])           # η=10 채널은 선형
