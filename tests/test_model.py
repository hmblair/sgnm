"""Tests for the SGNM model forward pass and checkpoint round-tripping.

These exercise the batched GNM variance computation, which depends on
ciffy supporting (*, N, N) adjacency input (ciffy >= 1.0.3).
"""
import pytest
import torch

import sgnm
from sgnm.models import SGNM


def make_model(**kw):
    defaults = dict(dim=8, out_channels=2, gnm_channels=4, layers=1)
    defaults.update(kw)
    return SGNM(**defaults).eval()


def random_frames(n):
    # Valid rotation matrices via QR of a random matrix.
    return torch.linalg.qr(torch.randn(n, 3, 3))[0]


def test_forward_output_shape():
    model = make_model(out_channels=2)
    out = model(torch.randn(12, 3), random_frames(12))
    assert out.shape == (12, 2)


def test_forward_without_frames():
    model = make_model()
    out = model(torch.randn(12, 3), None)
    assert out.shape == (12, 2)


def test_forward_is_deterministic():
    torch.manual_seed(0)
    model = make_model()
    coords, frames = torch.randn(10, 3), random_frames(10)
    assert torch.allclose(model(coords, frames), model(coords, frames))


def test_frames_change_output():
    torch.manual_seed(0)
    model = make_model()
    coords = torch.randn(10, 3)
    assert not torch.allclose(model(coords, None), model(coords, random_frames(10)))


def test_out_channels_respected():
    model = make_model(out_channels=1)
    assert model(torch.randn(8, 3), None).shape == (8, 1)


def test_forward_is_differentiable():
    model = make_model()
    coords = torch.randn(10, 3, requires_grad=True)
    model(coords, None).sum().backward()
    assert coords.grad is not None and torch.isfinite(coords.grad).all()


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = make_model()
    path = tmp_path / "ckpt.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "init_kwargs": model._init_kwargs,
            "model_type": "SGNM",
        },
        path,
    )
    loaded = sgnm.load(str(path))
    coords, frames = torch.randn(10, 3), random_frames(10)
    assert torch.allclose(model(coords, frames), loaded(coords, frames))


def test_load_unknown_model_type_raises(tmp_path):
    path = tmp_path / "bad.pth"
    torch.save({"model_state_dict": {}, "model_type": "DoesNotExist"}, path)
    with pytest.raises(ValueError):
        sgnm.load(str(path))
