"""Tests for sgnm.scoring.metric, the differentiable metric used as both
the training loss and the evaluation/ranking score."""
import pytest
import torch

from sgnm.scoring import metric, normalize


def test_correlation_self_is_one():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert metric(x, x, metric="correlation").item() == pytest.approx(1.0, abs=1e-5)


def test_correlation_negated_is_minus_one():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert metric(x, -x, metric="correlation").item() == pytest.approx(-1.0, abs=1e-5)


def test_correlation_matches_numpy_reference():
    torch.manual_seed(0)
    p, t = torch.randn(50), torch.randn(50)
    expected = torch.corrcoef(torch.stack([p, t]))[0, 1]
    assert metric(p, t, metric="correlation").item() == pytest.approx(expected.item(), abs=1e-5)


def test_mae_and_mse_known_values():
    assert metric(torch.zeros(3), torch.ones(3), metric="mae").item() == pytest.approx(1.0)
    assert metric(torch.zeros(3), 2 * torch.ones(3), metric="mse").item() == pytest.approx(4.0)


def test_multichannel_returns_per_channel():
    out = metric(torch.randn(10, 3), torch.randn(10, 3), metric="mae")
    assert out.shape == (3,)


def test_single_channel_is_scalar():
    out = metric(torch.randn(10), torch.randn(10), metric="mae")
    assert out.dim() == 0


def test_nan_targets_are_masked():
    target = torch.tensor([1.0, 2.0, float("nan"), 4.0])
    pred = torch.tensor([1.0, 2.0, 99.0, 4.0])  # error only at the masked position
    assert metric(pred, target, metric="mae").item() == pytest.approx(0.0)


def test_explicit_mask_overrides_default():
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    pred = torch.tensor([1.0, 2.0, 3.0, 99.0])
    mask = torch.tensor([True, True, True, False])
    assert metric(pred, target, mask=mask, metric="mae").item() == pytest.approx(0.0)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        metric(torch.randn(5), torch.randn(6))


def test_unknown_metric_raises():
    with pytest.raises(ValueError):
        metric(torch.randn(5), torch.randn(5), metric="bogus")


def test_metric_is_differentiable():
    pred = torch.randn(10, requires_grad=True)
    target = torch.randn(10)
    loss = -metric(pred, target, metric="correlation")  # negate to minimize
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_normalize_maps_to_unit_range():
    out = normalize(torch.tensor([0.0, 5.0, 10.0]))
    assert out.tolist() == [0.0, 0.5, 1.0]


def test_normalize_is_per_column():
    out = normalize(torch.tensor([[0.0, 100.0], [10.0, 200.0]]))
    assert out.tolist() == [[0.0, 0.0], [1.0, 1.0]]
