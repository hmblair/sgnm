"""Tests for the StructureScorer / ranking pipeline.

StructureScorer is model-agnostic: it works with any object exposing a
``.ciffy(poly)`` method. We use a stub that returns a fixed profile so the
scoring logic can be tested without checkpoints or .cif files.
"""
import torch

from sgnm.config import ScoringConfig
from sgnm.scoring import StructureScorer


class StubModel(torch.nn.Module):
    """Returns a fixed prediction regardless of input."""

    def __init__(self, prediction):
        super().__init__()
        self._prediction = prediction

    def ciffy(self, poly):
        return self._prediction


def test_score_self_correlation_is_one():
    profile = torch.tensor([1.0, 2.0, 3.0, 4.0])
    scorer = StructureScorer(StubModel(profile), ScoringConfig(metric="correlation"))
    assert scorer.score(profile, poly=None) == 1.0


def test_score_anticorrelation_is_minus_one():
    profile = torch.tensor([1.0, 2.0, 3.0, 4.0])
    scorer = StructureScorer(StubModel(-profile), ScoringConfig(metric="correlation"))
    assert scorer.score(profile, poly=None) == -1.0


def test_score_returns_python_float():
    profile = torch.tensor([1.0, 2.0, 3.0, 4.0])
    scorer = StructureScorer(StubModel(profile), ScoringConfig(metric="correlation"))
    assert isinstance(scorer.score(profile, poly=None), float)


def test_channel_selection():
    # Model emits two channels: 0 matches the target, 1 is anti-correlated.
    # Selecting channel 0 should score 1.0 against the single-channel target.
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    pred = torch.stack([target, -target], dim=1)  # (N, 2)
    scorer = StructureScorer(StubModel(pred), ScoringConfig(metric="correlation", channels=[0]))
    assert scorer.score(target, poly=None) == 1.0


def test_scorer_freezes_model_gradients():
    model = StubModel(torch.zeros(3))
    lin = torch.nn.Linear(3, 3)
    model.add_module("lin", lin)
    StructureScorer(model, ScoringConfig())
    assert all(not p.requires_grad for p in model.parameters())
