"""
Scoring and metrics for SGNM.

This module provides:
- metric: Compute MAE, MSE, or Pearson correlation between predictions and targets.
- StructureScorer: Score a model's predictions against experimental profiles.
- rank: Rank decoy structures by reactivity agreement.
- pearsonr_np: Numpy Pearson correlation matching scipy.stats.pearsonr interface.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import ciffy
from tqdm import tqdm

from .config import ScoringConfig


# =============================================================================
# Utilities
# =============================================================================


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Min-max normalize to [0, 1], per-column for 2D input."""
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    min_val = x.min(dim=0).values
    max_val = (x - min_val).max(dim=0).values.clamp(min=1e-8)
    return ((x - min_val) / max_val).squeeze()


# =============================================================================
# Metrics
# =============================================================================


def pearsonr_np(x, y):
    """Pearson correlation coefficient for numpy arrays.

    Returns (r, p_value) matching scipy.stats.pearsonr interface.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    xm, ym = x - x.mean(), y - y.mean()
    r = (xm * ym).sum() / (np.sqrt((xm ** 2).sum() * (ym ** 2).sum()) + 1e-12)
    t = r * np.sqrt((n - 2) / (1 - r ** 2 + 1e-12))
    p = 2 * np.exp(-0.5 * t ** 2) / np.sqrt(2 * np.pi) if n > 2 else 1.0
    return float(r), float(p)


def metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    metric: str = "correlation",
) -> torch.Tensor:
    """Compute a metric between pred and target, handling per-channel masking.

    Differentiable — can be used directly as a training loss (negate
    correlation to minimize).

    Args:
        pred: Predictions, shape (N,) or (N, C).
        target: Targets, shape (N,) or (N, C).
        mask: Boolean mask, shape (N,) or (N, C). Defaults to non-NaN positions.
        metric: One of "mae", "mse", "correlation".

    Returns:
        Per-channel values, shape (C,). Scalar if single-channel.

    Raises:
        ValueError: If pred and target have incompatible shapes.
    """
    if pred.dim() == 1:
        pred = pred.unsqueeze(-1)
    if target.dim() == 1:
        target = target.unsqueeze(-1)

    if mask is None:
        mask = ~torch.isnan(target)

    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}"
        )

    if mask.dim() == 1:
        mask = mask.unsqueeze(-1).expand_as(pred)

    # Zero out masked positions so they don't contribute to sums
    p = pred.where(mask, pred.new_zeros(()))
    t = target.where(mask, target.new_zeros(()))
    n = mask.sum(dim=0).clamp(min=1)  # (C,)

    if metric == "mae":
        result = torch.abs(p - t).sum(dim=0) / n
    elif metric == "mse":
        result = ((p - t) ** 2).sum(dim=0) / n
    elif metric == "correlation":
        p = p - (p.sum(dim=0) / n)
        t = t - (t.sum(dim=0) / n)
        p = p * mask
        t = t * mask
        num = (p * t).sum(dim=0)
        den = torch.linalg.norm(p, dim=0) * torch.linalg.norm(t, dim=0) + 1e-8
        result = num / den
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return result.squeeze()


# =============================================================================
# StructureScorer
# =============================================================================

class StructureScorer:
    """
    Core scoring logic for comparing predictions against target profiles.

    This class is model-agnostic and can work with any model that implements
    a .ciffy() method.
    """

    def __init__(
        self,
        model: nn.Module,
        config: ScoringConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or ScoringConfig()

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def score(
        self,
        target_profile: torch.Tensor,
        poly: ciffy.Polymer,
    ) -> float:
        """Score a polymer against a target reactivity profile.

        Args:
            target_profile: Target reactivity, shape (N,) or (N, C).
            poly: RNA polymer structure.

        Returns:
            Scalar score (higher = better for correlation).
        """
        pred = self.model.ciffy(poly)
        if self.config.channels is not None:
            pred = pred[:, self.config.channels]
        return metric(pred, target_profile, metric=self.config.metric).mean().item()



# =============================================================================
# Ranking
# =============================================================================

@dataclass
class RankingEntry:
    """A single ranked structure."""

    file: str
    score: float
    rank: int


@dataclass
class RankingResult:
    """Result of ranking a set of decoy structures."""

    entries: list[RankingEntry]
    reactivity_length: int

    @property
    def best(self) -> RankingEntry:
        return self.entries[0]

    @property
    def worst(self) -> RankingEntry:
        return self.entries[-1]

    def to_csv(self, path: str) -> None:
        import csv
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["rank", "file", "score"])
            writer.writeheader()
            for e in self.entries:
                writer.writerow({"rank": e.rank, "file": e.file, "score": e.score})


def rank(
    model: nn.Module,
    structures: str | Path | list[str | Path],
    reactivity: torch.Tensor,
    config: ScoringConfig | None = None,
) -> RankingResult:
    """Rank decoy structures by agreement between predicted and experimental reactivity.

    Args:
        model: Trained model with a .ciffy() method.
        structures: Directory of .cif files, or a list of .cif paths.
        reactivity: Target reactivity aligned to structure length,
            shape (L,) or (L, C) for multi-channel.
        config: Scoring configuration (blank masking, metric).

    Returns:
        RankingResult sorted by score (best first).
    """
    device = next(model.parameters()).device
    reactivity = reactivity.to(device)
    scorer = StructureScorer(model, config)

    if isinstance(structures, (str, Path)):
        cif_files = sorted(Path(structures).glob("*.cif"))
    else:
        cif_files = [Path(p) for p in structures]

    entries = []
    for path in tqdm(cif_files, desc="Ranking"):
        try:
            poly = ciffy.load(str(path), backend="torch")
            if device.type == "cuda":
                poly = poly.cuda()
            score = scorer.score(reactivity, poly)
            entries.append(RankingEntry(
                file=path.name, score=score, rank=0,
            ))
        except Exception as e:
            print(f"  Ranking error on {path.name}: {e}")
            continue

    entries.sort(key=lambda e: e.score, reverse=True)
    for i, e in enumerate(entries):
        e.rank = i + 1

    return RankingResult(entries=entries, reactivity_length=reactivity.shape[0])
