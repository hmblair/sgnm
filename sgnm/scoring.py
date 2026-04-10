"""
Scoring components for SGNM.

This module provides:
- StructureScorer: Core scoring logic (model-agnostic)
- StructureRelaxer: Gradient-based structure optimization
- rank: Rank decoy structures by reactivity agreement
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from copy import deepcopy
import torch
import torch.nn as nn
import ciffy
from tqdm import tqdm

from .config import ScoringConfig, RelaxConfig


# =============================================================================
# Result Containers
# =============================================================================

@dataclass
class ScoringResult:
    """Result container for scoring operations."""

    score: torch.Tensor
    """The computed score (MAE, MSE, or correlation)."""

    prediction: torch.Tensor
    """Model prediction."""

    target: torch.Tensor
    """Target profile."""

    mask: torch.Tensor
    """Boolean mask for valid positions."""

    metadata: dict = field(default_factory=dict)
    """Additional metadata (e.g., file path, name)."""

    @property
    def score_value(self) -> float:
        """Get score as a Python float."""
        return self.score.item()


@dataclass
class RelaxResult:
    """Result container for structure relaxation."""

    original: ciffy.Polymer
    """Original polymer structure."""

    relaxed: ciffy.Polymer
    """Relaxed polymer structure."""

    history: list[dict]
    """Optimization history (loss, score, rmsd per step)."""

    final_score: float
    """Final score after relaxation."""

    def save(self, path: str | Path) -> None:
        """Save relaxed structure to file."""
        self.relaxed.write(str(path))


# =============================================================================
# Metrics
# =============================================================================


def _metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    metric: str = "correlation",
) -> torch.Tensor:
    """Compute a metric between pred and target, handling per-channel masking.

    Args:
        pred: Predictions, shape (N,) or (N, C).
        target: Targets, shape (N,) or (N, C).
        mask: Boolean mask, shape (N,) or (N, C).
        metric: One of "mae", "mse", "correlation".

    Returns:
        Scalar mean across channels.
    """
    if pred.dim() == 1:
        pred = pred.unsqueeze(-1)
    if target.dim() == 1:
        target = target.unsqueeze(-1)
    if mask.dim() == 1:
        mask = mask.unsqueeze(-1).expand_as(pred)

    # Zero out masked positions so they don't contribute to sums
    p = pred.where(mask, pred.new_zeros(()))
    t = target.where(mask, target.new_zeros(()))
    n = mask.sum(dim=0).clamp(min=1)  # (C,)

    if metric == "mae":
        return (torch.abs(p - t).sum(dim=0) / n).mean()
    elif metric == "mse":
        return (((p - t) ** 2).sum(dim=0) / n).mean()
    elif metric == "correlation":
        p = p - (p.sum(dim=0) / n)
        t = t - (t.sum(dim=0) / n)
        p = p * mask
        t = t * mask
        num = (p * t).sum(dim=0)
        den = torch.linalg.norm(p, dim=0) * torch.linalg.norm(t, dim=0) + 1e-8
        return (num / den).mean()
    else:
        raise ValueError(f"Unknown metric: {metric}")


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
        """
        Initialize scorer.

        Args:
            model: Model to use for predictions
            config: Scoring configuration
        """
        self.model = model
        self.config = config or ScoringConfig()

        # Freeze model for scoring
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def _apply_blank_mask(self, profile: torch.Tensor) -> torch.Tensor:
        """Create mask excluding blank regions and NaN values."""
        mask = ~torch.isnan(profile)
        if self.config.blank_start is not None:
            mask[:self.config.blank_start] = False
        if self.config.blank_end is not None:
            mask[-self.config.blank_end:] = False
        return mask

    def _compute_metric(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the scoring metric."""
        return _metric(pred, target, mask, self.config.metric)

    def score(
        self,
        target_profile: torch.Tensor,
        coords: torch.Tensor,
        frames: torch.Tensor | None = None,
    ) -> ScoringResult:
        """
        Score raw tensor inputs.

        Args:
            target_profile: Target SHAPE profile, shape (N,)
            coords: Residue coordinates, shape (N, 3)
            frames: Local frames, shape (N, 3, 3) or None

        Returns:
            ScoringResult with score, prediction, and metadata
        """
        pred = self.model(coords, frames)
        mask = self._apply_blank_mask(target_profile)
        score = self._compute_metric(pred, target_profile, mask)
        return ScoringResult(
            score=score,
            prediction=pred,
            target=target_profile,
            mask=mask,
        )

    def score_polymer(
        self,
        target_profile: torch.Tensor,
        poly: ciffy.Polymer,
    ) -> ScoringResult:
        """
        Score a ciffy Polymer object.

        Args:
            target_profile: Target SHAPE profile
            poly: RNA polymer structure

        Returns:
            ScoringResult with score, prediction, and metadata
        """
        pred = self.model.ciffy(poly)
        mask = self._apply_blank_mask(target_profile)
        score = self._compute_metric(pred, target_profile, mask)
        return ScoringResult(
            score=score,
            prediction=pred,
            target=target_profile,
            mask=mask,
        )

    def score_cif_file(
        self,
        cif_path: str | Path,
        target_profile: torch.Tensor,
    ) -> ScoringResult:
        """
        Score a .cif file directly.

        Args:
            cif_path: Path to .cif file
            target_profile: Target SHAPE profile

        Returns:
            ScoringResult with score, prediction, and metadata
        """
        poly = ciffy.load(str(cif_path), backend="torch")
        result = self.score_polymer(target_profile, poly)
        result.metadata["path"] = str(cif_path)
        result.metadata["name"] = Path(cif_path).stem
        return result


# =============================================================================
# StructureRelaxer
# =============================================================================

class StructureRelaxer:
    """
    Gradient-based structure optimization.

    Optimizes coordinates to minimize the scoring loss while regularizing
    against RMSD drift from the original structure.
    """

    def __init__(
        self,
        scorer: StructureScorer,
        config: RelaxConfig | None = None,
    ) -> None:
        """
        Initialize relaxer.

        Args:
            scorer: Scorer to use as objective
            config: Relaxation configuration
        """
        self.scorer = scorer
        self.config = config or RelaxConfig()

    def relax(
        self,
        target_profile: torch.Tensor,
        poly: ciffy.Polymer,
        progress: bool = True,
    ) -> RelaxResult:
        """
        Optimize structure coordinates to match target profile.

        Args:
            target_profile: Target SHAPE profile
            poly: Initial polymer structure
            progress: Show progress bar

        Returns:
            RelaxResult with original, relaxed structure, and history
        """
        relaxed = deepcopy(poly)
        relaxed.coordinates.requires_grad_(True)

        optimizer = torch.optim.Adam([relaxed.coordinates], lr=self.config.lr)
        history = []

        iterator = trange(self.config.steps) if progress else range(self.config.steps)

        for step in iterator:
            optimizer.zero_grad()

            result = self.scorer.score_polymer(target_profile, relaxed)
            rmsd = ciffy.rmsd(poly, relaxed)
            loss = result.score + self.config.alpha * rmsd

            loss.backward()
            optimizer.step()

            history.append({
                "step": step,
                "loss": loss.item(),
                "score": result.score_value,
                "rmsd": rmsd.item(),
            })

        return RelaxResult(
            original=poly,
            relaxed=relaxed,
            history=history,
            final_score=history[-1]["score"],
        )


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
) -> RankingResult:
    """Rank decoy structures by agreement between predicted and experimental reactivity.

    Args:
        model: Trained model with a .ciffy() method.
        structures: Directory of .cif files, or a list of .cif paths.
        reactivity: Target reactivity aligned to structure length,
            shape (L,) or (L, C) for multi-channel.

    Returns:
        RankingResult sorted by score (best first).
    """
    if isinstance(structures, (str, Path)):
        cif_files = sorted(Path(structures).glob("*.cif"))
    else:
        cif_files = [Path(p) for p in structures]

    entries = []
    model.eval()
    with torch.no_grad():
        for path in tqdm(cif_files, desc="Ranking"):
            try:
                poly = ciffy.load(str(path), backend="torch")
                pred = model.ciffy(poly)
                mask = ~torch.isnan(reactivity)
                if mask.dim() == 1:
                    mask = mask & ~torch.isnan(pred.squeeze(-1))
                score = _metric(pred, reactivity, mask, "correlation").item()
                entries.append(RankingEntry(file=path.name, score=score, rank=0))
            except Exception:
                continue

    entries.sort(key=lambda e: e.score, reverse=True)
    for i, e in enumerate(entries):
        e.rank = i + 1

    return RankingResult(entries=entries, reactivity_length=reactivity.shape[0])
