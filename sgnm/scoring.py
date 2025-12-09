"""
Scoring components for SGNM.

This module provides:
- StructureScorer: Core scoring logic (model-agnostic)
- BatchScorer: Batch processing of .cif folders
- StructureRelaxer: Gradient-based structure optimization
- Score: Backward-compatible wrapper
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from copy import deepcopy
from typing import Iterator
import torch
import torch.nn as nn
import ciffy
from tqdm import trange, tqdm

from .models import SGNM, BaseSGNM
from .config import ScoringConfig, FilterConfig, RelaxConfig, BatchScoringConfig


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

    def save(self, path: str | Path, format: str = "pdb") -> None:
        """Save relaxed structure to file."""
        self.relaxed.write(str(path))


@dataclass
class BatchScoringResults:
    """Container for batch scoring results with filtering."""

    results: list[ScoringResult]
    """All scoring results."""

    filter_config: FilterConfig
    """Configuration used for filtering."""

    @property
    def scores(self) -> list[float]:
        """Get all scores as a list."""
        return [r.score_value for r in self.results]

    @property
    def passed(self) -> list[ScoringResult]:
        """Results that pass the filter threshold."""
        filtered = [
            r for r in self.results
            if self._passes_filter(r.score_value)
        ]

        # Apply top_k if specified
        if self.filter_config.top_k is not None:
            reverse = self.filter_config.mode == "above"
            filtered = sorted(filtered, key=lambda r: r.score_value, reverse=reverse)
            filtered = filtered[:self.filter_config.top_k]

        return filtered

    @property
    def failed(self) -> list[ScoringResult]:
        """Results that fail the filter threshold."""
        passed_set = set(id(r) for r in self.passed)
        return [r for r in self.results if id(r) not in passed_set]

    def _passes_filter(self, score: float) -> bool:
        if self.filter_config.mode == "below":
            return score < self.filter_config.threshold
        return score > self.filter_config.threshold

    def summary(self) -> dict:
        """Return summary statistics."""
        scores = self.scores
        if not scores:
            return {"total": 0, "passed": 0, "failed": 0}

        return {
            "total": len(self.results),
            "passed": len(self.passed),
            "failed": len(self.failed),
            "mean_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
        }


# =============================================================================
# StructureScorer
# =============================================================================

class StructureScorer:
    """
    Core scoring logic for comparing predictions against target profiles.

    This class is model-agnostic and can work with any model that implements
    the BaseSGNM interface.
    """

    def __init__(
        self,
        model: BaseSGNM,
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
        if self.config.metric == "mae":
            return torch.abs(pred[mask] - target[mask]).mean()
        elif self.config.metric == "mse":
            return ((pred[mask] - target[mask]) ** 2).mean()
        elif self.config.metric == "correlation":
            p, t = pred[mask], target[mask]
            p = p - p.mean()
            t = t - t.mean()
            return (p * t).sum() / (torch.linalg.norm(p) * torch.linalg.norm(t))
        raise ValueError(f"Unknown metric: {self.config.metric}")

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
        poly = ciffy.load(str(cif_path))
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
# BatchScorer
# =============================================================================

class BatchScorer:
    """
    Process folders of .cif predictions and filter by score.

    Main interface for synthetic data pipeline.
    """

    def __init__(
        self,
        scorer: StructureScorer,
        config: BatchScoringConfig,
        filter_config: FilterConfig | None = None,
    ) -> None:
        """
        Initialize batch scorer.

        Args:
            scorer: Scorer to use for individual files
            config: Batch scoring configuration
            filter_config: Filtering configuration
        """
        self.scorer = scorer
        self.config = config
        self.filter_config = filter_config or FilterConfig()
        self._profile_cache: dict[str, torch.Tensor] = {}

    def _load_profiles(self) -> None:
        """Load all profiles from the profile file."""
        if not self.config.profile_path:
            return

        import h5py
        with h5py.File(self.config.profile_path, 'r') as f:
            # Try different HDF5 formats
            if 'ids' in f:
                # v2 format
                names = list(f['ids'][:].astype(str))
                if 'PDB130-2A3/reactivity' in f:
                    reacs = torch.from_numpy(f['PDB130-2A3/reactivity'][:])
                    for i, name in enumerate(names):
                        self._profile_cache[name] = reacs[i]
            elif 'id_strings' in f:
                # v1 format
                names = f['id_strings'][0].astype(str)
                reacs = torch.from_numpy(f['r_norm'][:])
                for name, reac in zip(names, reacs):
                    self._profile_cache[name] = reac

    def _get_profile(self, name: str) -> torch.Tensor | None:
        """Get profile for a given name."""
        if not self._profile_cache and self.config.profile_path:
            self._load_profiles()
        return self._profile_cache.get(name)

    def _get_cif_files(self) -> list[Path]:
        """Get list of .cif files to process."""
        input_dir = Path(self.config.input_dir)
        if self.config.recursive:
            return list(input_dir.rglob(self.config.file_pattern))
        return list(input_dir.glob(self.config.file_pattern))

    def score_all(
        self,
        profiles: dict[str, torch.Tensor] | None = None,
        progress: bool = True,
    ) -> BatchScoringResults:
        """
        Score all .cif files in the input directory.

        Args:
            profiles: Dict mapping names to profiles (if not using profile_path)
            progress: Show progress bar

        Returns:
            BatchScoringResults with all results
        """
        if profiles:
            self._profile_cache = profiles

        cif_files = self._get_cif_files()
        results = []

        iterator = tqdm(cif_files) if progress else cif_files

        for cif_path in iterator:
            profile = self._get_profile(cif_path.stem)
            if profile is None:
                continue

            try:
                result = self.scorer.score_cif_file(cif_path, profile)
                results.append(result)
            except Exception as e:
                print(f"Error processing {cif_path}: {e}")

        return BatchScoringResults(
            results=results,
            filter_config=self.filter_config,
        )

    def filter_and_copy(
        self,
        profiles: dict[str, torch.Tensor] | None = None,
        progress: bool = True,
    ) -> BatchScoringResults:
        """
        Score all files, filter by threshold, and copy passing files.

        Args:
            profiles: Dict mapping names to profiles
            progress: Show progress bar

        Returns:
            BatchScoringResults with filtered results
        """
        import shutil

        batch_results = self.score_all(profiles, progress)

        if self.config.output_dir:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            for result in batch_results.passed:
                src = Path(result.metadata["path"])
                dst = output_dir / src.name
                shutil.copy(src, dst)

        return batch_results


# =============================================================================
# Score (Backward Compatible)
# =============================================================================

class Score(nn.Module):
    """
    Score structures against experimental profiles.

    This class provides backward compatibility with the original API.
    For new code, consider using StructureScorer directly.
    """

    def __init__(
        self: Score,
        weights: str | None = None,
    ) -> None:
        """
        Initialize Score with model weights.

        Args:
            weights: Path to weights file, or None for base model
        """
        super().__init__()

        self.module = SGNM.load(weights)
        self.module.eval()
        for param in self.module.parameters():
            param.requires_grad = False

    def forward(
        self: Score,
        profile: torch.Tensor,
        coords: torch.Tensor,
        frames: torch.Tensor | None = None,
        blank: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the MAE between the provided and predicted SHAPE profiles.

        Args:
            profile: Target SHAPE profile
            coords: Residue coordinates
            frames: Local coordinate frames
            blank: Tuple of (start, end) residues to mask

        Returns:
            Tuple of (MAE score, predicted profile)
        """
        pred = self.module(coords, frames)
        ix = ~torch.isnan(profile)

        if blank is not None:
            ix[:blank[0]] = False
            ix[-blank[1]:] = False

        return torch.abs(pred[ix] - profile[ix]).mean(), pred

    def ciffy(
        self: Score,
        profile: torch.Tensor,
        poly: ciffy.Polymer,
        blank: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Score a ciffy Polymer object.

        Args:
            profile: Target SHAPE profile
            poly: RNA polymer structure
            blank: Tuple of (start, end) residues to mask

        Returns:
            Tuple of (MAE score, predicted profile)
        """
        pred = self.module.ciffy(poly)
        ix = ~torch.isnan(profile)

        if blank is not None:
            ix[:blank[0]] = False
            ix[-blank[1]:] = False

        return torch.mean((pred[ix] - profile[ix]).abs()), pred

    def relax(
        self: Score,
        profile: torch.Tensor,
        poly: ciffy.Polymer,
        steps: int = 1,
        lr: float = 1e-3,
        alpha: float = 1e-3,
    ) -> ciffy.Polymer:
        """
        Relax structure coordinates to match target profile.

        Args:
            profile: Target SHAPE profile
            poly: Initial polymer structure
            steps: Number of optimization steps
            lr: Learning rate
            alpha: RMSD regularization weight

        Returns:
            Relaxed polymer structure
        """
        relaxed = deepcopy(poly)

        relaxed.coordinates.requires_grad_(True)
        opt = torch.optim.Adam([relaxed.coordinates], lr=lr)

        for _ in trange(steps):
            opt.zero_grad()
            mae, _ = self.ciffy(profile, relaxed)
            loss = mae + alpha * ciffy.rmsd(poly, relaxed)
            loss.backward()
            opt.step()

        return relaxed
