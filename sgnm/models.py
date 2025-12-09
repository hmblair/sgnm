"""
SGNM model classes.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import ciffy

from .nn import RadialBasisFunctions, DenseNetwork
from .config import ModelConfig, FRAME1, FRAME2, FRAME3
from .gnm import (
    _gnm_variances,
    _orientation_score,
    _local_frame,
)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to [0, 1] range."""
    min_val = x.min()
    max_val = (x - min_val).max()
    return (x - min_val) / max_val


def _base_frame(poly: ciffy.Polymer) -> torch.Tensor:
    """
    Get pairwise base orientations based on the C2-C4-C6 frame.
    """
    poly1 = poly.get_by_name(FRAME1)
    poly2 = poly.get_by_name(FRAME2)
    poly3 = poly.get_by_name(FRAME3)

    diff1 = poly1.coordinates - poly2.coordinates
    diff2 = poly1.coordinates - poly3.coordinates
    return _local_frame(diff1, diff2)


class BaseSGNM(nn.Module):
    """
    Base SGNM model using inverse-square distance weighting.

    This non-parametric model computes GNM variances directly from
    coordinates without any learnable parameters.
    """

    def __init__(self: BaseSGNM) -> None:
        super().__init__()

    def forward(
        self: BaseSGNM,
        coords: torch.Tensor,
        frames: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute normalized GNM variances from coordinates.

        Args:
            coords: Residue center coordinates, shape (N, 3)
            frames: Ignored in base model

        Returns:
            Normalized variance predictions, shape (N,)
        """
        dists = torch.cdist(coords, coords)
        dists = 1 / dists ** 2

        ix = range(dists.size(0))
        dists[ix, ix] = 0.0

        emb = _gnm_variances(dists)
        return _normalize(emb)

    def ciffy(
        self: BaseSGNM,
        poly: ciffy.Polymer,
    ) -> torch.Tensor:
        """
        Compute predictions from a ciffy Polymer object.

        Args:
            poly: RNA polymer structure

        Returns:
            Normalized variance predictions
        """
        poly = poly.frame()
        _, coords = poly.center(ciffy.RESIDUE)
        return self(coords)


class SGNM(BaseSGNM):
    """
    Structure-Guided Normal Mode model with learnable embeddings.

    Uses two pathways:
    1. Distance-based: RBF embedding of pairwise distances
    2. Orientation-based: RBF embedding of base orientation scores

    The embeddings are combined multiplicatively before computing
    GNM variances.
    """

    def __init__(
        self: SGNM,
        config_or_dim: ModelConfig | int,
        out_dim: int = 1,
        layers: int = 1,
    ) -> None:
        """
        Initialize SGNM model.

        Args:
            config_or_dim: ModelConfig dataclass or dimension (int for backward compatibility)
            out_dim: Output dimension (only used if config_or_dim is int)
            layers: Number of hidden layers (only used if config_or_dim is int)
        """
        super().__init__()

        # Handle backward compatibility
        if isinstance(config_or_dim, int):
            config = ModelConfig(dim=config_or_dim, out_dim=out_dim, layers=layers)
        else:
            config = config_or_dim

        self.config = config
        dim = config.dim

        # Distance pathway
        self.rbf1 = RadialBasisFunctions(dim)
        self.linear1 = DenseNetwork(dim, config.out_dim, [dim] * config.layers)

        # Orientation pathway
        self.rbf2 = RadialBasisFunctions(dim)
        self.linear2 = DenseNetwork(dim, config.out_dim, [dim] * config.layers)

    def embed1(
        self: SGNM,
        dists: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distance-based embedding."""
        adj = self.rbf1(dists)
        adj = self.linear1(adj).squeeze(-1)
        return _normalize(adj)

    def embed2(
        self: SGNM,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        """Compute orientation-based embedding."""
        adj = self.rbf2(scores)
        adj = self.linear2(adj).squeeze(-1)
        return _normalize(adj)

    def forward(
        self: SGNM,
        coords: torch.Tensor,
        frames: torch.Tensor | None = None,
        sequence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute predictions from coordinates and optional frames.

        Args:
            coords: Residue center coordinates, shape (N, 3)
            frames: Local coordinate frames, shape (N, 3, 3)
            sequence: Tokenized sequence (currently unused)

        Returns:
            Normalized variance predictions, shape (N,)
        """
        dists = torch.cdist(coords, coords)
        emb = self.embed1(dists)

        if frames is not None:
            scores = _orientation_score(frames)
            emb = emb * self.embed2(scores)

        emb = _gnm_variances(emb)
        return _normalize(emb)

    def ciffy(
        self: SGNM,
        poly: ciffy.Polymer,
        sequence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute predictions from a ciffy Polymer object.

        Args:
            poly: RNA polymer structure
            sequence: Tokenized sequence (optional)

        Returns:
            Normalized variance predictions
        """
        poly = poly.frame().strip()
        poly, coords = poly.center(ciffy.RESIDUE)
        frames = _base_frame(poly)

        return self(coords, frames, sequence)

    @classmethod
    def load(
        cls: type[SGNM],
        path: str | None = None,
    ) -> BaseSGNM:
        """
        Load model from weights file.

        If path is None, returns a non-parametric BaseSGNM model.

        Args:
            path: Path to weights file, or None for base model

        Returns:
            Loaded model (SGNM if path provided, BaseSGNM otherwise)
        """
        if path is None:
            return BaseSGNM()

        weights = torch.load(path, weights_only=True)
        dim = weights['rbf1.mu'].size(0)

        module = SGNM(dim)
        module.load_state_dict(weights)

        return module
