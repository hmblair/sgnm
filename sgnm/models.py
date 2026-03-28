"""
SGNM model classes.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import ciffy
from ciffy.biochemistry.constants import PurineBase, PyrimidineBase
from ciffy.biochemistry.atom import AtomGroup
from ciffy.biochemistry.linking import FrameDefinition
from ciffy.operations.frames import gather
from ciffy.geometry.transforms import frame_from_positions

from .nn import RadialBasisFunctions, DenseNetwork
from .config import ModelConfig
from .gnm import (
    _gnm_variances,
    _orientation_score,
)

# Unified C2/C4/C6 atom groups across purines and pyrimidines
_NucleobaseC2 = AtomGroup(
    "NucleobaseC2",
    {**PurineBase.C2._members, **PyrimidineBase.C2._members},
)
_NucleobaseC4 = AtomGroup(
    "NucleobaseC4",
    {**PurineBase.C4._members, **PyrimidineBase.C4._members},
)
_NucleobaseC6 = AtomGroup(
    "NucleobaseC6",
    {**PurineBase.C6._members, **PyrimidineBase.C6._members},
)

NUCLEOBASE_FRAME = FrameDefinition(
    origin=_NucleobaseC2,
    axis_ref=_NucleobaseC4,
    plane_ref=_NucleobaseC6,
)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to [0, 1] range."""
    min_val = x.min()
    max_val = (x - min_val).max()
    return (x - min_val) / max_val


def _base_frame(poly: ciffy.Polymer) -> torch.Tensor:
    """Get per-residue C2-C4-C6 nucleobase frames."""
    positions = gather(poly, [
        NUCLEOBASE_FRAME.origin,
        NUCLEOBASE_FRAME.axis_ref,
        NUCLEOBASE_FRAME.plane_ref,
    ])
    _, R = frame_from_positions(positions)
    return R


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
        poly = poly.torch()
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
    ) -> torch.Tensor:
        """
        Compute predictions from coordinates and optional frames.

        Args:
            coords: Residue center coordinates, shape (N, 3)
            frames: Local coordinate frames, shape (N, 3, 3)

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
    ) -> torch.Tensor:
        """
        Compute predictions from a ciffy Polymer object.

        Args:
            poly: RNA polymer structure

        Returns:
            Normalized variance predictions
        """
        poly = poly.torch().strip()
        poly, coords = poly.center(ciffy.RESIDUE)
        frames = _base_frame(poly)

        return self(coords, frames)

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
