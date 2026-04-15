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

from dlu import RadialBasisFunctions, DenseNetwork
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





def _base_frame(poly: ciffy.Polymer) -> torch.Tensor:
    """Get per-residue C2-C4-C6 nucleobase frames."""
    positions = gather(poly, [
        NUCLEOBASE_FRAME.origin,
        NUCLEOBASE_FRAME.axis_ref,
        NUCLEOBASE_FRAME.plane_ref,
    ])
    _, R = frame_from_positions(positions)
    return R


class SGNM(nn.Module):
    """
    Structure-Guided Normal Mode model with learnable embeddings.

    Uses two pathways:
    1. Distance-based: RBF embedding of pairwise distances
    2. Orientation-based: RBF embedding of base orientation scores

    The embeddings are combined multiplicatively before computing
    GNM variances.
    """

    def __init__(
        self,
        dim: int = 32,
        out_channels: int = 2,
        gnm_channels: int = 4,
        layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self._init_kwargs = {
            "dim": dim, "out_channels": out_channels,
            "gnm_channels": gnm_channels, "layers": layers,
            "dropout": dropout,
        }

        k = gnm_channels

        # Distance pathway: (N, N) -> (N, N, k)
        self.rbf1 = RadialBasisFunctions(dim)
        self.linear1 = DenseNetwork(dim, k, [dim] * layers)

        # Orientation pathway: (N, N) -> (N, N, k)
        self.rbf2 = RadialBasisFunctions(dim)
        self.linear2 = DenseNetwork(dim, k, [dim] * layers)

        # Project GNM variances to output channels: (N, k) -> (N, out_channels)
        self.out_proj = nn.Linear(k, out_channels)

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
            Predicted reactivity, shape (N, out_channels)
        """
        dists = torch.cdist(coords, coords)

        # (N, N) -> (N, N, k)
        emb = self.linear1(self.rbf1(dists))

        if frames is not None:
            scores = _orientation_score(frames)
            emb = emb * self.linear2(self.rbf2(scores))

        # (N, N, k) -> (k, N, N) for batched GNM
        emb = emb.permute(2, 0, 1)
        variances = _gnm_variances(emb)  # (k, N)
        variances = variances.permute(1, 0)  # (N, k)

        return self.out_proj(variances)  # (N, out_channels)

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
        path: str,
    ) -> SGNM:
        """
        Load model from checkpoint.

        Args:
            path: Path to checkpoint file (as saved by Trainer).

        Returns:
            Loaded SGNM model.
        """
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        kwargs = checkpoint.get("init_kwargs", checkpoint.get("config", {}))
        model = cls(**kwargs)
        model.load_state_dict(state_dict)
        return model
