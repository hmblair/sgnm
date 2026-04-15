"""SO(3)-equivariant transformer model for reactivity prediction.

Uses flash-eq's EquivariantTransformer with ciffy's PolymerEmbedding.
Requires flash-eq: pip install 'sgnm[equivariant]'
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from ciffy import Scale, Reduction
from ciffy.nn import PolymerEmbedding

if TYPE_CHECKING:
    from ciffy import Polymer


def build_knn_graph(
    coords: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build k-NN graph from coordinates.

    Args:
        coords: (N, 3) atom coordinates.
        k: Number of neighbors.

    Returns:
        src: (E,) source indices.
        dst: (E,) destination indices.
    """
    n = coords.shape[0]
    k = min(k, n - 1)

    diff = coords.unsqueeze(0) - coords.unsqueeze(1)
    dist_sq = (diff**2).sum(dim=-1)

    dist_sq.fill_diagonal_(float("inf"))
    _, indices = dist_sq.topk(k, dim=-1, largest=False)

    dst = torch.arange(n, device=coords.device).unsqueeze(1).expand(-1, k).reshape(-1)
    src = indices.reshape(-1)

    return src, dst


class EquivariantReactivityModel(nn.Module):
    """SO(3)-equivariant transformer for reactivity prediction.

    Uses flash-eq's EquivariantTransformer with ciffy's PolymerEmbedding
    to predict per-residue reactivity profiles from RNA 3D structure.
    """

    def __init__(
        self,
        embed_dim: int = 32,
        hidden_mult: int = 16,
        hidden_layers: int = 4,
        k_neighbors: int = 16,
        num_heads: int = 4,
        num_bins: int = 100,
        rank: int = 4,
        hidden_dim: int = 64,
        min_dist: float = 0.0,
        max_dist: float = 10.0,
        lvals: list[int] | None = None,
        dropout: float = 0.0,
        out_channels: int = 2,
    ):
        super().__init__()

        self._init_kwargs = {
            "embed_dim": embed_dim, "hidden_mult": hidden_mult,
            "hidden_layers": hidden_layers, "k_neighbors": k_neighbors,
            "num_heads": num_heads, "num_bins": num_bins, "rank": rank,
            "hidden_dim": hidden_dim, "min_dist": min_dist,
            "max_dist": max_dist, "lvals": lvals, "dropout": dropout,
            "out_channels": out_channels,
        }

        try:
            from flash_eq import Repr, Graph, EquivariantTransformer
        except ImportError as e:
            raise ImportError(
                "flash-eq is required for EquivariantReactivityModel. "
                "Install with: pip install 'sgnm[equivariant]'"
            ) from e

        if lvals is None:
            lvals = [0, 1]

        self.lvals = lvals
        self.k_neighbors = k_neighbors

        in_repr = Repr(lvals=[0], mult=embed_dim)
        hidden_repr = Repr(lvals=lvals, mult=hidden_mult)
        out_repr = Repr(lvals=[0], mult=hidden_mult)

        self.in_repr = in_repr
        self.hidden_repr = hidden_repr
        self.out_repr = out_repr

        atom_dim = embed_dim // 2
        element_dim = embed_dim // 4
        residue_dim = embed_dim - atom_dim - element_dim

        self.embedding = PolymerEmbedding(
            scale=Scale.ATOM,
            atom_dim=atom_dim if atom_dim > 0 else None,
            residue_dim=residue_dim if residue_dim > 0 else None,
            element_dim=element_dim if element_dim > 0 else None,
            dropout=dropout,
        )

        self.transformer = EquivariantTransformer(
            in_repr=in_repr,
            hidden_repr=hidden_repr,
            out_repr=out_repr,
            num_layers=hidden_layers,
            num_heads=num_heads,
            num_bins=num_bins,
            rank=rank,
            hidden_dim=hidden_dim,
            min_dist=min_dist,
            max_dist=max_dist,
            mlp_ratio=2,
            dropout=dropout,
        )

        self.out_proj = nn.Linear(hidden_mult, out_channels)
        self.repr_dim = in_repr.dim()

        self._Graph = Graph

    def forward(self, polymer: "Polymer") -> torch.Tensor:
        """Predict reactivity from polymer structure.

        Args:
            polymer: Input polymer (torch backend).

        Returns:
            (n_residues, out_channels) predicted reactivities.
        """
        polymer = polymer.torch()
        n_atoms = polymer.size(Scale.ATOM)

        if n_atoms == 0:
            return polymer.coordinates.new_zeros(0, self.out_proj.out_features)

        embed = self.embedding(polymer)
        scalar_features = embed.unsqueeze(-1)

        coords = polymer.coordinates
        src, dst = build_knn_graph(coords, self.k_neighbors)
        graph = self._Graph(src=src, dst=dst, num_nodes=coords.shape[0])

        output = self.transformer(coords, scalar_features, graph)

        atom_features = output.squeeze(-1)

        residue_features = polymer.reduce(atom_features, Scale.RESIDUE, Reduction.MEAN)

        return self.out_proj(residue_features)

    def ciffy(
        self,
        poly: "Polymer",
    ) -> torch.Tensor:
        """Predict reactivity from a ciffy Polymer object.

        Args:
            poly: RNA polymer structure.

        Returns:
            (n_residues, out_channels) predicted reactivities.
        """
        return self(poly)

    @classmethod
    def load(cls, path: str) -> "EquivariantReactivityModel":
        """Load model from checkpoint.

        Args:
            path: Path to checkpoint file (as saved by Trainer).

        Returns:
            Loaded EquivariantReactivityModel.
        """
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        kwargs = checkpoint.get("init_kwargs", {})
        model = cls(**kwargs)
        model.load_state_dict(state_dict)
        return model
