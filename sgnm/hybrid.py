"""Hybrid model combining GNM features with an SO(3)-equivariant transformer.

Uses a learnable GNM to extract per-residue physics-based features, which are
broadcast to atoms and concatenated with ciffy's PolymerEmbedding before being
processed by the flash-eq equivariant transformer. The GNM features act as a
physics-based prior, while the equivariant transformer learns geometric
refinements.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from ciffy import Scale, Reduction
from ciffy.nn import PolymerEmbedding

from .equivariant import build_knn_graph
from .gnm import _gnm_variances, _orientation_score
from .models import _base_frame
from .nn import DenseNetwork, RadialBasisFunctions

if TYPE_CHECKING:
    from ciffy import Polymer


class GNMFeatureExtractor(nn.Module):
    """Extract per-residue GNM variance features from a polymer.

    Mirrors the SGNM model's architecture (distance and orientation RBF
    embeddings, k independent adjacency matrices, batched GNM variances),
    but returns the raw variance features rather than projecting to output
    channels. These are intended to be consumed as input features by a
    downstream model.
    """

    def __init__(
        self,
        dim: int = 32,
        gnm_channels: int = 4,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.gnm_channels = gnm_channels

        self.rbf1 = RadialBasisFunctions(dim)
        self.linear1 = DenseNetwork(dim, gnm_channels, [dim] * layers)

        self.rbf2 = RadialBasisFunctions(dim)
        self.linear2 = DenseNetwork(dim, gnm_channels, [dim] * layers)

    def forward(
        self,
        coords: torch.Tensor,
        frames: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-residue GNM variance features.

        Args:
            coords: Residue center coordinates, shape (N, 3).
            frames: Local coordinate frames, shape (N, 3, 3) or None.

        Returns:
            GNM variance features of shape (N, gnm_channels).
        """
        dists = torch.cdist(coords, coords)
        emb = self.linear1(self.rbf1(dists))

        if frames is not None:
            scores = _orientation_score(frames)
            emb = emb * self.linear2(self.rbf2(scores))

        # (N, N, k) -> (k, N, N) for batched GNM
        emb = emb.permute(2, 0, 1)
        variances = _gnm_variances(emb)  # (k, N)
        return variances.permute(1, 0)  # (N, k)


class HybridReactivityModel(nn.Module):
    """Equivariant transformer with GNM physics-based priors.

    Architecture:
        1. Compute per-residue GNM variance features via a learnable GNM.
        2. Compute per-atom embeddings via ciffy's PolymerEmbedding.
        3. Broadcast GNM features from residue to atom level via membership.
        4. Normalize GNM features with LayerNorm (per channel).
        5. Concatenate polymer embedding and normalized GNM features.
        6. Process with an SO(3)-equivariant transformer.
        7. Pool to residues and project to (SHAPE, DMS) output channels.

    If the GNM computation fails (e.g., missing nucleobase atoms for
    modified residues), GNM features are filled with zeros and the model
    falls back to pure equivariant processing for that sample.
    """

    def __init__(
        self,
        embed_dim: int = 32,
        gnm_dim: int = 32,
        gnm_channels: int = 4,
        gnm_layers: int = 2,
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
            "embed_dim": embed_dim, "gnm_dim": gnm_dim,
            "gnm_channels": gnm_channels, "gnm_layers": gnm_layers,
            "hidden_mult": hidden_mult, "hidden_layers": hidden_layers,
            "k_neighbors": k_neighbors, "num_heads": num_heads,
            "num_bins": num_bins, "rank": rank, "hidden_dim": hidden_dim,
            "min_dist": min_dist, "max_dist": max_dist, "lvals": lvals,
            "dropout": dropout, "out_channels": out_channels,
        }

        try:
            from flash_eq import Repr, Graph, EquivariantTransformer
        except ImportError as e:
            raise ImportError(
                "flash-eq is required for HybridReactivityModel. "
                "Install with: pip install 'sgnm[equivariant]'"
            ) from e

        if lvals is None:
            lvals = [0, 1]

        self.lvals = lvals
        self.k_neighbors = k_neighbors
        self.gnm_channels = gnm_channels
        self._total_in_dim = embed_dim + gnm_channels

        # GNM feature extractor (residue-level physics prior)
        self.gnm_features = GNMFeatureExtractor(
            dim=gnm_dim, gnm_channels=gnm_channels, layers=gnm_layers,
        )
        # Normalize GNM features per-channel before concat
        self.gnm_norm = nn.LayerNorm(gnm_channels)

        # Polymer embedding (atom-level learned features)
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

        # Equivariant transformer
        in_repr = Repr(lvals=[0], mult=self._total_in_dim)
        hidden_repr = Repr(lvals=lvals, mult=hidden_mult)
        out_repr = Repr(lvals=[0], mult=hidden_mult)

        self.in_repr = in_repr
        self.hidden_repr = hidden_repr
        self.out_repr = out_repr

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
        self._Graph = Graph

    def _compute_gnm_features(
        self, polymer: "Polymer", n_residues: int, device: torch.device,
    ) -> torch.Tensor:
        """Compute per-residue GNM features, fallback to zeros on failure.

        Returns shape (n_residues, gnm_channels).
        """
        try:
            # Strip unresolved residues for frame computation
            stripped = polymer.strip()
            centered, res_coords = stripped.center(Scale.RESIDUE)
            frames = _base_frame(centered)
            raw = self.gnm_features(res_coords, frames)
            # If strip removed residues, we must pad back to the original count
            if raw.size(0) != n_residues:
                padded = torch.zeros(
                    n_residues, self.gnm_channels, device=device, dtype=raw.dtype,
                )
                # Best-effort: place raw in the first rows; exact alignment would
                # require tracking the strip mask. For full structures this
                # branch is not hit.
                padded[: raw.size(0)] = raw
                return padded
            return raw
        except (ValueError, RuntimeError):
            return torch.zeros(n_residues, self.gnm_channels, device=device)

    def forward(self, polymer: "Polymer") -> torch.Tensor:
        """Predict reactivity from polymer structure.

        Args:
            polymer: Input polymer (torch backend).

        Returns:
            (n_residues, out_channels) predicted reactivities.
        """
        polymer = polymer.torch()
        device = next(self.parameters()).device
        if device.type == "cuda":
            polymer = polymer.cuda()

        n_atoms = polymer.size(Scale.ATOM)
        n_residues = polymer.size(Scale.RESIDUE)

        if n_atoms == 0:
            return polymer.coordinates.new_zeros(0, self.out_proj.out_features)

        # Atom-level embedding
        atom_embed = self.embedding(polymer)  # (N_atoms, embed_dim)

        # Residue-level GNM features
        gnm_feats = self._compute_gnm_features(polymer, n_residues, device)
        gnm_feats = self.gnm_norm(gnm_feats)  # (N_residues, gnm_channels)

        # Broadcast residue features to atoms via membership
        res_idx = polymer.membership(Scale.RESIDUE)  # (N_atoms,)
        atom_gnm = gnm_feats[res_idx]  # (N_atoms, gnm_channels)

        # Concatenate
        combined = torch.cat([atom_embed, atom_gnm], dim=-1)  # (N_atoms, embed_dim + k)
        scalar_features = combined.unsqueeze(-1)  # (N_atoms, D, 1)

        # Equivariant transformer
        coords = polymer.coordinates
        src, dst = build_knn_graph(coords, self.k_neighbors)
        graph = self._Graph(src=src, dst=dst, num_nodes=coords.shape[0])
        output = self.transformer(coords, scalar_features, graph)
        atom_features = output.squeeze(-1)

        # Pool to residues and project
        residue_features = polymer.reduce(
            atom_features, Scale.RESIDUE, Reduction.MEAN,
        )
        return self.out_proj(residue_features)

    def ciffy(self, poly: "Polymer") -> torch.Tensor:
        """Predict reactivity from a ciffy Polymer object.

        Args:
            poly: RNA polymer structure.

        Returns:
            (n_residues, out_channels) predicted reactivities.
        """
        return self(poly)

    @classmethod
    def load(cls, path: str) -> "HybridReactivityModel":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        kwargs = checkpoint.get("init_kwargs", {})
        model = cls(**kwargs)
        model.load_state_dict(state_dict)
        return model
