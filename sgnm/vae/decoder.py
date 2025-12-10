"""
E(3)-equivariant decoder for the Structure VAE.

The decoder maps per-residue latent vectors back to 3D coordinates.
It maintains E(3) equivariance by using anchor coordinates for initialization
and updating coordinates equivariantly via EGNN layers.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .config import VAEConfig
from .layers import (
    RadialBasisExpansion,
    EGNNConv,
    build_radius_graph,
    build_sequential_edges,
)


class E3Decoder(nn.Module):
    """
    E(3)-equivariant decoder that maps latent vectors to coordinates.

    The decoder takes per-residue latent vectors and reconstructs 3D coordinates.
    It requires anchor coordinates to establish a reference frame, ensuring
    the output is equivariant to transformations of the input.
    """

    def __init__(self, config: VAEConfig) -> None:
        """
        Args:
            config: VAE configuration
        """
        super().__init__()
        self.config = config

        # Project latent to hidden dimension
        self.latent_proj = nn.Linear(config.latent_dim, config.hidden_dim)

        # RBF expansion for edge distances
        self.rbf = RadialBasisExpansion(config.num_rbf, config.cutoff)

        # EGNN layers WITH coordinate updates
        self.layers = nn.ModuleList(
            [
                EGNNConv(
                    config.hidden_dim,
                    config.num_rbf,
                    update_coords=True,
                    coord_init_scale=config.coord_update_init_scale,
                )
                for _ in range(config.num_decoder_layers)
            ]
        )

    def _initialize_coords(
        self,
        N: int,
        anchor_coords: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Initialize coordinates for decoding.

        Args:
            N: Number of residues
            anchor_coords: (N, 3) optional anchor coordinates
            device: Device for tensor

        Returns:
            (N, 3) initial coordinates
        """
        if anchor_coords is not None:
            # Use provided anchors, centered
            x = anchor_coords - anchor_coords.mean(dim=0, keepdim=True)
        else:
            # Initialize as linear chain (canonical starting point)
            # Typical RNA backbone spacing is ~5.9 Angstroms between phosphates
            x = torch.zeros(N, 3, device=device)
            x[:, 0] = torch.arange(N, device=device).float() * 5.9
            x = x - x.mean(dim=0, keepdim=True)

        return x

    def _build_graph(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build graph with both radius and sequential edges.

        Args:
            x: (N, 3) coordinates

        Returns:
            edge_index: (2, E) edge indices
            edge_attr: (E, num_rbf) edge features
        """
        N = x.size(0)
        device = x.device

        # Radius-based edges
        radius_edges = build_radius_graph(x, self.config.cutoff)

        # Sequential backbone edges (always connected)
        seq_edges = build_sequential_edges(N, device)

        # Combine edges (may have duplicates, but that's okay)
        if radius_edges.size(1) > 0 and seq_edges.size(1) > 0:
            edge_index = torch.cat([radius_edges, seq_edges], dim=1)
        elif radius_edges.size(1) > 0:
            edge_index = radius_edges
        else:
            edge_index = seq_edges

        # Compute edge features
        if edge_index.size(1) > 0:
            src, dst = edge_index
            edge_dists = (x[src] - x[dst]).norm(dim=-1)
            edge_attr = self.rbf(edge_dists)
        else:
            edge_attr = torch.zeros(0, self.config.num_rbf, device=device)

        return edge_index, edge_attr

    def forward(
        self,
        z: torch.Tensor,
        anchor_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Decode latent vectors to coordinates.

        Args:
            z: (N, latent_dim) per-residue latent vectors
            anchor_coords: (N, 3) optional anchor coordinates for initialization

        Returns:
            (N, 3) reconstructed coordinates
        """
        N = z.size(0)
        device = z.device

        # Project latent to hidden dimension
        h = self.latent_proj(z)

        # Initialize coordinates
        x = self._initialize_coords(N, anchor_coords, device)

        # Iteratively refine coordinates with EGNN
        for layer in self.layers:
            # Rebuild graph at each layer (distances change as coords update)
            edge_index, edge_attr = self._build_graph(x)
            h, x = layer(h, x, edge_index, edge_attr)

        return x
