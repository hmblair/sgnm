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


class AllAtomE3Decoder(nn.Module):
    """
    E(3)-equivariant decoder for all-atom structures.

    Two-stage decoding:
    1. Residue-level EGNN to decode per-residue latents to residue centers
    2. Atom expansion: predict atom positions relative to residue centers
    3. Optional atom-level EGNN refinement
    """

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.config = config

        # Project latent to hidden dimension
        self.latent_proj = nn.Linear(config.latent_dim, config.hidden_dim)

        # Residue type embedding
        self.residue_embed = nn.Embedding(4, config.hidden_dim)

        # RBF for residue-level graph
        self.rbf = RadialBasisExpansion(config.num_rbf, config.cutoff)

        # Residue-level EGNN layers (with coord updates)
        self.residue_layers = nn.ModuleList(
            [
                EGNNConv(
                    config.hidden_dim,
                    config.num_rbf,
                    update_coords=True,
                    coord_init_scale=config.coord_update_init_scale,
                )
                for _ in range(config.num_residue_decoder_layers)
            ]
        )

        # Atom expansion: predict relative positions
        self.atom_embed = nn.Embedding(config.num_atom_types, config.hidden_dim)

        # MLP to predict relative position from (residue_feat, atom_embed)
        self.rel_pos_mlp = nn.Sequential(
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 3),
        )

        # Initialize relative position prediction to small values
        with torch.no_grad():
            self.rel_pos_mlp[-1].weight.mul_(0.1)
            self.rel_pos_mlp[-1].bias.zero_()

        # RBF for atom-level graph
        self.atom_rbf = RadialBasisExpansion(config.num_rbf, config.atom_cutoff)

        # Atom-level refinement EGNN layers (with coord updates)
        self.atom_layers = nn.ModuleList(
            [
                EGNNConv(
                    config.hidden_dim,
                    config.num_rbf,
                    update_coords=True,
                    coord_init_scale=config.coord_update_init_scale,
                )
                for _ in range(config.num_atom_decoder_layers)
            ]
        )

    def _initialize_residue_coords(
        self,
        N: int,
        anchor_coords: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        """Initialize residue center coordinates."""
        if anchor_coords is not None:
            x = anchor_coords - anchor_coords.mean(dim=0, keepdim=True)
        else:
            x = torch.zeros(N, 3, device=device)
            x[:, 0] = torch.arange(N, device=device).float() * 5.9
            x = x - x.mean(dim=0, keepdim=True)
        return x

    def _build_residue_graph(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build residue-level graph with radius and sequential edges."""
        N = x.size(0)
        device = x.device

        radius_edges = build_radius_graph(x, self.config.cutoff)
        seq_edges = build_sequential_edges(N, device)

        if radius_edges.size(1) > 0 and seq_edges.size(1) > 0:
            edge_index = torch.cat([radius_edges, seq_edges], dim=1)
        elif radius_edges.size(1) > 0:
            edge_index = radius_edges
        else:
            edge_index = seq_edges

        if edge_index.size(1) > 0:
            src, dst = edge_index
            edge_dists = (x[src] - x[dst]).norm(dim=-1)
            edge_attr = self.rbf(edge_dists)
        else:
            edge_attr = torch.zeros(0, self.config.num_rbf, device=device)

        return edge_index, edge_attr

    def _expand_to_atoms(
        self,
        residue_features: torch.Tensor,
        residue_centers: torch.Tensor,
        atom_types: torch.Tensor,
        residue_indices: torch.Tensor,
        atoms_per_residue: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Expand residue features to atom positions.

        Args:
            residue_features: (N, hidden_dim) residue features
            residue_centers: (N, 3) residue center coordinates
            atom_types: (A,) atom type indices
            residue_indices: (A,) maps atom to residue
            atoms_per_residue: (N,) atoms per residue

        Returns:
            atom_features: (A, hidden_dim) atom features
            atom_coords: (A, 3) atom coordinates
        """
        A = atom_types.size(0)
        N = residue_features.size(0)
        device = residue_features.device

        # Embed atom types
        atom_emb = self.atom_embed(atom_types)  # (A, hidden)

        # Expand residue features to atoms
        res_feat_expanded = residue_features[residue_indices]  # (A, hidden)

        # Combine for relative position prediction
        combined = torch.cat([res_feat_expanded, atom_emb], dim=-1)  # (A, 2*hidden)

        # Predict relative positions
        rel_pos = self.rel_pos_mlp(combined)  # (A, 3)

        # Expand residue centers to atoms and add relative positions
        res_centers_expanded = residue_centers[residue_indices]  # (A, 3)
        atom_coords = res_centers_expanded + rel_pos

        # Atom features for refinement
        atom_features = res_feat_expanded + atom_emb

        return atom_features, atom_coords

    def forward(
        self,
        z: torch.Tensor,
        residue_types: torch.Tensor,
        atom_types: torch.Tensor,
        residue_indices: torch.Tensor,
        atoms_per_residue: torch.Tensor,
        anchor_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Decode latent vectors to all-atom coordinates.

        Args:
            z: (N, latent_dim) per-residue latent vectors
            residue_types: (N,) nucleotide type indices
            atom_types: (A,) atom type indices
            residue_indices: (A,) maps atom to residue index
            atoms_per_residue: (N,) atoms per residue
            anchor_coords: (N, 3) optional anchor residue centers

        Returns:
            (A, 3) all-atom coordinates
        """
        N = z.size(0)
        A = atom_types.size(0)
        device = z.device

        # Project latent and add residue embedding
        h = self.latent_proj(z) + self.residue_embed(residue_types)

        # Initialize residue centers
        x = self._initialize_residue_coords(N, anchor_coords, device)

        # Residue-level EGNN to decode centers
        for layer in self.residue_layers:
            edge_index, edge_attr = self._build_residue_graph(x)
            h, x = layer(h, x, edge_index, edge_attr)

        # Expand to atoms
        atom_features, atom_coords = self._expand_to_atoms(
            h, x, atom_types, residue_indices, atoms_per_residue
        )

        # Atom-level refinement
        for layer in self.atom_layers:
            edge_index = build_radius_graph(atom_coords, self.config.atom_cutoff)

            if edge_index.size(1) > 0:
                src, dst = edge_index
                edge_dists = (atom_coords[src] - atom_coords[dst]).norm(dim=-1)
                edge_attr = self.atom_rbf(edge_dists)
            else:
                edge_attr = torch.zeros(0, self.config.num_rbf, device=device)

            atom_features, atom_coords = layer(
                atom_features, atom_coords, edge_index, edge_attr
            )

        return atom_coords
