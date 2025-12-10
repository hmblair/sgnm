"""
E(3)-equivariant encoder for the Structure VAE.

The encoder maps RNA 3D coordinates to per-residue latent distributions (mu, logvar).
The latent space is E(3)-invariant.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VAEConfig
from .layers import RadialBasisExpansion, EGNNConv, build_radius_graph


class E3Encoder(nn.Module):
    """
    E(3)-equivariant encoder that maps coordinates to latent distributions.

    The encoder produces per-residue latent vectors that are invariant to
    rotations and translations of the input structure.
    """

    def __init__(self, config: VAEConfig) -> None:
        """
        Args:
            config: VAE configuration
        """
        super().__init__()
        self.config = config

        # Input projection for node types (4 nucleotide types)
        self.input_proj = nn.Linear(4, config.hidden_dim)

        # Optional: project local frame features
        if config.use_frames:
            # 6 invariant features from 3x3 frame: diag(3) + frob(1) + trace(1) + det(1)
            self.frame_proj = nn.Linear(6, config.hidden_dim)

        # RBF expansion for edge distances
        self.rbf = RadialBasisExpansion(config.num_rbf, config.cutoff)

        # EGNN layers (no coordinate updates - we only need invariant features)
        self.layers = nn.ModuleList(
            [
                EGNNConv(
                    config.hidden_dim,
                    config.num_rbf,
                    update_coords=False,  # Encoder doesn't update coords
                )
                for _ in range(config.num_encoder_layers)
            ]
        )

        # Output heads for distribution parameters
        self.mu_head = nn.Linear(config.hidden_dim, config.latent_dim)
        self.logvar_head = nn.Linear(config.hidden_dim, config.latent_dim)

        # Initialize logvar head to output small values initially
        with torch.no_grad():
            self.logvar_head.weight.mul_(0.01)
            self.logvar_head.bias.fill_(-2.0)  # exp(-2) ≈ 0.135 std

    def _extract_frame_features(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract rotation-invariant features from local coordinate frames.

        Args:
            frames: (N, 3, 3) rotation matrices defining local frames

        Returns:
            (N, 6) invariant features
        """
        # Diagonal elements
        diag = torch.diagonal(frames, dim1=-2, dim2=-1)  # (N, 3)

        # Frobenius norm (should be ~sqrt(3) for orthonormal frames)
        frob = torch.sqrt((frames**2).sum(dim=(-2, -1), keepdim=True)).squeeze(
            -1
        )  # (N, 1)

        # Trace
        trace = torch.einsum("nii->n", frames).unsqueeze(-1)  # (N, 1)

        # Determinant (should be +1 for proper rotations)
        det = torch.linalg.det(frames).unsqueeze(-1)  # (N, 1)

        return torch.cat([diag, frob, trace, det], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        node_types: torch.Tensor,
        frames: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode structure to latent distribution parameters.

        Args:
            x: (N, 3) residue coordinates
            node_types: (N,) nucleotide type indices (0-3: A, C, G, U)
            frames: (N, 3, 3) optional local coordinate frames

        Returns:
            mu: (N, latent_dim) mean of latent distribution
            logvar: (N, latent_dim) log variance of latent distribution
        """
        # Center coordinates for translation invariance
        if self.config.center_coords:
            x = x - x.mean(dim=0, keepdim=True)

        # Build graph based on distance cutoff
        edge_index = build_radius_graph(x, self.config.cutoff)

        # Compute edge features (RBF of distances)
        src, dst = edge_index
        if edge_index.size(1) > 0:
            edge_dists = (x[src] - x[dst]).norm(dim=-1)
            edge_attr = self.rbf(edge_dists)
        else:
            # Handle case with no edges (very small molecule)
            edge_attr = torch.zeros(0, self.config.num_rbf, device=x.device)

        # Initial node features from nucleotide type
        node_onehot = F.one_hot(node_types.long(), num_classes=4).float()
        h = self.input_proj(node_onehot)

        # Add frame features if available
        if self.config.use_frames and frames is not None:
            frame_feat = self._extract_frame_features(frames)
            h = h + self.frame_proj(frame_feat)

        # Apply EGNN layers
        for layer in self.layers:
            h, _ = layer(h, x, edge_index, edge_attr)

        # Output distribution parameters
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-10, max=2)

        return mu, logvar


class AllAtomE3Encoder(nn.Module):
    """
    E(3)-equivariant encoder for all-atom structures.

    Uses atom-level EGNN followed by attention-based pooling to produce
    per-residue latent distributions.
    """

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.config = config

        # Atom type embedding (149 types from ciffy)
        self.atom_embed = nn.Embedding(config.num_atom_types, config.hidden_dim)

        # Residue type embedding for conditioning (4 nucleotide types)
        self.residue_embed = nn.Embedding(4, config.hidden_dim)

        # RBF expansion for atom-level edges
        self.rbf = RadialBasisExpansion(config.num_rbf, config.atom_cutoff)

        # Atom-level EGNN layers (no coord updates - invariant features only)
        self.atom_layers = nn.ModuleList(
            [
                EGNNConv(
                    config.hidden_dim,
                    config.num_rbf,
                    update_coords=False,
                )
                for _ in range(config.num_atom_encoder_layers)
            ]
        )

        # Attention pooling: atoms → residues
        self.query_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.key_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.value_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.attn_scale = config.hidden_dim ** -0.5

        # Output heads for per-residue distribution parameters
        self.mu_head = nn.Linear(config.hidden_dim, config.latent_dim)
        self.logvar_head = nn.Linear(config.hidden_dim, config.latent_dim)

        # Initialize logvar head for small initial variance
        with torch.no_grad():
            self.logvar_head.weight.mul_(0.01)
            self.logvar_head.bias.fill_(-2.0)

    def _attention_pool(
        self,
        atom_features: torch.Tensor,
        residue_features: torch.Tensor,
        residue_indices: torch.Tensor,
        atoms_per_residue: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool atom features to residues using attention.

        Args:
            atom_features: (A, hidden_dim) atom features
            residue_features: (N, hidden_dim) residue embeddings (query)
            residue_indices: (A,) maps atom to residue
            atoms_per_residue: (N,) number of atoms per residue

        Returns:
            (N, hidden_dim) pooled residue features
        """
        N = residue_features.size(0)
        device = atom_features.device

        # Project queries, keys, values
        queries = self.query_proj(residue_features)  # (N, hidden)
        keys = self.key_proj(atom_features)  # (A, hidden)
        values = self.value_proj(atom_features)  # (A, hidden)

        # Compute attention for each residue
        pooled = torch.zeros_like(residue_features)

        # Process each residue
        offset = 0
        for res_idx in range(N):
            n_atoms = atoms_per_residue[res_idx].item()
            if n_atoms == 0:
                continue

            # Get atoms for this residue
            q = queries[res_idx:res_idx+1]  # (1, hidden)
            k = keys[offset:offset+n_atoms]  # (n_atoms, hidden)
            v = values[offset:offset+n_atoms]  # (n_atoms, hidden)

            # Attention scores
            scores = (q @ k.T) * self.attn_scale  # (1, n_atoms)
            weights = F.softmax(scores, dim=-1)  # (1, n_atoms)

            # Weighted sum
            pooled[res_idx] = (weights @ v).squeeze(0)

            offset += n_atoms

        return pooled

    def forward(
        self,
        atom_coords: torch.Tensor,
        atom_types: torch.Tensor,
        residue_indices: torch.Tensor,
        residue_types: torch.Tensor,
        atoms_per_residue: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode all-atom structure to per-residue latent distribution.

        Args:
            atom_coords: (A, 3) all-atom coordinates
            atom_types: (A,) atom type indices
            residue_indices: (A,) maps atom to residue index
            residue_types: (N,) nucleotide type indices
            atoms_per_residue: (N,) atoms per residue

        Returns:
            mu: (N, latent_dim) mean of latent distribution
            logvar: (N, latent_dim) log variance
        """
        A = atom_coords.size(0)
        N = residue_types.size(0)

        # Center coordinates
        if self.config.center_coords:
            atom_coords = atom_coords - atom_coords.mean(dim=0, keepdim=True)

        # Build atom-level graph
        edge_index = build_radius_graph(atom_coords, self.config.atom_cutoff)

        # Compute edge features
        src, dst = edge_index
        if edge_index.size(1) > 0:
            edge_dists = (atom_coords[src] - atom_coords[dst]).norm(dim=-1)
            edge_attr = self.rbf(edge_dists)
        else:
            edge_attr = torch.zeros(0, self.config.num_rbf, device=atom_coords.device)

        # Initial atom features from atom type embedding
        h = self.atom_embed(atom_types)  # (A, hidden)

        # Add residue embedding to each atom
        residue_emb = self.residue_embed(residue_types)  # (N, hidden)
        h = h + residue_emb[residue_indices]  # (A, hidden)

        # Apply atom-level EGNN layers
        for layer in self.atom_layers:
            h, _ = layer(h, atom_coords, edge_index, edge_attr)

        # Attention pooling: atoms → residues
        residue_query = self.residue_embed(residue_types)  # (N, hidden)
        residue_features = self._attention_pool(
            h, residue_query, residue_indices, atoms_per_residue
        )

        # Output distribution parameters
        mu = self.mu_head(residue_features)
        logvar = self.logvar_head(residue_features)
        logvar = torch.clamp(logvar, min=-10, max=2)

        return mu, logvar
