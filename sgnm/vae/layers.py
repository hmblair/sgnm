"""
E(3)-equivariant graph neural network layers for the VAE.

Implements EGNN (E(n) Equivariant Graph Neural Networks) which update
both node features (invariant) and coordinates (equivariant).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class RadialBasisExpansion(nn.Module):
    """
    Expand scalar distances into radial basis function features.

    Uses fixed Gaussian centers evenly spaced from 0 to cutoff.
    This is similar to the learnable RadialBasisFunctions in nn.py
    but with fixed parameters for stability.
    """

    def __init__(self, num_rbf: int, cutoff: float) -> None:
        """
        Args:
            num_rbf: Number of radial basis functions
            cutoff: Maximum distance (determines RBF spacing)
        """
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff

        # Fixed Gaussian centers from 0 to cutoff
        centers = torch.linspace(0, cutoff, num_rbf)
        self.register_buffer("centers", centers)

        # Width based on spacing
        self.width = (cutoff / num_rbf) * 0.5

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Expand distances to RBF features.

        Args:
            distances: (...,) tensor of distances

        Returns:
            (..., num_rbf) tensor of RBF features
        """
        # Expand last dimension for broadcasting
        d = distances.unsqueeze(-1)  # (..., 1)
        return torch.exp(-((d - self.centers) ** 2) / (2 * self.width**2))


class EGNNConv(nn.Module):
    """
    Single E(n) Equivariant Graph Neural Network convolution layer.

    Updates both node features (invariant) and coordinates (equivariant):
        1. Compute edge messages: m_ij = phi_e(h_i, h_j, ||x_i - x_j||^2, edge_attr)
        2. Update coordinates: x_i' = x_i + sum_j (x_i - x_j) * phi_x(m_ij)
        3. Aggregate messages: m_i = sum_j m_ij
        4. Update nodes: h_i' = h_i + phi_h(h_i, m_i)

    Reference: Satorras et al., "E(n) Equivariant Graph Neural Networks" (2021)
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        update_coords: bool = True,
        coord_init_scale: float = 1e-3,
    ) -> None:
        """
        Args:
            hidden_dim: Dimension of node features
            edge_dim: Dimension of edge features (e.g., RBF dimension)
            update_coords: Whether to update coordinates (False for encoder)
            coord_init_scale: Scale for coordinate update weight initialization
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.update_coords = update_coords

        # Edge MLP: takes (h_i, h_j, ||x_i - x_j||^2, edge_attr)
        edge_input_dim = 2 * hidden_dim + 1 + edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Coordinate update (outputs scalar weight per edge)
        if update_coords:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1, bias=False),
            )
            # Initialize to small values for stable training
            with torch.no_grad():
                self.coord_mlp[-1].weight.mul_(coord_init_scale)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            h: (N, hidden_dim) node features
            x: (N, 3) node coordinates
            edge_index: (2, E) edge indices [source, target]
            edge_attr: (E, edge_dim) edge features

        Returns:
            h_out: (N, hidden_dim) updated node features
            x_out: (N, 3) updated coordinates
        """
        src, dst = edge_index  # Source and destination node indices

        # Compute relative positions and squared distances
        rel_pos = x[src] - x[dst]  # (E, 3)
        dist_sq = (rel_pos**2).sum(dim=-1, keepdim=True)  # (E, 1)

        # Edge message computation
        edge_input = torch.cat([h[src], h[dst], dist_sq, edge_attr], dim=-1)
        m_ij = self.edge_mlp(edge_input)  # (E, hidden_dim)

        # Coordinate update (equivariant)
        if self.update_coords:
            coord_weights = self.coord_mlp(m_ij)  # (E, 1)
            coord_update = rel_pos * coord_weights  # (E, 3)

            # Aggregate coordinate updates to destination nodes
            x_out = x.clone()
            x_out.index_add_(0, dst, coord_update)
        else:
            x_out = x

        # Aggregate messages to nodes
        m_i = torch.zeros_like(h)
        m_i.index_add_(0, dst, m_ij)

        # Node update (invariant) with residual connection
        h_out = h + self.node_mlp(torch.cat([h, m_i], dim=-1))

        return h_out, x_out


def build_radius_graph(
    x: torch.Tensor,
    cutoff: float,
    include_self_loops: bool = False,
) -> torch.Tensor:
    """
    Build graph edges based on distance cutoff.

    Args:
        x: (N, 3) node coordinates
        cutoff: Distance cutoff for edges
        include_self_loops: Whether to include self-loop edges

    Returns:
        (2, E) edge index tensor
    """
    N = x.size(0)

    # Compute all pairwise distances
    dists = torch.cdist(x, x)  # (N, N)

    # Create edges for pairs within cutoff
    if include_self_loops:
        mask = dists < cutoff
    else:
        mask = (dists < cutoff) & (dists > 0)

    edge_index = mask.nonzero().t()  # (2, E)
    return edge_index


def build_sequential_edges(
    N: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build sequential (backbone) edges connecting adjacent residues.

    Args:
        N: Number of residues
        device: Device for tensor

    Returns:
        (2, 2*(N-1)) edge index tensor (bidirectional)
    """
    if N <= 1:
        return torch.zeros(2, 0, dtype=torch.long, device=device)

    # Forward edges: i -> i+1
    src_fwd = torch.arange(N - 1, device=device)
    dst_fwd = torch.arange(1, N, device=device)

    # Backward edges: i+1 -> i
    src_bwd = dst_fwd
    dst_bwd = src_fwd

    src = torch.cat([src_fwd, src_bwd])
    dst = torch.cat([dst_fwd, dst_bwd])

    return torch.stack([src, dst])
