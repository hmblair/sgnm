"""
Configuration dataclasses for the VAE module.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class VAEConfig:
    """Configuration for Structure VAE architecture."""

    hidden_dim: int = 128
    """Hidden dimension for node features in EGNN layers."""

    latent_dim: int = 32
    """Dimension of per-residue latent vectors."""

    num_encoder_layers: int = 4
    """Number of EGNN layers in encoder."""

    num_decoder_layers: int = 4
    """Number of EGNN layers in decoder."""

    num_rbf: int = 16
    """Number of radial basis functions for distance encoding."""

    cutoff: float = 10.0
    """Distance cutoff for graph edges (Angstroms)."""

    kl_weight: float = 0.001
    """Weight for KL divergence term (beta-VAE parameter)."""

    kl_warmup_epochs: int = 10
    """Number of epochs to warmup KL weight from 0 to target."""

    use_frames: bool = False
    """Whether to use local coordinate frames as additional input features."""

    center_coords: bool = True
    """Whether to center coordinates for translation invariance."""

    coord_update_init_scale: float = 1e-3
    """Scale for initializing coordinate update weights (smaller = more stable)."""
