"""
Main StructureVAE model combining encoder and decoder.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .config import VAEConfig
from .encoder import E3Encoder
from .decoder import E3Decoder


class StructureVAE(nn.Module):
    """
    E(3)-Equivariant Variational Autoencoder for RNA structures.

    This model encodes 3D RNA structures into per-residue latent vectors
    and decodes them back to 3D coordinates. The latent space is E(3)-invariant,
    while the decoder output is E(3)-equivariant.

    Example usage:
        config = VAEConfig(hidden_dim=128, latent_dim=32)
        model = StructureVAE(config)

        # Training
        outputs = model(coords, node_types, frames)
        loss = compute_loss(outputs['recon'], coords, outputs['mu'], outputs['logvar'])

        # Sampling
        new_coords = model.sample(num_residues=100)

        # Interpolation
        interpolated = model.interpolate(coords1, coords2, node_types)
    """

    def __init__(self, config: VAEConfig) -> None:
        """
        Args:
            config: VAE configuration
        """
        super().__init__()
        self.config = config
        self.encoder = E3Encoder(config)
        self.decoder = E3Decoder(config)

    def encode(
        self,
        coords: torch.Tensor,
        node_types: torch.Tensor,
        frames: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode coordinates to latent distribution parameters.

        Args:
            coords: (N, 3) residue coordinates
            node_types: (N,) nucleotide type indices
            frames: (N, 3, 3) optional local coordinate frames

        Returns:
            mu: (N, latent_dim) mean
            logvar: (N, latent_dim) log variance
        """
        return self.encoder(coords, node_types, frames)

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution.

        z = mu + std * epsilon, where epsilon ~ N(0, I)

        Args:
            mu: (N, latent_dim) mean
            logvar: (N, latent_dim) log variance

        Returns:
            z: (N, latent_dim) sampled latent vectors
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(
        self,
        z: torch.Tensor,
        anchor_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Decode latent vectors to coordinates.

        Args:
            z: (N, latent_dim) latent vectors
            anchor_coords: (N, 3) optional anchor coordinates

        Returns:
            (N, 3) reconstructed coordinates
        """
        return self.decoder(z, anchor_coords)

    def forward(
        self,
        coords: torch.Tensor,
        node_types: torch.Tensor,
        frames: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass: encode, sample, decode.

        Args:
            coords: (N, 3) residue coordinates
            node_types: (N,) nucleotide type indices
            frames: (N, 3, 3) optional local coordinate frames

        Returns:
            Dictionary with keys:
                - 'recon': (N, 3) reconstructed coordinates
                - 'mu': (N, latent_dim) latent mean
                - 'logvar': (N, latent_dim) latent log variance
                - 'z': (N, latent_dim) sampled latent vectors
        """
        # Encode
        mu, logvar = self.encode(coords, node_types, frames)

        # Sample (use reparameterization during training, mean at inference)
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu

        # Decode (use input coords as anchor for equivariance)
        recon = self.decode(z, anchor_coords=coords)

        return {
            "recon": recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

    @torch.no_grad()
    def sample(
        self,
        num_residues: int,
        anchor_coords: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample new structure from the prior distribution.

        Args:
            num_residues: Number of residues to generate
            anchor_coords: (N, 3) optional anchor coordinates
            temperature: Sampling temperature (scales std of prior)

        Returns:
            (N, 3) sampled coordinates
        """
        device = next(self.parameters()).device

        # Sample from prior N(0, I)
        z = (
            torch.randn(num_residues, self.config.latent_dim, device=device)
            * temperature
        )

        return self.decode(z, anchor_coords)

    @torch.no_grad()
    def interpolate(
        self,
        coords1: torch.Tensor,
        coords2: torch.Tensor,
        node_types: torch.Tensor,
        frames1: torch.Tensor | None = None,
        frames2: torch.Tensor | None = None,
        num_steps: int = 10,
    ) -> list[torch.Tensor]:
        """
        Interpolate between two structures in latent space.

        Args:
            coords1: (N, 3) first structure coordinates
            coords2: (N, 3) second structure coordinates
            node_types: (N,) nucleotide types (assumed same for both)
            frames1: (N, 3, 3) optional frames for first structure
            frames2: (N, 3, 3) optional frames for second structure
            num_steps: Number of interpolation steps

        Returns:
            List of (N, 3) interpolated coordinate tensors
        """
        # Encode both structures
        mu1, _ = self.encode(coords1, node_types, frames1)
        mu2, _ = self.encode(coords2, node_types, frames2)

        # Linear interpolation in latent space
        interpolations = []
        for alpha in torch.linspace(0, 1, num_steps, device=mu1.device):
            z = (1 - alpha) * mu1 + alpha * mu2
            # Use first structure as anchor
            recon = self.decode(z, anchor_coords=coords1)
            interpolations.append(recon)

        return interpolations

    @torch.no_grad()
    def reconstruct(
        self,
        coords: torch.Tensor,
        node_types: torch.Tensor,
        frames: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Reconstruct coordinates through the VAE (encode then decode with mean).

        Args:
            coords: (N, 3) input coordinates
            node_types: (N,) nucleotide types
            frames: (N, 3, 3) optional local frames

        Returns:
            (N, 3) reconstructed coordinates
        """
        mu, _ = self.encode(coords, node_types, frames)
        return self.decode(mu, anchor_coords=coords)

    def save(self, path: str) -> None:
        """
        Save model weights and config.

        Args:
            path: Path to save file
        """
        torch.save(
            {
                "config": self.config,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "StructureVAE":
        """
        Load model from saved file.

        Args:
            path: Path to saved file
            device: Device to load model to

        Returns:
            Loaded StructureVAE model
        """
        checkpoint = torch.load(path, map_location=device)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model
