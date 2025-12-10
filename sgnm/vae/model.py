"""
Main StructureVAE model combining encoder and decoder.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import torch
import torch.nn as nn

from .config import VAEConfig
from .encoder import E3Encoder, AllAtomE3Encoder
from .decoder import E3Decoder, AllAtomE3Decoder

if TYPE_CHECKING:
    from .training import AllAtomStructureSample


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


class AllAtomStructureVAE(nn.Module):
    """
    E(3)-Equivariant VAE for all-atom RNA structures.

    Uses hierarchical encoding (atoms → residues) and decoding (residues → atoms)
    with per-residue latent vectors. Supports saving reconstructed structures
    as CIF files using ciffy.

    Example usage:
        config = VAEConfig(hidden_dim=128, latent_dim=32)
        model = AllAtomStructureVAE(config)

        # Training
        outputs = model(sample)
        loss = compute_loss(outputs['recon'], sample.atom_coords, ...)

        # Reconstruction with Polymer output
        polymer = model.reconstruct_polymer(sample)
        polymer.write("reconstructed.cif")

        # Ball sampling around reference
        polymers = model.sample_around_reference(sample, num_samples=10)
    """

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = AllAtomE3Encoder(config)
        self.decoder = AllAtomE3Decoder(config)

    def encode(
        self,
        sample: AllAtomStructureSample,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode all-atom structure to per-residue latent distribution.

        Args:
            sample: AllAtomStructureSample

        Returns:
            mu: (N, latent_dim) mean
            logvar: (N, latent_dim) log variance
        """
        return self.encoder(
            sample.atom_coords,
            sample.atom_types,
            sample.residue_indices,
            sample.residue_types,
            sample.atoms_per_residue,
        )

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(
        self,
        z: torch.Tensor,
        sample: AllAtomStructureSample,
        anchor_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Decode per-residue latents to all-atom coordinates.

        Args:
            z: (N, latent_dim) latent vectors
            sample: AllAtomStructureSample (provides structure info)
            anchor_coords: (N, 3) optional residue anchor coordinates

        Returns:
            (A, 3) all-atom coordinates
        """
        return self.decoder(
            z,
            sample.residue_types,
            sample.atom_types,
            sample.residue_indices,
            sample.atoms_per_residue,
            anchor_coords,
        )

    def _compute_residue_centers(
        self,
        atom_coords: torch.Tensor,
        residue_indices: torch.Tensor,
        atoms_per_residue: torch.Tensor,
    ) -> torch.Tensor:
        """Compute residue centers from atom coordinates."""
        N = atoms_per_residue.size(0)
        device = atom_coords.device

        centers = torch.zeros(N, 3, device=device)
        offset = 0
        for res_idx in range(N):
            n_atoms = atoms_per_residue[res_idx].item()
            if n_atoms > 0:
                centers[res_idx] = atom_coords[offset:offset+n_atoms].mean(dim=0)
            offset += n_atoms

        return centers

    def forward(
        self,
        sample: AllAtomStructureSample,
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass: encode, sample, decode.

        Args:
            sample: AllAtomStructureSample

        Returns:
            Dictionary with keys:
                - 'recon': (A, 3) reconstructed atom coordinates
                - 'mu': (N, latent_dim) latent mean
                - 'logvar': (N, latent_dim) latent log variance
                - 'z': (N, latent_dim) sampled latent vectors
        """
        # Encode
        mu, logvar = self.encode(sample)

        # Sample
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu

        # Compute residue centers for anchor
        anchor_coords = self._compute_residue_centers(
            sample.atom_coords,
            sample.residue_indices,
            sample.atoms_per_residue,
        )

        # Decode
        recon = self.decode(z, sample, anchor_coords)

        return {
            "recon": recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

    @torch.no_grad()
    def reconstruct(
        self,
        sample: AllAtomStructureSample,
    ) -> torch.Tensor:
        """
        Reconstruct atom coordinates through the VAE.

        Args:
            sample: AllAtomStructureSample

        Returns:
            (A, 3) reconstructed atom coordinates
        """
        mu, _ = self.encode(sample)

        anchor_coords = self._compute_residue_centers(
            sample.atom_coords,
            sample.residue_indices,
            sample.atoms_per_residue,
        )

        return self.decode(mu, sample, anchor_coords)

    @torch.no_grad()
    def reconstruct_polymer(
        self,
        sample: AllAtomStructureSample,
    ):
        """
        Reconstruct structure and return as ciffy Polymer.

        Args:
            sample: AllAtomStructureSample (must have polymer attribute)

        Returns:
            ciffy.Polymer with reconstructed coordinates
        """
        if sample.polymer is None:
            raise ValueError("Sample must have polymer attribute for reconstruction")

        # Reconstruct coordinates
        recon_coords = self.reconstruct(sample)

        # Create new Polymer with updated coordinates
        return sample.polymer.with_coordinates(recon_coords)

    @torch.no_grad()
    def sample_around_reference(
        self,
        sample: AllAtomStructureSample,
        num_samples: int = 10,
        radius: float = 1.0,
    ) -> list:
        """
        Sample structures in a ball around a reference in latent space.

        Args:
            sample: Reference AllAtomStructureSample
            num_samples: Number of samples to generate
            radius: Radius of ball in latent space (std units)

        Returns:
            List of ciffy.Polymer objects with sampled coordinates
        """
        if sample.polymer is None:
            raise ValueError("Sample must have polymer attribute for sampling")

        # Encode reference
        mu, logvar = self.encode(sample)
        std = torch.exp(0.5 * logvar)

        # Compute anchor coordinates
        anchor_coords = self._compute_residue_centers(
            sample.atom_coords,
            sample.residue_indices,
            sample.atoms_per_residue,
        )

        # Sample in ball around mu
        polymers = []
        for _ in range(num_samples):
            # Sample random direction and distance
            eps = torch.randn_like(mu)
            eps = eps / eps.norm(dim=-1, keepdim=True)  # Unit direction
            distance = torch.rand(1, device=mu.device) ** (1/3)  # Uniform in ball
            z = mu + radius * std * eps * distance

            # Decode
            coords = self.decode(z, sample, anchor_coords)

            # Create Polymer with new coordinates
            poly = sample.polymer.with_coordinates(coords)
            polymers.append(poly)

        return polymers

    def save(self, path: str) -> None:
        """Save model weights and config."""
        torch.save(
            {
                "config": self.config,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "AllAtomStructureVAE":
        """Load model from saved file."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model
