"""
VAE module for RNA structure generation.

This module provides an E(3)-equivariant Variational Autoencoder
for learning latent representations of RNA 3D structures.

Includes both residue-center and all-atom variants.
"""

from .config import VAEConfig
from .model import StructureVAE, AllAtomStructureVAE
from .losses import (
    VAELoss,
    AllAtomVAELoss,
    kabsch_rmsd,
    kl_divergence,
    per_residue_rmsd,
)
from .training import (
    VAETrainConfig,
    VAETrainer,
    AllAtomVAETrainer,
    train_vae,
    StructureSample,
    AllAtomStructureSample,
)
from .data import (
    StructureOnlyDataset,
    StructureListDataset,
    StructureDataConfig,
    AllAtomDataset,
)

__all__ = [
    # Config
    "VAEConfig",
    "VAETrainConfig",
    "StructureDataConfig",
    # Models
    "StructureVAE",
    "AllAtomStructureVAE",
    # Losses
    "VAELoss",
    "AllAtomVAELoss",
    "kabsch_rmsd",
    "kl_divergence",
    "per_residue_rmsd",
    # Training
    "VAETrainer",
    "AllAtomVAETrainer",
    "train_vae",
    "StructureSample",
    "AllAtomStructureSample",
    # Data
    "StructureOnlyDataset",
    "StructureListDataset",
    "AllAtomDataset",
]
