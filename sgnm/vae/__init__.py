"""
VAE module for RNA structure generation.

This module provides an E(3)-equivariant Variational Autoencoder
for learning latent representations of RNA 3D structures.
"""

from .config import VAEConfig
from .model import StructureVAE
from .losses import VAELoss, kabsch_rmsd, kl_divergence
from .training import VAETrainConfig, VAETrainer, train_vae, StructureSample
from .data import StructureOnlyDataset, StructureListDataset, StructureDataConfig

__all__ = [
    # Config
    "VAEConfig",
    "VAETrainConfig",
    "StructureDataConfig",
    # Model
    "StructureVAE",
    # Losses
    "VAELoss",
    "kabsch_rmsd",
    "kl_divergence",
    # Training
    "VAETrainer",
    "train_vae",
    "StructureSample",
    # Data
    "StructureOnlyDataset",
    "StructureListDataset",
]
