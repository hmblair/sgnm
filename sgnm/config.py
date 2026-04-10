"""
Configuration dataclasses for SGNM.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

@dataclass
class ModelConfig:
    """Configuration for SGNM model architecture."""

    dim: int = 32
    """Dimension of radial basis functions and hidden layers."""

    out_channels: int = 2
    """Number of output channels (e.g. 2 for SHAPE+DMS)."""

    gnm_channels: int = 4
    """Number of independent GNM adjacency matrices. Each produces a separate
    variance, then a linear layer projects to out_channels."""

    layers: int = 1
    """Number of hidden layers in the dense networks."""

    dropout: float = 0.0
    """Dropout rate for hidden layers."""

    weights_path: str | None = None
    """Path to pre-trained weights for loading."""


# =============================================================================
# Data Configuration
# =============================================================================

@dataclass
class DataConfig:
    """Configuration for dataset loading and preprocessing."""

    reactivity_path: str = ""
    """Path to HDF5 file containing reactivity profiles."""

    fasta_path: str = ""
    """Path to FASTA file containing probed sequences."""

    structures_dir: str = ""
    """Directory containing structural files (CIF/PDB)."""

    data_format: Literal["v1", "v2"] = "v2"
    """Format of the reactivity data file (v1: old format, v2: PDB130 format)."""

    max_chains: int = 2
    """Maximum number of chains per structure (filter out larger complexes)."""

    train_split: float = 0.8
    """Fraction of data for training."""

    val_split: float = 0.1
    """Fraction of data for validation."""

    test_split: float = 0.1
    """Fraction of data for testing."""

    seed: int = 42
    """Random seed for reproducible splits."""


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainConfig:
    """Configuration for training loop and optimization."""

    learning_rate: float = 1e-2
    """Learning rate for optimizer."""

    weight_decay: float = 0.0
    """Weight decay for AdamW. Set > 0 to use AdamW instead of Adam."""

    max_epochs: int = 100
    """Maximum number of training epochs."""

    loss_type: Literal["mae", "mse", "correlation"] = "mae"
    """Loss function to use."""

    warmup_epochs: float = 0.0
    """Number of warmup epochs for cosine schedule (0 to disable)."""

    min_lr_ratio: float = 0.1
    """Minimum LR as fraction of initial LR (for cosine schedule)."""

    checkpoint_dir: str = "./checkpoints"
    """Directory for saving model checkpoints."""

    save_best: bool = True
    """Save checkpoint when validation loss improves."""

    log_every: int = 10
    """Log metrics every N steps."""

    visualize_every: int = 0
    """Generate visualization plots every N steps (0 to disable)."""

    gradient_clip: float | None = 1.0
    """Maximum gradient norm for clipping (None to disable)."""

    patience: int = 10
    """Epochs without improvement before early stopping."""

    noise_std: float = 0.0
    """Standard deviation of Gaussian noise added to coordinates during
    training. Set > 0 for noise-augmented training. Units are Angstroms."""

    val_noise_std: float = 0.0
    """Standard deviation of Gaussian noise added during validation.
    Useful for evaluating robustness to structural perturbations."""

    device: str = "cpu"
    """Device to train on ('cpu', 'cuda', 'mps')."""

    wandb_project: str | None = None
    """Weights & Biases project name (None to disable)."""

    wandb_run: str | None = None
    """Weights & Biases run name."""


# =============================================================================
# Scoring Configuration
# =============================================================================

@dataclass
class ScoringConfig:
    """Configuration for scoring operations."""

    weights_path: str | None = None
    """Path to model weights for scoring."""

    blank_start: int | None = None
    """Number of residues to mask at sequence start."""

    blank_end: int | None = None
    """Number of residues to mask at sequence end."""

    metric: Literal["mae", "mse", "correlation"] = "mae"
    """Scoring metric to use."""


@dataclass
class FilterConfig:
    """Configuration for filtering predictions by score."""

    threshold: float = 0.5
    """Score threshold for filtering."""

    mode: Literal["below", "above"] = "below"
    """Filter mode: 'below' keeps scores < threshold (good for MAE)."""

    top_k: int | None = None
    """Alternative: keep top K best predictions."""


@dataclass
class RelaxConfig:
    """Configuration for structure relaxation."""

    steps: int = 100
    """Number of relaxation steps."""

    lr: float = 1e-3
    """Learning rate for coordinate optimization."""

    alpha: float = 1e-3
    """RMSD regularization weight."""


@dataclass
class BatchScoringConfig:
    """Configuration for batch scoring of .cif folders."""

    input_dir: str = ""
    """Directory containing .cif files to score."""

    output_dir: str | None = None
    """Directory for filtered output (if copying files)."""

    file_pattern: str = "*.cif"
    """Glob pattern for files to process."""

    recursive: bool = False
    """Whether to search subdirectories."""
