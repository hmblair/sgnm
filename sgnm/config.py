"""
Configuration dataclasses and constants for SGNM.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import torch
from ciffy.enum import Adenosine, Cytosine, Guanosine, Uridine


# =============================================================================
# Constants: Nucleobase frame atoms (C2-C4-C6)
# =============================================================================

FRAME1 = torch.tensor([
    Adenosine.C2.value,
    Cytosine.C2.value,
    Guanosine.C2.value,
    Uridine.C2.value,
])

FRAME2 = torch.tensor([
    Adenosine.C4.value,
    Cytosine.C4.value,
    Guanosine.C4.value,
    Uridine.C4.value,
])

FRAME3 = torch.tensor([
    Adenosine.C6.value,
    Cytosine.C6.value,
    Guanosine.C6.value,
    Uridine.C6.value,
])

FRAMES = torch.cat([FRAME1, FRAME2, FRAME3])

# Nucleotide vocabulary for sequence tokenization
NUCLEOTIDE_DICT = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3}


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for SGNM model architecture."""

    dim: int = 32
    """Dimension of radial basis functions and hidden layers."""

    out_dim: int = 1
    """Output dimension (1 for SHAPE, 2 for SHAPE+DMS)."""

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

    structures_dir: str = ""
    """Directory containing structural files (CIF/PDB)."""

    data_format: Literal["v1", "v2"] = "v2"
    """Format of the reactivity data file (v1: old format, v2: PDB130 format)."""

    offset: int = 51
    """Offset for aligning reactivity to structure (dataset-specific)."""

    trim_ends: int = 5
    """Number of residues to trim from each end (unreliable data)."""

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

    max_epochs: int = 100
    """Maximum number of training epochs."""

    eval_only: bool = False
    """If True, run validation only without optimization."""

    checkpoint_dir: str = "./checkpoints"
    """Directory for saving model checkpoints."""

    save_every: int = 10
    """Save checkpoint every N epochs."""

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

    profile_path: str | None = None
    """Path to HDF5 file with target profiles."""

    file_pattern: str = "*.cif"
    """Glob pattern for files to process."""

    recursive: bool = False
    """Whether to search subdirectories."""
