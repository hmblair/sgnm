"""
SGNM: Structure-Guided Normal Mode Model for RNA reactivity prediction.
"""

# =============================================================================
# Backward Compatible Exports (original API)
# =============================================================================
from .models import SGNM, BaseSGNM
from .scoring import Score as score

# =============================================================================
# New API
# =============================================================================

# Configuration
from .config import (
    ModelConfig,
    DataConfig,
    TrainConfig,
    ScoringConfig,
    FilterConfig,
    RelaxConfig,
    BatchScoringConfig,
    FRAME1,
    FRAME2,
    FRAME3,
    FRAMES,
    NUCLEOTIDE_DICT,
)

# Neural Network Layers
from .nn import RadialBasisFunctions, DenseNetwork

# Loss Functions and Schedulers
from .losses import pearson_correlation, mae_loss, mse_loss, correlation_loss
from .schedulers import get_cosine_schedule_with_warmup

# Scoring
from .scoring import (
    Score,
    StructureScorer,
    BatchScorer,
    StructureRelaxer,
    ScoringResult,
    RelaxResult,
    BatchScoringResults,
)

# Training
from .training import Trainer, train_sgnm, TrainResults

# Data
from .data import ReactivityDataset, Sample, ProfileLoader, load_reactivity_index

# =============================================================================
# Convenience Functions
# =============================================================================

def equivariant(**kwargs) -> "EquivariantReactivityModel":
    """Create an EquivariantReactivityModel. Requires flash-eq."""
    from .equivariant import EquivariantReactivityModel
    return EquivariantReactivityModel(**kwargs)


def load(path: str | None = None) -> BaseSGNM:
    """
    Load a model from weights file.

    Args:
        path: Path to weights file, or None for base model

    Returns:
        Loaded model
    """
    return SGNM.load(path)


__version__ = "1.1.0"

__all__ = [
    # Backward compatible
    "SGNM",
    "BaseSGNM",
    "score",
    # Models
    "load",
    # Config
    "ModelConfig",
    "DataConfig",
    "TrainConfig",
    "ScoringConfig",
    "FilterConfig",
    "RelaxConfig",
    "BatchScoringConfig",
    # Constants
    "FRAME1",
    "FRAME2",
    "FRAME3",
    "FRAMES",
    "NUCLEOTIDE_DICT",
    # NN
    "RadialBasisFunctions",
    "DenseNetwork",
    # Scoring
    "Score",
    "StructureScorer",
    "BatchScorer",
    "StructureRelaxer",
    "ScoringResult",
    "RelaxResult",
    "BatchScoringResults",
    # Training
    "Trainer",
    "train_sgnm",
    "TrainResults",
    # Data
    "ReactivityDataset",
    "Sample",
    "ProfileLoader",
    "load_reactivity_index",
    # VAE
    "vae",
    "VAEConfig",
    "VAETrainConfig",
    "StructureVAE",
    "VAELoss",
    "VAETrainer",
    "train_vae",
    "StructureOnlyDataset",
    "StructureDataConfig",
]
