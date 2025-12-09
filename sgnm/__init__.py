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
from .data import HDF5Dataset, Sample, ProfileLoader, tokenize


# =============================================================================
# Convenience Functions
# =============================================================================

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
    "HDF5Dataset",
    "Sample",
    "ProfileLoader",
    "tokenize",
]
