"""
SGNM: Structure-Guided Normal Mode Model for RNA reactivity prediction.
"""

# =============================================================================
# Backward Compatible Exports (original API)
# =============================================================================
from .models import SGNM

# =============================================================================
# New API
# =============================================================================

# Configuration
from .config import (
    ModelConfig,
    DataConfig,
    TrainConfig,
    ScoringConfig,
    RelaxConfig,
)

# Neural Network Layers
from .nn import RadialBasisFunctions, DenseNetwork

# Loss Functions and Schedulers
from .losses import pearson_correlation, mae_loss, mse_loss, correlation_loss
from .schedulers import get_cosine_schedule_with_warmup

# Scoring
from .scoring import (
    StructureScorer,
    StructureRelaxer,
    ScoringResult,
    RelaxResult,
)

# Training
from .training import train, TrainResults

# Data
from .data import ReactivityDataset, Sample, ProfileLoader, load_reactivity_index

# =============================================================================
# Convenience Functions
# =============================================================================

def equivariant(**kwargs) -> "EquivariantReactivityModel":
    """Create an EquivariantReactivityModel. Requires flash-eq."""
    from .equivariant import EquivariantReactivityModel
    return EquivariantReactivityModel(**kwargs)


def load(path: str) -> SGNM:
    """
    Load an SGNM model from a checkpoint.

    Args:
        path: Path to checkpoint file.

    Returns:
        Loaded SGNM model.
    """
    return SGNM.load(path)


__version__ = "2.0.0"

__all__ = [
    # Models
    "SGNM",
    # Models
    "load",
    # Config
    "ModelConfig",
    "DataConfig",
    "TrainConfig",
    "ScoringConfig",
    "RelaxConfig",
    # NN
    "RadialBasisFunctions",
    "DenseNetwork",
    # Scoring
    "StructureScorer",
    "StructureRelaxer",
    "ScoringResult",
    "RelaxResult",
    # Training
    "train",
    "TrainResults",
    # Data
    "ReactivityDataset",
    "Sample",
    "ProfileLoader",
    "load_reactivity_index",
]
