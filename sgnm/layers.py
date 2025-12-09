"""
DEPRECATED: This module is maintained for backward compatibility only.

All classes have been moved to their own modules:
- RadialBasisFunctions, DenseNetwork -> sgnm.nn
- BaseSGNM, SGNM -> sgnm.models
- Score -> sgnm.scoring
- FRAME1, FRAME2, FRAME3, FRAMES -> sgnm.config

Please update your imports to use the new module locations.
"""
import warnings

# Re-export from new locations for backward compatibility
from .config import FRAME1, FRAME2, FRAME3, FRAMES
from .nn import RadialBasisFunctions, DenseNetwork
from .models import BaseSGNM, SGNM, _normalize, _base_frame
from .scoring import Score

# Issue deprecation warning on import
warnings.warn(
    "The 'sgnm.layers' module is deprecated. "
    "Please import from 'sgnm' directly or from the specific modules: "
    "sgnm.nn, sgnm.models, sgnm.scoring, sgnm.config",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "FRAME1",
    "FRAME2",
    "FRAME3",
    "FRAMES",
    "RadialBasisFunctions",
    "DenseNetwork",
    "BaseSGNM",
    "SGNM",
    "Score",
    "_normalize",
    "_base_frame",
]
