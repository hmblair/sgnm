"""
SGNM: Structure-Guided Normal Mode Model for RNA reactivity prediction.
"""

from .models import SGNM
from .equivariant import EquivariantReactivityModel
from .scoring import StructureScorer, rank
from .training import train
from .data import load_reactivity_index

_MODEL_REGISTRY = {
    "SGNM": SGNM,
    "EquivariantReactivityModel": EquivariantReactivityModel,
}


def load(path: str):
    """Load a model from a checkpoint.

    Dispatches to the correct model class based on the ``model_type``
    key in the checkpoint.  Falls back to SGNM for legacy checkpoints.
    """
    import torch
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    model_type = checkpoint.get("model_type", "SGNM")
    cls = _MODEL_REGISTRY.get(model_type)
    if cls is None:
        raise ValueError(f"Unknown model type: {model_type}")
    return cls.load(path)


try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0.dev0"
