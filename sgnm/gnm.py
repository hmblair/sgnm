"""GNM math and orientation scoring for SGNM models.

GNM variances are delegated to ciffy.operations.gnm.
"""
import torch

from ciffy.operations.gnm import gnm_variances as _gnm_variances  # noqa: F401


def _orientation_score(frames: torch.Tensor) -> torch.Tensor:
    """
    Get pairwise base orientations based on the input frames, and score them
    using the SO(3) metric.
    """
    score = frames[..., :, None, :, :] @ frames[..., None, :, :, :].transpose(-1, -2)
    score = torch.einsum("...ii->...", score)
    score = (score - 1).abs() / 2
    return score
