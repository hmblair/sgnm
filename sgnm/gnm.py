"""Geometry utilities for SGNM models.

Provides local frame construction and orientation scoring. GNM math
(variances, correlations, Laplacian) is delegated to ciffy.operations.
"""
import torch

from ciffy.operations.gnm import gnm_variances as _gnm_variances  # noqa: F401


def _local_frame(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Get the local frame spanned by the two input vectors and their normal.
    """
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 - x1 * (x1 * x2).sum(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)
    x3 = torch.cross(x1, x2, dim=-1)
    return torch.stack([x1, x2, x3], dim=-2)


def _orientation_score(frames: torch.Tensor) -> torch.Tensor:
    """
    Get pairwise base orientations based on the input frames, and score them
    using the SO(3) metric.
    """
    score = frames[..., :, None, :, :] @ frames[..., None, :, :, :].transpose(-1, -2)
    score = torch.einsum("...ii->...", score)
    score = (score - 1).abs() / 2
    return score
