import torch


def _local_frame(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Get the local frame spanned by the two input vectors and their normal.
    """

    # Normalize the first vector and subtract the projection of the
    # first vector onto the second vector

    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 - x1 * (x1 * x2).sum(dim=-1, keepdim=True)

    # Normalize the second vector

    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # Get the third vector by taking the cross product of the first two

    x3 = torch.cross(x1, x2, dim=-1)

    # Stack the vectors to get the local frame

    return torch.stack([x1, x2, x3], dim=-2)


def _laplacian(adj: torch.Tensor) -> torch.Tensor:
    """
    Compute the graph Laplacian / Kirchoff matrix.
    """

    deg = torch.diag_embed(adj.sum(-1))
    return deg - adj


def _gnm_correlations(adj: torch.Tensor, rtol: float = 1E-2) -> torch.Tensor:
    """
    Compute correlations between molecules under a Gaussian model.
    """

    lap = _laplacian(adj) / adj.size(0)
    return torch.linalg.pinv(lap, rtol=rtol)


def _gnm_variances(adj: torch.Tensor, rtol: float = 1E-2) -> torch.Tensor:
    """
    Compute the variance in molcular positions under a Gaussian model.
    """

    corr = _gnm_correlations(adj, rtol)
    return torch.diagonal(corr, dim1=-1, dim2=-2)


def _relative_orientation(frames: torch.Tensor) -> torch.Tensor:

    return (
        frames[..., :, None, :, :] @
        frames[..., None, :, :, :].transpose(-1, -2)
    )


def _orientation_score(frames: torch.Tensor) -> torch.Tensor:
    """
    Get pairwise base orientations based on the input frames, and score them
    using the SO(3) metric.
    """

    score = frames[..., :, None, :, :] @ frames[..., None, :, :, :].transpose(-1, -2)
    score = torch.einsum("...ii->...", score)
    score = (score - 1).abs() / 2

    return score
