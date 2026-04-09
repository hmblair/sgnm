"""Loss functions for reactivity prediction."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def pearsonr_np(x, y):
    """Pearson correlation coefficient for numpy arrays.

    Returns (r, p_value) matching scipy.stats.pearsonr interface.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    xm, ym = x - x.mean(), y - y.mean()
    r = (xm * ym).sum() / (np.sqrt((xm ** 2).sum() * (ym ** 2).sum()) + 1e-12)
    t = r * np.sqrt((n - 2) / (1 - r ** 2 + 1e-12))
    # Two-sided p-value from t-distribution (approximation)
    p = 2 * np.exp(-0.5 * t ** 2) / np.sqrt(2 * np.pi) if n > 2 else 1.0
    return float(r), float(p)


def pearson_correlation(
    pred: torch.Tensor,
    target: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    """Compute Pearson correlation coefficient."""
    pred_centered = pred - pred.mean(dim=dim, keepdim=True)
    target_centered = target - target.mean(dim=dim, keepdim=True)

    numerator = (pred_centered * target_centered).sum(dim=dim)
    denominator = torch.sqrt(
        (pred_centered**2).sum(dim=dim) * (target_centered**2).sum(dim=dim)
    )

    return numerator / (denominator + 1e-8)


def mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute MSE loss over valid entries."""
    if valid_mask is None:
        valid_mask = ~torch.isnan(target)

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    return F.mse_loss(pred[valid_mask], target[valid_mask])


def mae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute MAE loss over valid entries."""
    if valid_mask is None:
        valid_mask = ~torch.isnan(target)

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    return torch.abs(pred[valid_mask] - target[valid_mask]).mean()


def correlation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    min_valid: int = 3,
) -> torch.Tensor:
    """Compute negative correlation loss.

    Minimizing this loss maximizes Pearson correlation.
    Operates per-channel if pred has shape (N, C).
    """
    if valid_mask is None:
        valid_mask = ~torch.isnan(target)

    if pred.dim() == 1:
        pred = pred.unsqueeze(-1)
        target = target.unsqueeze(-1)
        valid_mask = valid_mask.unsqueeze(-1)

    total_corr = 0.0
    n_valid = 0

    for c in range(pred.shape[1]):
        mask = valid_mask[:, c]
        if mask.sum() >= min_valid:
            corr = pearson_correlation(pred[mask, c], target[mask, c])
            total_corr = total_corr + corr
            n_valid += 1

    if n_valid == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    return -total_corr / n_valid
