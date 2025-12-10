"""
Loss functions for the Structure VAE.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .config import VAEConfig


def kabsch_rmsd(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Compute RMSD after optimal rigid alignment using the Kabsch algorithm.

    This loss is E(3)-invariant: the same loss value regardless of
    rotation/translation applied to either structure.

    Args:
        pred: (N, 3) predicted coordinates
        target: (N, 3) target coordinates

    Returns:
        Scalar RMSD value (differentiable)
    """
    # Center both structures
    pred_centered = pred - pred.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)

    # Compute covariance matrix H = pred^T @ target
    H = pred_centered.T @ target_centered  # (3, 3)

    # SVD: H = U @ S @ V^T
    U, S, Vt = torch.linalg.svd(H)

    # Handle reflection case (det < 0)
    # R = V @ D @ U^T where D corrects for reflections
    d = torch.sign(torch.linalg.det(Vt.T @ U.T)).detach()
    D = torch.diag(torch.tensor([1.0, 1.0, d.item()], device=pred.device, dtype=pred.dtype))

    # Optimal rotation
    R = Vt.T @ D @ U.T  # (3, 3)

    # Apply rotation to prediction
    pred_aligned = pred_centered @ R

    # Compute RMSD
    diff = pred_aligned - target_centered
    msd = (diff**2).sum(dim=-1).mean()
    rmsd = torch.sqrt(msd + 1e-8)  # Small epsilon for numerical stability

    return rmsd


def kl_divergence(
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """
    Compute KL divergence from N(mu, sigma^2) to standard normal N(0, I).

    KL(q || p) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    Args:
        mu: (..., latent_dim) mean of approximate posterior
        logvar: (..., latent_dim) log variance of approximate posterior

    Returns:
        Scalar KL divergence (mean over all dimensions)
    """
    # KL = -0.5 * (1 + logvar - mu^2 - exp(logvar))
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl.mean()


def pairwise_distance_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Loss based on pairwise distances (fully E(3)-invariant).

    This compares the internal geometry directly without alignment.
    Useful as an auxiliary loss alongside RMSD.

    Args:
        pred: (N, 3) predicted coordinates
        target: (N, 3) target coordinates

    Returns:
        Scalar mean absolute error on pairwise distances
    """
    pred_dists = torch.cdist(pred, pred)
    target_dists = torch.cdist(target, target)

    return torch.abs(pred_dists - target_dists).mean()


class VAELoss(nn.Module):
    """
    Combined VAE loss: reconstruction + KL divergence.

    Supports KL warmup (annealing) over training epochs.
    """

    def __init__(
        self,
        config: VAEConfig,
        use_pairwise: bool = False,
        pairwise_weight: float = 0.1,
    ) -> None:
        """
        Args:
            config: VAE configuration
            use_pairwise: Whether to add pairwise distance loss
            pairwise_weight: Weight for pairwise distance loss
        """
        super().__init__()
        self.config = config
        self.use_pairwise = use_pairwise
        self.pairwise_weight = pairwise_weight

        # Track current epoch for KL warmup
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch for KL warmup scheduling."""
        self._epoch = epoch

    @property
    def kl_weight(self) -> float:
        """Get current KL weight with warmup schedule."""
        if self._epoch >= self.config.kl_warmup_epochs:
            return self.config.kl_weight

        # Linear warmup from 0 to kl_weight
        progress = self._epoch / max(self.config.kl_warmup_epochs, 1)
        return self.config.kl_weight * progress

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute VAE loss.

        Args:
            pred: (N, 3) predicted/reconstructed coordinates
            target: (N, 3) target coordinates
            mu: (N, latent_dim) latent mean
            logvar: (N, latent_dim) latent log variance

        Returns:
            Dictionary with keys:
                - 'total': Total loss for backprop
                - 'recon': Reconstruction loss (RMSD)
                - 'kl': KL divergence
                - 'kl_weight': Current KL weight
                - 'pairwise': Pairwise distance loss (if enabled)
        """
        # Reconstruction loss (Kabsch RMSD)
        recon_loss = kabsch_rmsd(pred, target)

        # KL divergence
        kl_loss = kl_divergence(mu, logvar)

        # Total loss
        total = recon_loss + self.kl_weight * kl_loss

        result = {
            "total": total,
            "recon": recon_loss,
            "kl": kl_loss,
            "kl_weight": torch.tensor(self.kl_weight, device=pred.device),
        }

        # Optional pairwise distance loss
        if self.use_pairwise:
            pairwise_loss = pairwise_distance_loss(pred, target)
            result["pairwise"] = pairwise_loss
            result["total"] = total + self.pairwise_weight * pairwise_loss

        return result
