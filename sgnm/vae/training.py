"""
Training infrastructure for the Structure VAE.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator
import torch
import torch.nn as nn
import ciffy

from ..config import FRAME1, FRAME2, FRAME3
from ..data import tokenize
from ..training import TrainState, TrainResults
from .config import VAEConfig
from .model import StructureVAE
from .losses import VAELoss


def _base_frame(poly: ciffy.Polymer) -> torch.Tensor:
    """
    Extract local coordinate frames from nucleobase C2-C4-C6 atoms.

    Args:
        poly: ciffy Polymer object

    Returns:
        (N, 3, 3) local frame matrices
    """
    from ..gnm import _local_frame

    # Get sequence to determine correct atom indices per residue type
    seq = poly.str()
    nuc_map = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}

    N = poly.size(ciffy.RESIDUE)
    coords = poly.coordinates.view(N, -1, 3)

    # Get frame atom indices for each residue based on type
    frame1_idx = torch.tensor([FRAME1[nuc_map.get(s, 0)] for s in seq])
    frame2_idx = torch.tensor([FRAME2[nuc_map.get(s, 0)] for s in seq])
    frame3_idx = torch.tensor([FRAME3[nuc_map.get(s, 0)] for s in seq])

    # Extract atom coordinates for each frame point
    c2 = coords[torch.arange(N), frame1_idx]
    c4 = coords[torch.arange(N), frame2_idx]
    c6 = coords[torch.arange(N), frame3_idx]

    # Compute local frames
    v1 = c4 - c2
    v2 = c6 - c2

    return _local_frame(v1, v2)


@dataclass
class VAETrainConfig:
    """Configuration for VAE training."""

    learning_rate: float = 1e-4
    """Learning rate for optimizer."""

    weight_decay: float = 1e-5
    """Weight decay for AdamW optimizer."""

    max_epochs: int = 200
    """Maximum number of training epochs."""

    gradient_clip: float = 1.0
    """Maximum gradient norm for clipping."""

    patience: int = 20
    """Epochs without improvement before early stopping."""

    device: str = "cpu"
    """Device to train on ('cpu', 'cuda', 'mps')."""

    checkpoint_dir: str = "./checkpoints/vae"
    """Directory for saving model checkpoints."""

    save_every: int = 10
    """Save checkpoint every N epochs."""

    save_best: bool = True
    """Save checkpoint when validation loss improves."""

    log_every: int = 10
    """Log metrics every N steps."""

    use_pairwise_loss: bool = False
    """Whether to add pairwise distance loss."""

    pairwise_weight: float = 0.1
    """Weight for pairwise distance loss."""

    wandb_project: str | None = None
    """Weights & Biases project name (None to disable)."""

    wandb_run: str | None = None
    """Weights & Biases run name."""


@dataclass
class StructureSample:
    """A single structure sample for VAE training."""

    name: str
    """Sample identifier."""

    coords: torch.Tensor
    """(N, 3) residue coordinates."""

    node_types: torch.Tensor
    """(N,) nucleotide type indices."""

    frames: torch.Tensor | None = None
    """(N, 3, 3) optional local coordinate frames."""


class VAETrainer:
    """
    Training loop for StructureVAE.

    Similar structure to the SGNM Trainer but adapted for VAE losses
    and coordinate-based data.
    """

    def __init__(
        self,
        model: StructureVAE,
        vae_config: VAEConfig,
        train_config: VAETrainConfig,
        train_data: Iterator[StructureSample],
        val_data: Iterator[StructureSample] | None = None,
    ) -> None:
        """
        Initialize trainer.

        Args:
            model: StructureVAE model to train
            vae_config: VAE architecture configuration
            train_config: Training configuration
            train_data: Training data iterator
            val_data: Validation data iterator (optional)
        """
        self.model = model.to(train_config.device)
        self.vae_config = vae_config
        self.train_config = train_config
        self.train_data = train_data
        self.val_data = val_data

        self.state = TrainState()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        # Loss function
        self.loss_fn = VAELoss(
            vae_config,
            use_pairwise=train_config.use_pairwise_loss,
            pairwise_weight=train_config.pairwise_weight,
        )

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Initialize logging (wandb, etc.)."""
        if self.train_config.wandb_project:
            try:
                import wandb

                wandb.init(
                    project=self.train_config.wandb_project,
                    name=self.train_config.wandb_run,
                    config={
                        "vae_config": vars(self.vae_config),
                        "train_config": vars(self.train_config),
                    },
                )
                self.wandb = wandb
            except ImportError:
                print("Warning: wandb not installed, skipping logging")
                self.wandb = None
        else:
            self.wandb = None

    def _to_device(self, sample: StructureSample) -> StructureSample:
        """Move sample to training device."""
        device = self.train_config.device
        return StructureSample(
            name=sample.name,
            coords=sample.coords.to(device),
            node_types=sample.node_types.to(device),
            frames=sample.frames.to(device) if sample.frames is not None else None,
        )

    def train(self) -> TrainResults:
        """
        Main training loop.

        Returns:
            TrainResults with training history
        """
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False

        history = []

        epoch_iter = range(self.train_config.max_epochs)
        if use_tqdm:
            epoch_iter = tqdm(epoch_iter, desc="Training", unit="epoch")

        for epoch in epoch_iter:
            self.state.epoch = epoch
            self.loss_fn.set_epoch(epoch)

            # Training
            train_metrics = self._train_epoch()
            history.append({"epoch": epoch, "train": train_metrics})

            # Validation
            if self.val_data is not None:
                val_metrics = self._validate()
                history[-1]["val"] = val_metrics

                self.scheduler.step(val_metrics["loss"])

                if self._check_improvement(val_metrics["loss"]):
                    self.state.patience_counter = 0
                    if self.train_config.save_best:
                        self._save_checkpoint("best.pth")
                else:
                    self.state.patience_counter += 1

                if self.state.should_stop(self.train_config.patience):
                    print(f"Early stopping at epoch {epoch}")
                    break

            # Periodic checkpointing
            if epoch % self.train_config.save_every == 0:
                self._save_checkpoint(f"epoch_{epoch}.pth")

            # Logging
            self._log_epoch(epoch, history[-1])

            # Update tqdm description with metrics
            if use_tqdm:
                desc = f"Epoch {epoch}"
                if "train" in history[-1]:
                    desc += f" | loss={history[-1]['train']['loss']:.4f}"
                if "val" in history[-1]:
                    desc += f" | val={history[-1]['val']['loss']:.4f}"
                epoch_iter.set_postfix_str(desc.split(" | ", 1)[-1] if " | " in desc else "")

        return TrainResults(
            best_val_loss=self.state.best_val_loss,
            final_epoch=self.state.epoch,
            total_steps=self.state.global_step,
            history=history,
        )

    def _train_epoch(self) -> dict[str, float]:
        """Execute one training epoch."""
        self.model.train()

        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_samples = 0

        try:
            from tqdm import tqdm
            sample_iter = tqdm(
                self.train_data,
                desc=f"  Epoch {self.state.epoch}",
                leave=False,
                unit="sample",
            )
        except ImportError:
            sample_iter = self.train_data

        for sample in sample_iter:
            sample = self._to_device(sample)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(sample.coords, sample.node_types, sample.frames)

            # Compute loss
            losses = self.loss_fn(
                outputs["recon"],
                sample.coords,
                outputs["mu"],
                outputs["logvar"],
            )

            # Backward pass
            losses["total"].backward()

            # Gradient clipping
            if self.train_config.gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config.gradient_clip,
                )

            self.optimizer.step()

            # Accumulate metrics
            total_loss += losses["total"].item()
            total_recon += losses["recon"].item()
            total_kl += losses["kl"].item()
            num_samples += 1
            self.state.global_step += 1

            # Step logging
            if self.state.global_step % self.train_config.log_every == 0:
                self._log_step(losses)

            # Update sample progress bar
            if hasattr(sample_iter, 'set_postfix'):
                sample_iter.set_postfix(
                    loss=total_loss / num_samples,
                    recon=total_recon / num_samples,
                )

        n = max(num_samples, 1)
        return {
            "loss": total_loss / n,
            "recon": total_recon / n,
            "kl": total_kl / n,
            "samples": num_samples,
        }

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        """Execute validation pass."""
        self.model.eval()

        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_samples = 0

        for sample in self.val_data:
            sample = self._to_device(sample)

            outputs = self.model(sample.coords, sample.node_types, sample.frames)

            losses = self.loss_fn(
                outputs["recon"],
                sample.coords,
                outputs["mu"],
                outputs["logvar"],
            )

            total_loss += losses["total"].item()
            total_recon += losses["recon"].item()
            total_kl += losses["kl"].item()
            num_samples += 1

        n = max(num_samples, 1)
        return {
            "loss": total_loss / n,
            "recon": total_recon / n,
            "kl": total_kl / n,
            "samples": num_samples,
        }

    def _check_improvement(self, val_loss: float) -> bool:
        """Check if validation loss improved."""
        if val_loss < self.state.best_val_loss:
            self.state.best_val_loss = val_loss
            return True
        return False

    def _save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint_dir = Path(self.train_config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / filename

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "epoch": self.state.epoch,
                "global_step": self.state.global_step,
                "best_val_loss": self.state.best_val_loss,
                "vae_config": self.vae_config,
                "train_config": self.train_config,
            },
            path,
        )

    def _log_step(self, losses: dict[str, torch.Tensor]) -> None:
        """Log step-level metrics."""
        if self.wandb:
            self.wandb.log(
                {
                    "train/loss": losses["total"].item(),
                    "train/recon": losses["recon"].item(),
                    "train/kl": losses["kl"].item(),
                    "train/kl_weight": losses["kl_weight"].item(),
                    "train/step": self.state.global_step,
                }
            )

    def _log_epoch(self, epoch: int, metrics: dict) -> None:
        """Log epoch-level metrics."""
        parts = [f"Epoch {epoch}"]
        if "train" in metrics:
            parts.append(f"train_loss={metrics['train']['loss']:.4f}")
            parts.append(f"recon={metrics['train']['recon']:.4f}")
            parts.append(f"kl={metrics['train']['kl']:.4f}")
        if "val" in metrics:
            parts.append(f"val_loss={metrics['val']['loss']:.4f}")
        print(" | ".join(parts))

        if self.wandb:
            log_dict = {"epoch": epoch}
            if "train" in metrics:
                log_dict["train/epoch_loss"] = metrics["train"]["loss"]
                log_dict["train/epoch_recon"] = metrics["train"]["recon"]
                log_dict["train/epoch_kl"] = metrics["train"]["kl"]
            if "val" in metrics:
                log_dict["val/loss"] = metrics["val"]["loss"]
                log_dict["val/recon"] = metrics["val"]["recon"]
                log_dict["val/kl"] = metrics["val"]["kl"]
            self.wandb.log(log_dict)


def train_vae(
    vae_config: VAEConfig,
    train_config: VAETrainConfig,
    train_data: Iterator[StructureSample],
    val_data: Iterator[StructureSample] | None = None,
) -> TrainResults:
    """
    High-level training function for Structure VAE.

    Example usage:
        from sgnm.vae import VAEConfig, VAETrainConfig, train_vae
        from sgnm.vae.data import StructureOnlyDataset

        vae_config = VAEConfig(hidden_dim=128, latent_dim=32)
        train_config = VAETrainConfig(learning_rate=1e-4, device='cuda')

        dataset = StructureOnlyDataset('/path/to/structures/')
        results = train_vae(vae_config, train_config, dataset)

    Args:
        vae_config: VAE architecture configuration
        train_config: Training configuration
        train_data: Training data iterator
        val_data: Validation data iterator (optional)

    Returns:
        TrainResults with training history
    """
    # Set seeds for reproducibility
    import random
    import numpy as np

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Create model
    model = StructureVAE(vae_config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer and run
    trainer = VAETrainer(
        model=model,
        vae_config=vae_config,
        train_config=train_config,
        train_data=train_data,
        val_data=val_data,
    )

    return trainer.train()
