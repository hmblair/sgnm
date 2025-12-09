"""
Training infrastructure for SGNM.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Callable
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from .config import TrainConfig, DataConfig, ModelConfig
from .models import SGNM, BaseSGNM
from .data import HDF5Dataset, Sample


def _normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to [0, 1] range."""
    min_val = x.min()
    max_val = (x - min_val).max()
    if max_val > 0:
        return (x - min_val) / max_val
    return x - min_val


@dataclass
class TrainState:
    """Mutable training state."""

    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float('inf')
    patience_counter: int = 0

    def should_stop(self, patience: int) -> bool:
        """Check if early stopping criteria met."""
        return self.patience_counter >= patience


@dataclass
class TrainResults:
    """Results from training run."""

    best_val_loss: float
    final_epoch: int
    total_steps: int
    history: list[dict] = field(default_factory=list)


class Trainer:
    """
    Training loop manager for SGNM models.

    This class provides a complete training loop with:
    - Proper backward pass and optimizer stepping (fixing the original bug)
    - Validation/eval-only mode
    - Checkpointing
    - Early stopping
    - Optional W&B logging
    """

    def __init__(
        self,
        model: BaseSGNM,
        config: TrainConfig,
        train_data: HDF5Dataset | Iterator[Sample],
        val_data: HDF5Dataset | Iterator[Sample] | None = None,
    ) -> None:
        """
        Initialize trainer.

        Args:
            model: Model to train
            config: Training configuration
            train_data: Training dataset or iterator
            val_data: Validation dataset or iterator (optional)
        """
        self.model = model.to(config.device)
        self.config = config
        self.train_data = train_data
        self.val_data = val_data

        self.state = TrainState()

        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
        )

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Initialize logging (wandb, etc.)."""
        if self.config.wandb_project:
            try:
                import wandb
                wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_run,
                )
                self.wandb = wandb
            except ImportError:
                print("Warning: wandb not installed, skipping logging")
                self.wandb = None
        else:
            self.wandb = None

    def train(self) -> TrainResults:
        """
        Main training loop.

        Returns:
            TrainResults with training history and best metrics
        """
        history = []

        for epoch in range(self.config.max_epochs):
            self.state.epoch = epoch

            # Training phase (skip if eval_only)
            if not self.config.eval_only:
                train_metrics = self._train_epoch()
                history.append({"epoch": epoch, "train": train_metrics})

            # Validation phase
            if self.val_data:
                val_metrics = self._validate()
                if history and "train" in history[-1]:
                    history[-1]["val"] = val_metrics
                else:
                    history.append({"epoch": epoch, "val": val_metrics})

                # Early stopping check
                if self._check_improvement(val_metrics['loss']):
                    self.state.patience_counter = 0
                    if self.config.save_best:
                        self._save_checkpoint('best.pth')
                else:
                    self.state.patience_counter += 1

                if self.state.should_stop(self.config.patience):
                    print(f"Early stopping at epoch {epoch}")
                    break

            # Periodic checkpointing
            if not self.config.eval_only and epoch % self.config.save_every == 0:
                self._save_checkpoint(f'epoch_{epoch}.pth')

            # Log epoch summary
            self._log_epoch(epoch, history[-1] if history else {})

        return TrainResults(
            best_val_loss=self.state.best_val_loss,
            final_epoch=self.state.epoch,
            total_steps=self.state.global_step,
            history=history,
        )

    def _train_epoch(self) -> dict[str, float]:
        """Execute one training epoch."""
        self.model.train()

        epoch_loss = 0.0
        num_samples = 0

        for sample in self.train_data:
            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            pred = self.model.ciffy(sample.polymer, sample.sequence)

            # Trim predictions to match reactivity if needed
            trim = getattr(self.train_data, 'config', None)
            if trim and hasattr(trim, 'trim_ends') and trim.trim_ends > 0:
                t = trim.trim_ends
                if pred.size(0) > 2 * t:
                    pred = pred[t:-t]

            # Normalize predictions
            pred = _normalize(pred)

            # Get target (handle multi-channel reactivity)
            if sample.reactivity.dim() > 1:
                target = _normalize(sample.reactivity[..., 0])
            else:
                target = _normalize(sample.reactivity)

            # Compute loss (only on valid positions)
            mask = sample.mask
            if mask.size(0) != pred.size(0):
                # Size mismatch - skip this sample
                continue

            loss = torch.abs(pred[mask] - target[mask]).mean()

            # Backward pass (THE CRITICAL FIX!)
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                )

            # Optimizer step (THE CRITICAL FIX!)
            self.optimizer.step()

            # Accumulate metrics
            epoch_loss += loss.item()
            num_samples += 1
            self.state.global_step += 1

            # Logging
            if self.state.global_step % self.config.log_every == 0:
                self._log_step(loss.item())

            # Visualization
            if (self.config.visualize_every > 0 and
                self.state.global_step % self.config.visualize_every == 0):
                self._visualize(sample, pred, target)

        return {'loss': epoch_loss / max(num_samples, 1), 'samples': num_samples}

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        """Execute validation pass."""
        self.model.eval()

        total_loss = 0.0
        num_samples = 0

        for sample in self.val_data:
            pred = self.model.ciffy(sample.polymer, sample.sequence)

            # Normalize
            pred = _normalize(pred)

            if sample.reactivity.dim() > 1:
                target = _normalize(sample.reactivity[..., 0])
            else:
                target = _normalize(sample.reactivity)

            mask = sample.mask
            if mask.size(0) != pred.size(0):
                continue

            loss = torch.abs(pred[mask] - target[mask]).mean()

            total_loss += loss.item()
            num_samples += 1

        return {'loss': total_loss / max(num_samples, 1), 'samples': num_samples}

    def _check_improvement(self, val_loss: float) -> bool:
        """Check if validation loss improved."""
        if val_loss < self.state.best_val_loss:
            self.state.best_val_loss = val_loss
            return True
        return False

    def _save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / filename

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.state.epoch,
            'global_step': self.state.global_step,
            'best_val_loss': self.state.best_val_loss,
        }, path)

    def _log_step(self, loss: float) -> None:
        """Log step-level metrics."""
        if self.wandb:
            self.wandb.log({
                'train/loss': loss,
                'train/step': self.state.global_step,
            })

    def _log_epoch(self, epoch: int, metrics: dict) -> None:
        """Log epoch-level metrics."""
        parts = [f"Epoch {epoch}"]
        if 'train' in metrics:
            parts.append(f"train_loss={metrics['train']['loss']:.4f}")
        if 'val' in metrics:
            parts.append(f"val_loss={metrics['val']['loss']:.4f}")
        print(" | ".join(parts))

        if self.wandb:
            log_dict = {'epoch': epoch}
            if 'train' in metrics:
                log_dict['train/epoch_loss'] = metrics['train']['loss']
            if 'val' in metrics:
                log_dict['val/loss'] = metrics['val']['loss']
            self.wandb.log(log_dict)

    def _visualize(self, sample: Sample, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Generate visualization plot."""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(target.detach().cpu(), color="red", alpha=0.5, label="Ground Truth")
        ax.plot(pred.detach().cpu(), color="blue", alpha=0.5, label="Prediction")
        ax.legend()
        ax.set_title(f"Sample: {sample.name}")
        ax.set_xlabel("Residue")
        ax.set_ylabel("Normalized Reactivity")

        if self.wandb:
            self.wandb.log({"visualization": self.wandb.Image(fig)})
        else:
            plt.show(block=False)
            plt.pause(0.5)

        plt.close(fig)


def train_sgnm(
    model_config: ModelConfig,
    data_config: DataConfig,
    train_config: TrainConfig,
) -> TrainResults:
    """
    Main entry point for training SGNM models.

    Example usage:
        from sgnm.config import ModelConfig, DataConfig, TrainConfig
        from sgnm.training import train_sgnm

        results = train_sgnm(
            model_config=ModelConfig(dim=32, layers=2),
            data_config=DataConfig(
                reactivity_path="/path/to/profiles.h5",
                structures_dir="/path/to/structures/",
            ),
            train_config=TrainConfig(
                learning_rate=1e-3,
                max_epochs=50,
            ),
        )

    Args:
        model_config: Model architecture configuration
        data_config: Dataset configuration
        train_config: Training loop configuration

    Returns:
        TrainResults with training history and best metrics
    """
    # Set seeds for reproducibility
    import random
    import numpy as np

    seed = data_config.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Create model
    if model_config.weights_path:
        model = SGNM.load(model_config.weights_path)
    else:
        model = SGNM(model_config)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Create datasets
    train_dataset = HDF5Dataset(data_config, split="train")
    val_dataset = HDF5Dataset(data_config, split="val")

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create trainer and run
    trainer = Trainer(
        model=model,
        config=train_config,
        train_data=train_dataset,
        val_data=val_dataset,
    )

    return trainer.train()
