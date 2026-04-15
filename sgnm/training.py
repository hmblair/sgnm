"""
Training infrastructure for SGNM.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import torch
import torch.nn as nn

from dlu.logging import WandbLogger, is_wandb_available, ConsoleProgress
from dlu.training import LossTracker

from .config import TrainConfig, DataConfig
from .data import ReactivityDataset, Sample
from .scoring import metric, normalize
from dlu.schedulers import get_cosine_schedule_with_warmup


@dataclass
class TrainResults:
    """Results from training run."""

    best_val_loss: float
    final_epoch: int
    total_steps: int
    history: list[dict] = field(default_factory=list)




class Trainer:
    """Per-model training state: optimizer, scheduler, logging, checkpointing."""

    def __init__(
        self,
        name: str,
        model: nn.Module,
        config: TrainConfig,
        num_samples: int,
    ) -> None:
        self.name = name
        self.model = model.to(config.device)
        self.config = config

        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # Optimizer
        if config.weight_decay > 0:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
            )

        # Scheduler
        self.scheduler = None
        if config.warmup_epochs > 0:
            num_training_steps = num_samples * config.max_epochs
            num_warmup_steps = int(num_samples * config.warmup_epochs)
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                min_lr_ratio=config.min_lr_ratio,
            )

        self._train_tracker = LossTracker(name)
        self._val_tracker = LossTracker(f"{name}/val")

        # Logging
        self._wandb: WandbLogger | None = None
        if config.wandb_project and is_wandb_available():
            run_name = config.wandb_run or name
            self._wandb = WandbLogger(config.wandb_project, name=run_name, reinit=True)

    def begin_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._train_tracker.start_epoch()
        self._epoch_skips = {"model_error": 0, "size_mismatch": 0}
        self.model.train()
        self.optimizer.zero_grad()

    def _perturb(self, polymer, noise_std: float):
        """Add Gaussian noise to polymer coordinates."""
        if noise_std > 0:
            polymer = polymer.copy()
            polymer.coordinates = (
                polymer.coordinates + torch.randn_like(polymer.coordinates) * noise_std
            )
        return polymer

    def train_step(self, sample: Sample) -> None:
        """Run one training step on a sample."""
        sample = sample.to(self.config.device)

        try:
            poly = self._perturb(sample.polymer, self.config.noise_std)
            pred = self.model.ciffy(poly)
        except (ValueError, RuntimeError) as e:
            self._epoch_skips["model_error"] += 1
            if self._epoch_skips["model_error"] <= 3:
                print(f"  [{self.name}] model_error on '{sample.name}': {e}")
            return

        target = sample.reactivity
        mask = sample.mask

        if mask.size(0) != pred.size(0):
            self._epoch_skips["size_mismatch"] += 1
            return

        # Normalize per-channel
        pred = pred[mask]
        target = target[mask]
        pred = normalize(pred)
        target = normalize(target)

        loss = metric(pred, target, metric=self.config.loss_type).mean()
        if self.config.loss_type == "correlation":
            loss = -loss
        loss.backward()

        if self.config.gradient_clip:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip,
            )
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad()

        self._train_tracker.update(loss.item())
        self.global_step += 1

        if self._wandb and self.global_step % self.config.log_every == 0:
            self._wandb.log_step(self._train_tracker.metrics)

        if (self._wandb
                and self.config.visualize_every > 0
                and self.global_step % self.config.visualize_every == 0):
            self._visualize(sample, pred, target)

    def end_epoch(self) -> dict:
        trained = self._train_tracker.current_step + 1
        skipped = sum(self._epoch_skips.values())
        return {
            "loss": self._train_tracker.average_loss,
            "trained": trained,
            "skipped": skipped,
            "skip_details": {k: v for k, v in self._epoch_skips.items() if v > 0},
        }

    def begin_validation(self) -> None:
        self._val_mae_tracker = LossTracker(f"{self.name}/val_mae")
        self._val_corr_tracker = LossTracker(f"{self.name}/val_corr")
        self._val_shape_mae = LossTracker(f"{self.name}/val_shape_mae")
        self._val_shape_corr = LossTracker(f"{self.name}/val_shape_corr")
        self._val_dms_mae = LossTracker(f"{self.name}/val_dms_mae")
        self._val_dms_corr = LossTracker(f"{self.name}/val_dms_corr")
        for t in (self._val_mae_tracker, self._val_corr_tracker,
                  self._val_shape_mae, self._val_shape_corr,
                  self._val_dms_mae, self._val_dms_corr):
            t.start_epoch()
        self.model.eval()

    @torch.no_grad()
    def validate_step(self, sample: Sample) -> None:
        sample = sample.to(self.config.device)

        try:
            poly = self._perturb(sample.polymer, self.config.val_noise_std)
            pred = self.model.ciffy(poly)
        except (ValueError, RuntimeError):
            return

        target = sample.reactivity
        mask = sample.mask

        if mask.size(0) != pred.size(0):
            return

        pred = normalize(pred[mask])
        target = normalize(target[mask])
        # Combined metrics
        self._val_mae_tracker.update(metric(pred, target, metric="mae").mean().item())
        corr = metric(pred, target, metric="correlation")
        self._val_corr_tracker.update(corr.mean().item())

        # Per-channel metrics
        if corr.dim() > 0 and corr.size(0) >= 2:
            self._val_shape_corr.update(corr[0].item())
            self._val_dms_corr.update(corr[1].item())
            mae = metric(pred, target, metric="mae")
            self._val_shape_mae.update(mae[0].item())
            self._val_dms_mae.update(mae[1].item())

    def end_validation(self) -> dict:
        val_mae = self._val_mae_tracker.average_loss
        val_corr = self._val_corr_tracker.average_loss

        # Early stopping uses the training loss type
        val_loss = val_corr if self.config.loss_type == "correlation" else val_mae
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            if self.config.save_best:
                self._save_checkpoint("best.pth")
        else:
            self.patience_counter += 1

        return {
            "mae": val_mae,
            "corr": val_corr,
            "shape_mae": self._val_shape_mae.average_loss,
            "shape_corr": self._val_shape_corr.average_loss,
            "dms_mae": self._val_dms_mae.average_loss,
            "dms_corr": self._val_dms_corr.average_loss,
        }

    def should_stop(self) -> bool:
        return self.patience_counter >= self.config.patience

    def log_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict | None) -> None:
        parts = [f"[{self.name}] Epoch {epoch}"]
        parts.append(f"train_loss={train_metrics['loss']:.4f}")
        parts.append(f"trained={train_metrics['trained']}")
        if train_metrics.get("skip_details"):
            skip_str = ", ".join(f"{k}={v}" for k, v in train_metrics["skip_details"].items())
            parts.append(f"skipped={train_metrics['skipped']} ({skip_str})")
        if val_metrics:
            parts.append(f"val_corr={val_metrics['corr']:.4f}")
            parts.append(f"shape={val_metrics['shape_corr']:.4f}")
            parts.append(f"dms={val_metrics['dms_corr']:.4f}")
        print(" | ".join(parts))

        if self._wandb:
            log = {"epoch": epoch, **self._train_tracker.epoch_metrics}
            if val_metrics:
                log["val/mae"] = val_metrics["mae"]
                log["val/corr"] = val_metrics["corr"]
                log["val/shape_mae"] = val_metrics["shape_mae"]
                log["val/shape_corr"] = val_metrics["shape_corr"]
                log["val/dms_mae"] = val_metrics["dms_mae"]
                log["val/dms_corr"] = val_metrics["dms_corr"]
            self._wandb.log_epoch(epoch, log)

    def _visualize(self, sample: Sample, pred: torch.Tensor, target: torch.Tensor) -> None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(target.detach().cpu(), color="red", alpha=0.5, label="Ground Truth")
        ax.plot(pred.detach().cpu(), color="blue", alpha=0.5, label="Prediction")
        ax.legend()
        ax.set_title(f"[{self.name}] {sample.name}")
        ax.set_xlabel("Residue")
        ax.set_ylabel("Normalized Reactivity")
        import wandb
        self._wandb.log_step({"visualization": wandb.Image(fig)})
        plt.close(fig)

    def _save_checkpoint(self, filename: str) -> None:
        checkpoint_dir = Path(self.config.checkpoint_dir) / self.name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_type": type(self.model).__name__,
            "model_state_dict": self.model.state_dict(),
            "init_kwargs": self.model._init_kwargs,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }
        torch.save(checkpoint, checkpoint_dir / filename)

    def finish(self) -> None:
        if self._wandb:
            self._wandb.close()

    def results(self) -> TrainResults:
        return TrainResults(
            best_val_loss=self.best_val_loss,
            final_epoch=self.epoch,
            total_steps=self.global_step,
        )


def train(
    name: str,
    model: nn.Module,
    data_config: DataConfig,
    train_config: TrainConfig,
) -> TrainResults:
    """Train a reactivity prediction model.

    Args:
        name: Model name (used for logging and checkpoints).
        model: Model with a .ciffy() method.
        data_config: Dataset configuration.
        train_config: Training loop configuration.

    Returns:
        TrainResults with best val loss and final epoch.
    """
    import random
    import numpy as np

    seed = data_config.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    from ciffy.biochemistry import Scale, Molecule
    from ciffy.nn import PolymerDataset
    from .data import load_reactivity_index

    index = load_reactivity_index(
        data_config.reactivity_paths,
        data_config.fasta_path,
    )

    structures = PolymerDataset(
        data_config.structures_dir,
        scale=Scale.CHAIN,
        molecule_types=Molecule.RNA,
        max_chains=data_config.max_chains,
    )
    splits = structures.split(
        train=data_config.train_split,
        val=data_config.val_split,
        test=data_config.test_split,
        seed=seed,
    )
    train_dataset = ReactivityDataset(splits[0], index)
    val_dataset = ReactivityDataset(splits[1], index) if len(splits) > 1 else None

    print(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")
    print(f"[{name}] Parameters: {sum(p.numel() for p in model.parameters())}")

    trainer = Trainer(name, model, train_config, len(train_dataset))
    progress = ConsoleProgress(train_dataset, name="train")

    for epoch in range(train_config.max_epochs):
        trainer.begin_epoch(epoch)

        progress.start_epoch(epoch)
        for sample in progress:
            trainer.train_step(sample)
            progress.update({"loss": trainer._train_tracker.current_loss})

        train_metrics = trainer.end_epoch()

        if epoch == 0:
            print(f"\nTraining data:\n{train_dataset.summary()}")

        val_metrics = None
        if val_dataset:
            trainer.begin_validation()
            for sample in val_dataset:
                trainer.validate_step(sample)
            val_metrics = trainer.end_validation()

        trainer.log_epoch(epoch, train_metrics, val_metrics)
        if val_metrics and trainer.should_stop():
            print(f"[{name}] Early stopping at epoch {epoch}")
            break

    progress.close()
    trainer.finish()
    return trainer.results()
