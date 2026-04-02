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
from .losses import mae_loss, mse_loss, correlation_loss
from .schedulers import get_cosine_schedule_with_warmup


def _normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to [0, 1] range, per-column for 2D input."""
    if x.dim() == 1:
        min_val = x.min()
        max_val = (x - min_val).max()
        return (x - min_val) / max_val.clamp(min=1e-8)
    # Per-column normalization for (N, C)
    min_val = x.min(dim=0).values
    max_val = (x - min_val).max(dim=0).values.clamp(min=1e-8)
    return (x - min_val) / max_val


@dataclass
class TrainResults:
    """Results from training run."""

    best_val_loss: float
    final_epoch: int
    total_steps: int
    history: list[dict] = field(default_factory=list)


LOSS_FNS = {"mae": mae_loss, "mse": mse_loss, "correlation": correlation_loss}


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

        self._loss_fn = LOSS_FNS[config.loss_type]
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

    def train_step(self, sample: Sample) -> None:
        """Run one training step on a sample."""
        sample = sample.to(self.config.device)

        try:
            pred = self.model.ciffy(sample.polymer)
        except (ValueError, RuntimeError):
            self._epoch_skips["model_error"] += 1
            return

        target = sample.reactivity
        mask = sample.mask

        if mask.size(0) != pred.size(0):
            self._epoch_skips["size_mismatch"] += 1
            return

        # Normalize per-channel
        pred = pred[mask]
        target = target[mask]
        pred = _normalize(pred)
        target = _normalize(target)

        loss = self._loss_fn(pred, target)
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
            pred = self.model.ciffy(sample.polymer)
        except (ValueError, RuntimeError):
            return

        target = sample.reactivity
        mask = sample.mask

        if mask.size(0) != pred.size(0):
            return

        pred = _normalize(pred[mask])
        target = _normalize(target[mask])

        # Combined metrics
        self._val_mae_tracker.update(mae_loss(pred, target).item())
        self._val_corr_tracker.update(correlation_loss(pred, target).item())

        # Per-channel metrics
        if pred.dim() > 1 and pred.size(1) >= 2:
            self._val_shape_mae.update(mae_loss(pred[:, 0], target[:, 0]).item())
            self._val_shape_corr.update(correlation_loss(pred[:, 0], target[:, 0]).item())
            self._val_dms_mae.update(mae_loss(pred[:, 1], target[:, 1]).item())
            self._val_dms_corr.update(correlation_loss(pred[:, 1], target[:, 1]).item())

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
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }, checkpoint_dir / filename)

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
    models: dict[str, nn.Module],
    data_config: DataConfig,
    train_config: TrainConfig,
) -> dict[str, TrainResults]:
    """
    Train one or more reactivity prediction models on the same data.

    All models see the same samples in the same order at each epoch.

    Args:
        models: Dict of {name: model}. Each model must have a .ciffy() method.
        data_config: Dataset configuration.
        train_config: Training loop configuration.

    Returns:
        Dict of {name: TrainResults}.
    """
    import random
    import numpy as np

    seed = data_config.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Build datasets (shared across all models)
    from ciffy.biochemistry import Scale, Molecule
    from ciffy.nn import PolymerDataset
    from .data import load_reactivity_index

    index = load_reactivity_index(
        data_config.reactivity_path,
        data_config.fasta_path,
        data_config.data_format,
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

    # Create per-model trainers
    trainers: dict[str, Trainer] = {}
    for name, model in models.items():
        # Tag wandb run per model
        tc = TrainConfig(**{
            f.name: getattr(train_config, f.name)
            for f in train_config.__dataclass_fields__.values()
        })
        if tc.wandb_run:
            tc.wandb_run = f"{tc.wandb_run}-{name}"
        elif tc.wandb_project:
            tc.wandb_run = name

        print(f"[{name}] Parameters: {sum(p.numel() for p in model.parameters())}")
        trainers[name] = Trainer(name, model, tc, len(train_dataset))

    # Console progress on training data
    progress = ConsoleProgress(train_dataset, name="train")

    # Shared training loop
    results: dict[str, TrainResults] = {}
    for epoch in range(train_config.max_epochs):
        # Train
        for t in trainers.values():
            t.begin_epoch(epoch)

        progress.start_epoch(epoch)
        for sample in progress:
            for t in trainers.values():
                t.train_step(sample)
            progress.update({"loss": list(trainers.values())[0]._train_tracker.current_loss})

        train_metrics = {n: t.end_epoch() for n, t in trainers.items()}

        # Print dataset summary after first epoch
        if epoch == 0:
            print(f"\nTraining data:\n{train_dataset.summary()}")

        # Validate
        val_metrics = {}
        if val_dataset:
            for t in trainers.values():
                t.begin_validation()

            for sample in val_dataset:
                for t in trainers.values():
                    t.validate_step(sample)

            val_metrics = {n: t.end_validation() for n, t in trainers.items()}

        # Log and check stopping
        stopped = []
        for name, t in trainers.items():
            t.log_epoch(epoch, train_metrics[name], val_metrics.get(name))
            if val_metrics and t.should_stop():
                print(f"[{name}] Early stopping at epoch {epoch}")
                stopped.append(name)

        for name in stopped:
            trainers[name].finish()
            results[name] = trainers[name].results()
            del trainers[name]

        if not trainers:
            break

    # Finalize remaining
    progress.close()
    for name, t in trainers.items():
        t.finish()
        results[name] = t.results()

    return results
