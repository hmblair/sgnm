"""Evaluate how prediction quality degrades with structural noise.

For each noise level, perturbs all validation structures and measures
correlation between predicted and experimental reactivity.

Usage:
    python scripts/noise_sweep.py configs/config-gnm.toml \
        --noise 0 0.5 1 2 5 10 20 \
        --repeats 5 \
        -o figures/noise_sweep.png
"""
import argparse
import sys
from pathlib import Path

import torch
import numpy as np

from sgnm.config import DataConfig, ModelConfig
from sgnm.data import ReactivityDataset, load_reactivity_index
from sgnm.losses import pearson_correlation


def evaluate_at_noise(model, dataset, noise_std, repeats=1, device="cpu"):
    """Evaluate model on dataset with given noise level.

    Returns per-channel correlations averaged over repeats.
    """
    model.eval()
    all_corrs = []

    for _ in range(repeats):
        shape_corrs, dms_corrs = [], []

        with torch.no_grad():
            for sample in dataset:
                if sample is None:
                    continue
                sample = sample.to(device)

                poly = sample.polymer
                if noise_std > 0:
                    poly = poly.copy()
                    poly.coordinates = (
                        poly.coordinates
                        + torch.randn_like(poly.coordinates) * noise_std
                    )

                try:
                    pred = model.ciffy(poly)
                except (ValueError, RuntimeError):
                    continue

                target = sample.reactivity
                mask = sample.mask
                if mask.size(0) != pred.size(0):
                    continue

                pred = pred[mask]
                target = target[mask]

                if pred.dim() > 1 and pred.size(1) >= 2:
                    if pred[:, 0].std() > 1e-8 and target[:, 0].std() > 1e-8:
                        shape_corrs.append(
                            pearson_correlation(pred[:, 0], target[:, 0]).item()
                        )
                    if pred[:, 1].std() > 1e-8 and target[:, 1].std() > 1e-8:
                        dms_corrs.append(
                            pearson_correlation(pred[:, 1], target[:, 1]).item()
                        )

        if shape_corrs and dms_corrs:
            all_corrs.append((np.mean(shape_corrs), np.mean(dms_corrs)))

    if not all_corrs:
        return 0.0, 0.0, 0.0, 0.0

    shape_vals = [c[0] for c in all_corrs]
    dms_vals = [c[1] for c in all_corrs]
    return (
        np.mean(shape_vals), np.std(shape_vals),
        np.mean(dms_vals), np.std(dms_vals),
    )


def main():
    import tomllib

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Training config TOML (for data paths and model)")
    parser.add_argument("--weights", required=True, help="Path to model checkpoint")
    parser.add_argument("--model", default="gnm", choices=["gnm", "equivariant"])
    parser.add_argument("--noise", nargs="+", type=float,
                        default=[0, 0.5, 1, 2, 5, 10, 20])
    parser.add_argument("--repeats", type=int, default=5,
                        help="Repeats per noise level (for error bars)")
    parser.add_argument("--output", "-o", default="figures/noise_sweep.png")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)

    data_config = DataConfig(**cfg.get("data", {}))
    device = cfg.get("train", {}).get("device", "cpu")

    # Load model
    if args.model == "gnm":
        from sgnm.models import SGNM
        model = SGNM.load(args.weights)
    else:
        from sgnm.equivariant import EquivariantReactivityModel
        model = EquivariantReactivityModel.load(args.weights)
    model = model.to(device)

    # Load validation data
    index = load_reactivity_index(
        data_config.reactivity_path, data_config.fasta_path, data_config.data_format
    )
    from ciffy.nn import PolymerDataset
    import ciffy
    poly_dataset = PolymerDataset(
        data_config.structures_dir,
        scale=ciffy.Scale.CHAIN,
        molecule_types=ciffy.Molecule.RNA,
        max_chains=data_config.max_chains,
    )
    splits = poly_dataset.split(
        train=data_config.train_split,
        val=data_config.val_split,
        seed=data_config.seed,
    )
    val_structures = splits[1] if len(splits) > 1 else splits[0]
    dataset = ReactivityDataset(val_structures, index)

    # Sweep
    print(f"{'Noise (A)':>10} {'SHAPE':>8} {'±':>6} {'DMS':>8} {'±':>6}")
    print("-" * 45)

    results = []
    for noise in args.noise:
        s_mean, s_std, d_mean, d_std = evaluate_at_noise(
            model, dataset, noise, args.repeats, device
        )
        results.append((noise, s_mean, s_std, d_mean, d_std))
        print(f"{noise:10.1f} {s_mean:8.4f} {s_std:6.4f} {d_mean:8.4f} {d_std:6.4f}")

    # Plot
    import hmbp
    noise_vals = [r[0] for r in results]
    shape_means = [r[1] for r in results]
    dms_means = [r[3] for r in results]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    hmbp.quick.lines(
        [shape_means, dms_means],
        noise_vals,
        labels=["SHAPE", "DMS"],
        xlabel="Coordinate Noise (Angstroms)",
        ylabel="Pearson Correlation",
        title=f"Prediction Quality vs Structural Noise ({args.model.upper()})",
        path=args.output,
    )
    print(f"\nPlot saved to {args.output}")


if __name__ == "__main__":
    main()
