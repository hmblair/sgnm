"""Evaluate models on a dataset's validation split."""
import argparse
import sys
import tomllib

import torch

from sgnm.config import DataConfig
from sgnm.data import ReactivityDataset, load_reactivity_index
from sgnm.scoring import metric


def evaluate(model, val_dataset, device):
    """Return mean correlation over val set."""
    model.eval()
    model.to(device)
    corrs = []
    with torch.no_grad():
        for sample in val_dataset:
            if sample is None:
                continue
            sample = sample.to(device)
            try:
                pred = model.ciffy(sample.polymer)
            except Exception:
                continue
            mask = sample.mask
            if mask.sum() < 3:
                continue
            corr = metric(pred, sample.reactivity, mask)
            corrs.append(corr)
    if not corrs:
        return {}
    stacked = torch.stack(corrs)
    mean = stacked.mean(dim=0)
    if mean.dim() == 0:
        return {"corr": mean.item()}
    labels = ["shape", "dms"] if mean.shape[0] == 2 else [str(i) for i in range(mean.shape[0])]
    result = {"corr": mean.mean().item()}
    for i, label in enumerate(labels):
        result[label] = mean[i].item()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="TOML config (for data paths and splits)")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="label:path pairs, e.g. gnm-old:checkpoints/gnm-old/best.pth")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    data_config = DataConfig(**cfg.get("data", {}))

    from ciffy.biochemistry import Scale, Molecule
    from ciffy.nn import PolymerDataset

    index = load_reactivity_index(data_config.reactivity_paths, data_config.fasta_path)
    structures = PolymerDataset(
        data_config.structures_dir, scale=Scale.CHAIN,
        molecule_types=Molecule.RNA, max_chains=data_config.max_chains,
    )
    splits = structures.split(
        train=data_config.train_split, val=data_config.val_split,
        test=data_config.test_split, seed=data_config.seed,
    )
    val_dataset = ReactivityDataset(splits[1], index)
    print(f"Validation samples: {len(val_dataset)}")

    for spec in args.checkpoints:
        label, path = spec.split(":", 1)
        print(f"\nEvaluating {label} ({path})...")
        if "equivariant" in label:
            from sgnm.equivariant import EquivariantReactivityModel
            model = EquivariantReactivityModel.load(path)
        else:
            from sgnm.models import SGNM
            model = SGNM.load(path)
        result = evaluate(model, val_dataset, args.device)
        parts = [f"{k}={v:+.4f}" for k, v in result.items()]
        print(f"  [{label}] {' | '.join(parts)}")


if __name__ == "__main__":
    main()
