"""Predict per-residue reactivity from an RNA structure.

Usage:
    python -m sgnm structure.cif --weights checkpoints/gnm/best.pth
    python -m sgnm structure.cif --weights checkpoints/gnm/best.pth --out predictions.h5
"""
import argparse

import torch
import ciffy

from .models import SGNM


def main():
    parser = argparse.ArgumentParser(description="Predict reactivity from RNA 3D structure")
    parser.add_argument("file", help="Input .cif file")
    parser.add_argument("--weights", required=True, help="Path to model checkpoint")
    parser.add_argument("--out", help="Save predictions to HDF5 file")
    args = parser.parse_args()

    model = SGNM.load(args.weights)
    model.eval()

    poly = ciffy.load(args.file, backend="torch")

    with torch.no_grad():
        pred = model.ciffy(poly)  # (N, out_channels)

    if args.out:
        import h5py
        with h5py.File(args.out, "w") as f:
            f.create_dataset("reactivity", data=pred.numpy())

    # Print summary
    n = pred.size(0)
    channels = pred.size(1) if pred.dim() > 1 else 1
    print(f"Structure: {args.file}")
    print(f"Residues: {n}")
    print(f"Channels: {channels}")
    for c in range(channels):
        col = pred[:, c] if pred.dim() > 1 else pred
        print(f"  Channel {c}: mean={col.mean():.4f}, std={col.std():.4f}")


if __name__ == "__main__":
    main()
