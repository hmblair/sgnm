"""Predict per-residue reactivity from an RNA structure using the GNM model."""

import ciffy
import sgnm

# Load a trained SGNM checkpoint
model = sgnm.load("path/to/checkpoint.pth")
model.eval()

# Load an RNA structure and predict reactivity
poly = ciffy.load("structures/9FO9.cif", backend="torch").strip()
pred = model.ciffy(poly)

print(f"Structure: {poly.size(ciffy.RESIDUE)} residues")
print(f"Prediction shape: {pred.shape}")  # (N, out_channels)
print(f"Per-residue predictions:\n{pred[:5]}")
