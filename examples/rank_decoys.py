"""Rank a set of decoy structures by agreement with experimental reactivity."""

import torch
import sgnm

# Load a trained model
model = sgnm.load("path/to/checkpoint.pth")

# Experimental reactivity profile aligned to structure length
# Shape (L,) for single-channel or (L, C) for multi-channel
reactivity = torch.randn(100)  # replace with real data

# Rank all .cif files in a directory
result = sgnm.rank(model, "structures/", reactivity)

# Inspect results (sorted best-first)
print(f"Best:  {result.best.file} (score={result.best.score:.4f})")
print(f"Worst: {result.worst.file} (score={result.worst.score:.4f})")

for entry in result.entries:
    print(f"  #{entry.rank} {entry.file}: {entry.score:.4f}")

# Export to CSV
result.to_csv("rankings.csv")
