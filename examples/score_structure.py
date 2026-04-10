"""Score a structure against an experimental reactivity profile."""

import torch
import ciffy
import sgnm

# Load model and wrap in a scorer
model = sgnm.SGNM.load("path/to/checkpoint.pth")
scorer = sgnm.StructureScorer(model, sgnm.ScoringConfig(metric="mae"))

# Load structure and experimental profile
poly = ciffy.load("structures/9FO9.cif", backend="torch").strip()
experimental = torch.randn(poly.size(ciffy.RESIDUE))  # replace with real data

# Score: how well does the model's prediction match the experimental profile?
result = scorer.score_polymer(experimental, poly)
print(f"MAE: {result.score_value:.4f}")

# Can also score a .cif file directly
result = scorer.score_cif_file("structures/9FO9.cif", experimental)
print(f"MAE: {result.score_value:.4f}")
print(f"File: {result.metadata['name']}")
