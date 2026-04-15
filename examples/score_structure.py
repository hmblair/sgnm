"""Score a structure against an experimental reactivity profile."""

import torch
import ciffy
import sgnm
from sgnm.config import ScoringConfig

# Load model and wrap in a scorer
model = sgnm.load("path/to/checkpoint.pth")
scorer = sgnm.StructureScorer(model, ScoringConfig(metric="mae"))

# Load structure and experimental profile
poly = ciffy.load("structures/9FO9.cif", backend="torch").strip()
experimental = torch.randn(poly.size(ciffy.RESIDUE))  # replace with real data

# Score: how well does the model's prediction match the experimental profile?
score = scorer.score(experimental, poly)
print(f"MAE: {score:.4f}")
