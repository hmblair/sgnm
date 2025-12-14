import torch
import ciffy
import sgnm

# Load the polymer

file = "structures/9GBZ.cif"
poly = ciffy.load(file, backend="torch")

# Get the RNA (chain D)

ix = 3
rna = poly.by_index(ix).strip()

# Generate a random SHAPE profile

residues = rna.size(ciffy.RESIDUE)
profile = torch.randn(residues)

# Init the objective

path = "../checkpoints/weights.pth"
objective = sgnm.score(path)

# Compute the score

mae, _ = objective.ciffy(profile, rna)
print(f"MAE: {mae}")
