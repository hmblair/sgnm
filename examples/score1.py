import torch
import sgnm

residues = 200
coords = 3

# Generate random coordinates

x = torch.randn(residues, coords)

# Generate a random SHAPE profile

profile = torch.randn(residues)

# Init the objective

path = "../checkpoints/weights.pth"
objective = sgnm.score(path)

# Compute the score

score = objective(profile, x)
print(f"MAE: {score}")
