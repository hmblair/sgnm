import torch
import sgnm

batch = 2
residues = 200
coords = 3

# Generate random coordinates

x = torch.randn(batch, residues, coords)

# Generate a random SHAPE profile

profile = torch.randn(batch, residues)

# Init the objective

path = "../checkpoints/weights.pth"
objective = sgnm.score(path)

# Compute the score

mae, _ = objective(profile, x)
print(f"MAE: {mae}")
