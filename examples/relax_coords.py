"""
Coordinate-space relaxation.

Optimizes atom coordinates directly to match a target reactivity profile,
while regularizing with RMSD to stay close to the original structure.
"""

import ciffy
import torch
import torch.nn as nn
import sgnm
import matplotlib.pyplot as plt
from copy import deepcopy

# Load the polymer

file = "structures/9FO9.cif"
poly = ciffy.load(file, backend="torch")

# Get the RNA (chain D)

rna = poly.strip()
rna, _ = rna.center()
residues = rna.size(ciffy.RESIDUE)

# Init the objective

module = sgnm.SGNM.load()

# Init the profile

with torch.no_grad():
    profile = module.ciffy(rna)
profile_exp = profile.clone()

# Expose a random base pair

profile_exp[8] = 1.0
profile_exp[24] = 1.0

# Shield a random base

# profile_exp[16] = 0.0

# Relax via coordinate optimization

steps = 1000
lr = 1e-3
alpha = 0.001  # RMSD regularization weight

# Create a copy for relaxation
relaxed = deepcopy(rna)


# Coordinate optimization model
class CoordinateModel(nn.Module):
    def __init__(self, init_coords):
        super().__init__()
        self.coords = nn.Parameter(init_coords.clone())


model = CoordinateModel(relaxed.coordinates)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print(f"Optimizing {model.coords.numel()} coordinate values ({model.coords.shape[0]} atoms)")
print(f"{'Step':>5} {'MSE':>10} {'RMSD':>10}")
print("-" * 30)

for step in range(steps):
    optimizer.zero_grad()

    # Apply current coordinates to structure
    relaxed.coordinates = model.coords

    # Compute score against target profile
    pred = module.ciffy(relaxed)
    mse = ((pred - profile_exp) ** 2).mean()

    # RMSD regularization to prevent drift
    rmsd = ciffy.rmsd(rna, relaxed)

    # Combined loss
    loss = mse + alpha * rmsd

    loss.backward()
    optimizer.step()

    if step % 10 == 0 or step == steps - 1:
        print(f"{step:5d} {mse.item():10.4f} {rmsd.item():10.4f}")

with torch.no_grad():
    profile_new = module.ciffy(relaxed).detach()

rna.write("orig.cif")
relaxed.write("new_coords.cif")

plt.plot(profile, color="red", alpha=0.5, label="Original")
plt.plot(profile_new, color="blue", alpha=0.5, label="Modified")
plt.legend(fontsize=13)
plt.xlabel("Residue", fontsize=13)
plt.ylabel("Normalized Reactivity", fontsize=13)
plt.savefig("profile_comparison_coords.png", dpi=150)
plt.show()
plt.close()

print(f"\nFinal RMSD from original: {ciffy.rmsd(rna, relaxed).item():.3f} A")
