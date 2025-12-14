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

path = "../checkpoints/weights.pth"
module = sgnm.SGNM.load()

# Init the profile

with torch.no_grad():
    profile = module.ciffy(rna)

# Expose a random base

profile_exp = profile.clone()
profile_exp[8] = 1.0

# Relax via dihedral optimization

steps = 100
lr = 1e-2
alpha = 0.01

# Create a copy for relaxation
relaxed = deepcopy(rna)


# Dihedral optimization model
class DihedralModel(nn.Module):
    def __init__(self, init_dihedrals):
        super().__init__()
        self.dihedrals = nn.Parameter(init_dihedrals.clone())


model = DihedralModel(relaxed.dihedrals)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print(f"{'Step':>5} {'MSE':>10} {'RMSD':>10}")
print("-" * 30)

for step in range(steps):
    optimizer.zero_grad()

    # Apply current dihedrals to structure
    relaxed.dihedrals = model.dihedrals

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
relaxed.write("new.cif")

plt.plot(profile, color="red", alpha=0.5, label="Original")
plt.plot(profile_new, color="blue", alpha=0.5, label="Modified")
plt.legend(fontsize=13)
plt.xlabel("Residue", fontsize=13)
plt.ylabel("Normalized Reactivity", fontsize=13)
plt.show()
plt.close()
