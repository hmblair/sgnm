"""
Latent space relaxation using PolymerFlowModel.

Optimizes the latent representation of an RNA structure to match a target
reactivity profile, while regularizing to stay close to the original structure.
"""

import ciffy
import torch
import torch.nn as nn
import sgnm
import matplotlib.pyplot as plt
from copy import deepcopy

from ciffy.nn.flow import PolymerFlowModel

# Load the polymer

file = "structures/9FO9.cif"
poly = ciffy.load(file, backend="torch")

# Get the RNA (chain D)

rna = poly.strip()
rna, _ = rna.center()
n_residues = rna.size(ciffy.RESIDUE)

# Init the objective

module = sgnm.SGNM.load("../checkpoints/weights.pth")

# Init the profile

with torch.no_grad():
    profile = module.ciffy(rna)
profile_exp = profile.clone()

# Expose a random base pair

profile_exp[8] = 1.0
profile_exp[24] = 1.0

# Shield a random base

# profile_exp[16] = 0.0

# Load trained flow model

flow_model = PolymerFlowModel.load("/Users/hmblair/models/rna_flow")
flow_model.eval()

# Move to same device as RNA
device = rna.coordinates.device
flow_model = flow_model.to(device)

# Get sequence (int array of residue types)
sequence = rna.sequence

# Encode initial structure to latent space
with torch.no_grad():
    z_init = flow_model.encode_polymer(rna)  # (n_residues, latent_dim)

print(f"Encoded {n_residues} residues to latent dim {flow_model.latent_dim}")


# Latent optimization model
class LatentModel(nn.Module):
    def __init__(self, init_latents):
        super().__init__()
        self.latents = nn.Parameter(init_latents.clone())


model = LatentModel(z_init)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# Relaxation parameters
steps = 1000
alpha = 0.01  # RMSD regularization weight

print(f"Optimizing {model.latents.numel()} latent dimensions")
print(f"Latent shape: {model.latents.shape}")
print(f"{'Step':>5} {'MSE':>10} {'RMSD':>10}")
print("-" * 30)

# Create a copy for relaxation
relaxed = deepcopy(rna)

for step in range(steps):
    optimizer.zero_grad()

    # Decode latents to coordinates
    coords = flow_model.decode(model.latents, sequence)  # (N, 3)

    # Project to fix geometry (implicit=True for clean gradients)
    coords = flow_model.project_geometry(coords, sequence, n_steps=3)

    # Update relaxed structure
    relaxed.coordinates = coords

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

# Final structure with more Newton steps for polish
with torch.no_grad():
    final_coords = flow_model.decode(model.latents, sequence)
    final_coords = flow_model.project_geometry(final_coords, sequence, n_steps=5)
    relaxed.coordinates = final_coords
    profile_new = module.ciffy(relaxed).detach()

rna.write("orig.cif")
relaxed.write("new_latent.cif")

plt.plot(profile.cpu(), color="red", alpha=0.5, label="Original")
plt.plot(profile_new.cpu(), color="blue", alpha=0.5, label="Modified")
plt.legend(fontsize=13)
plt.xlabel("Residue", fontsize=13)
plt.ylabel("Normalized Reactivity", fontsize=13)
plt.savefig("profile_comparison.png", dpi=150)
plt.show()
plt.close()

print(f"\nFinal RMSD from original: {ciffy.rmsd(rna, relaxed).item():.3f} A")
