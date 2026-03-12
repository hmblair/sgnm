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
from ciffy.biochemistry import Residue

# Helper to compute O3'-P backbone distances
def get_backbone_distances(coords, sequence, flow_model):
    """Compute O3'(i) to P(i+1) distances for backbone bond loss."""
    res_type_map = {
        Residue.A.value: Residue.A,
        Residue.C.value: Residue.C,
        Residue.G.value: Residue.G,
        Residue.U.value: Residue.U,
    }

    offset = 0
    p_pos, o3p_pos = [], []

    for i, res_type_val in enumerate(sequence):
        res_type = int(res_type_val)
        res_model = flow_model._get_model(res_type)
        n_atoms = res_model.n_atoms
        res_coords = coords[offset:offset + n_atoms]
        atom_filter = [int(a) for a in res_model.atoms]
        res_enum = res_type_map.get(res_type)
        if res_enum:
            try:
                p_idx = atom_filter.index(res_enum.P.value)
                o3p_idx = atom_filter.index(res_enum.O3p.value)
                p_pos.append(res_coords[p_idx])
                o3p_pos.append(res_coords[o3p_idx])
            except ValueError:
                pass
        offset += n_atoms

    p_pos = torch.stack(p_pos)
    o3p_pos = torch.stack(o3p_pos)
    return torch.norm(o3p_pos[:-1] - p_pos[1:], dim=-1)

# Load the polymer

file = "structures/9FO9.cif"
poly = ciffy.load(file, backend="torch")

# Get the RNA (chain D)

rna_full = poly.strip()
rna_full, _ = rna_full.center()
n_residues = rna_full.size(ciffy.RESIDUE)

# Load trained flow model

flow_model = PolymerFlowModel.load("/Users/hmblair/models/rna_flow_9d")
flow_model.eval()

# Create a filtered template with only the atoms the flow model handles
# This ensures coordinate dimensions match during optimization
sequence_str = rna_full.sequence_str()
rna = ciffy.template(sequence_str, atoms=flow_model.atom_filter, backend="torch")

# Copy coordinates from full structure to filtered template
# The flow model's encode_polymer handles this filtering internally
with torch.no_grad():
    z_init = flow_model.encode_polymer(rna_full)
    initial_coords = flow_model.decode(z_init, rna.sequence, latent_bound=None)
    rna.coordinates = initial_coords.detach()

# Center the filtered structure
rna, _ = rna.center()

print(f"Full structure: {rna_full.size()} atoms")
print(f"Filtered template: {rna.size()} atoms (flow model subset)")

# Move to same device as RNA
device = rna.coordinates.device
flow_model = flow_model.to(device)

# Get sequence (int array of residue types)
sequence = rna.sequence

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

# Re-encode the centered structure
with torch.no_grad():
    z_init = flow_model.encode_polymer(rna)

print(f"Encoded {n_residues} residues to latent dim {flow_model.latent_dim}")


# Latent optimization model
class LatentModel(nn.Module):
    def __init__(self, init_latents):
        super().__init__()
        self.latents = nn.Parameter(init_latents.clone())


model = LatentModel(z_init)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# Relaxation parameters
steps = 100
alpha = 0.01  # RMSD regularization weight
beta = 0.1   # Gaussian regularization weight (keeps latents near N(0,1))
gamma = 1.0  # Backbone bond regularization weight
target_bond = 1.6  # Ideal O3'-P distance in Angstroms
latent_bound = 5.0  # Soft bound on latents to prevent gradient explosion
max_grad_norm = 10.0  # Gradient clipping threshold

print(f"Optimizing {model.latents.numel()} latent dimensions")
print(f"Latent shape: {model.latents.shape}")
print(f"Latent bound: {latent_bound}, Grad clip: {max_grad_norm}")
print(f"alpha={alpha} (RMSD), beta={beta} (Gaussian), gamma={gamma} (backbone)")
print(f"{'Step':>5} {'MSE':>10} {'RMSD':>10} {'Backbone':>10} {'GradNorm':>10}")
print("-" * 55)

# Create a copy for relaxation
relaxed = deepcopy(rna)

for step in range(steps):
    optimizer.zero_grad()

    # Decode latents to coordinates (with soft bounding to prevent gradient explosion)
    coords = flow_model.decode(model.latents, sequence, latent_bound=latent_bound)

    # Project to fix geometry (implicit=True for clean gradients)
    coords = flow_model.project_geometry(coords, sequence, n_steps=10)

    # Update relaxed structure
    relaxed.coordinates = coords

    # Compute score against target profile
    pred = module.ciffy(relaxed)
    mse = ((pred - profile_exp) ** 2).mean()

    # RMSD regularization to prevent drift
    rmsd = ciffy.rmsd(rna, relaxed)

    # Gaussian regularization on latents (keeps them near N(0,1))
    latent_reg = (model.latents ** 2).mean()

    # Backbone bond regularization (keep O3'-P distances near ideal)
    backbone_dists = get_backbone_distances(coords, sequence, flow_model)
    backbone_loss = ((backbone_dists - target_bond) ** 2).mean()

    # Combined loss
    loss = mse + alpha * rmsd + beta * latent_reg + gamma * backbone_loss

    loss.backward()

    # Gradient clipping for stability
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step()

    if step % 10 == 0 or step == steps - 1:
        print(f"{step:5d} {mse.item():10.4f} {rmsd.item():10.4f} {backbone_loss.item():10.4f} {grad_norm.item():10.2f}")

# Final structure with more Newton steps for polish
with torch.no_grad():
    final_coords = flow_model.decode(model.latents, sequence, latent_bound=latent_bound)
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
