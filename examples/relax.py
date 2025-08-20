import torch
import ciffy
import sgnm
import matplotlib.pyplot as plt

# Load the polymer

file = "structures/9FO9.cif"
poly = ciffy.load(file)

# Get the RNA (chain D)

rna = poly.strip().frame()
rna, _ = rna.center()

# Generate a random SHAPE profile

residues = rna.size(ciffy.RESIDUE)

# Init the objective

path = "../checkpoints/weights.pth"
module = sgnm.SGNM.load(path)
objective = sgnm.score(path)

# Init the profile

with torch.no_grad():
    profile = module.ciffy(rna)

# Expose a random base

profile_exp = profile.clone()
profile_exp[8] = 1.0

# Relax

steps = 100
lr = 1E-1
alpha = 1E-3
relaxed = objective.relax(profile_exp, rna, steps=steps, lr=lr, alpha=alpha)

with torch.no_grad():
    profile_new = module.ciffy(relaxed).detach()

rna.write("orig.pdb")
relaxed.write("new.pdb")

plt.plot(profile, color="red", alpha=0.5, label="Original")
plt.plot(profile_new, color="blue", alpha=0.5, label="Modified")
plt.legend(fontsize=13)
plt.xlabel("Residue", fontsize=13)
plt.ylabel("Normalized Reactivity", fontsize=13)
plt.show()
plt.close()
