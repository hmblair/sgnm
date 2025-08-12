import argparse
import torch
import ciffy
import matplotlib.pyplot as plt
import h5py
from .layers import SGNM

parser = argparse.ArgumentParser()
parser.add_argument(
    'file',
    help='The input .cif file.',
)
parser.add_argument(
    '--weights',
    help='The location of the trained weights.',
)
parser.add_argument(
    '--chain',
    help='Use this chain for prediction only.',
    type=int,
)
parser.add_argument(
    '--out',
    help='Save the reactivity profile to this location.',
)

if __name__ == "__main__":

    args = parser.parse_args()

    # Init the module

    if args.weights:
        module = SGNM.load(args.weights)
    else:
        module = SGNM.load()
    module.eval()

    # Load and preprocess

    poly = ciffy.load(args.file)
    if args.chain is not None:
        poly = poly.select(args.chain)

    poly = poly.subset(ciffy.RNA)
    n = poly.size(ciffy.RESIDUE)

    if n == 0:
        raise ValueError(f"No residues found in PDB {poly.id()}.")

    x = torch.arange(n)
    y = torch.ones(n) * torch.nan

    ix = poly.resolved()
    stripped = poly.strip()

    # Predict

    with torch.no_grad():
        y[ix] = module.ciffy(stripped)

    # Save

    if args.out:
        with h5py.File(args.out, "w") as f:
            f.create_dataset(f"{poly.id()}", data=y.numpy())

    # Plot

    low = 0
    high = 0

    for chain in poly.chains():

        high += chain.size(ciffy.RESIDUE)
        plt.plot(x[low:high], y[low:high], label=f"Chain {chain.names[0]}")
        low += chain.size(ciffy.RESIDUE)

    plt.title(f"PDB {poly.id()}", fontsize=14)
    plt.xlabel("Nucleotide", fontsize=13)
    plt.ylabel("Normalized Reactivity", fontsize=13)
    plt.legend(fontsize=13)
    plt.show()
    plt.close()
