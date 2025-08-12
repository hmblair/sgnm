import argparse
import torch
import ciffy
import matplotlib.pyplot as plt
from layers import SGNM

parser = argparse.ArgumentParser()
parser.add_argument(
    'file',
    help='The input .cif file.',
)
parser.add_argument(
    '--weights',
    help='The location of the trained weights.',
    required=True,
)

if __name__ == "__main__":

    args = parser.parse_args()

    # Init the module

    module = SGNM.load(args.weights)
    module.eval()

    # Predict

    poly = ciffy.load(args.file)
    with torch.no_grad():
        pred = module.ciffy(poly)

    # Plot

    plt.plot(pred, color="red")
    plt.show()
    plt.close()
