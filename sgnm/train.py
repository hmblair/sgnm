from __future__ import annotations
import os
import h5py
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import ciffy
import dlu
from .layers import SGNM, SGNM2


def _get_data(path: str) -> dict:

    with h5py.File(path, 'r') as f:
        names = f['id_strings'][0].astype(str)
        reacs = torch.from_numpy(f['r_norm'][:])
        seqs = f['sequences'][0].astype(str)

    return dict(zip(names, zip(seqs, reacs)))


DMS = 0
SHAPE = 1

PATH = "/Users/hmblair/academic/data/rna-libraries/pdb/profiles/raw2/train2_pdb130.hdf5"
OFFSET = 51

NAME = "Train"
DIR = "/Users/hmblair/academic/data/rna-libraries/pdb/structures/train"
RUN = ""

DIM = 32
LR = 1E-3

module = SGNM2(DIM)
opt = torch.optim.Adam(module.parameters(), lr=LR)
print(f"Parameters: {dlu.params(module)}")

data = _get_data(PATH)
pbar = dlu.pbar(os.listdir(DIR), opt, NAME, wandb=RUN)

pbar.epoch()

for name in pbar:

    path = os.path.join(DIR, name)
    poly = ciffy.load(path)

    try:

        for chain in poly.chains(ciffy.RNA):

            cid = chain.id(0)
            if cid not in data:
                continue

            stripped = chain.frame().strip()
            if stripped.empty():
                continue
            if stripped.size(ciffy.RESIDUE) * 3 != stripped.coordinates.size(0):
                continue

            seq, reac = data[cid]
            reac = reac[..., SHAPE]
            reac[reac < 0] = 0

            low = reac.size(0) - OFFSET - stripped.size(ciffy.RESIDUE)
            high = reac.size(0) - OFFSET

            if low < 0:
                continue

            seq = seq[low:high]
            reac = dlu.norm(reac[low:high])

            if seq.lower() != stripped.str():
                continue

            pred = module.ciffy(chain)
            loss = torch.mean((reac - pred).abs())

            pbar.update(loss)

            if pbar.step() % 10 == 0:
                plt.plot(reac, color="red", alpha=0.5)
                plt.plot(pred.detach(), color="blue", alpha=0.5)
                plt.show(block=False)
                plt.pause(1)
                plt.close()

    except Exception as e:
        print(e)
        continue
