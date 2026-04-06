# SGNM

Predict RNA chemical probing reactivity (SHAPE/DMS) from 3D structure.

## Overview

`sgnm` predicts per-residue SHAPE and DMS reactivity profiles from RNA 3D structures. Two model architectures:

- **SGNM**: Learnable Gaussian Network Model. Computes k independent adjacency matrices from distance and orientation embeddings (via radial basis functions), runs batched GNM variance computation, and projects to per-residue (SHAPE, DMS) predictions.
- **EquivariantReactivityModel**: SO(3)-equivariant transformer using [flash-eq](https://github.com/hmblair/flash-eq). Operates at the atom level with k-NN graphs and ciffy's PolymerEmbedding.

Both models output `(N, 2)` for SHAPE and DMS channels, accept [ciffy](https://github.com/hmblair/ciffy) `Polymer` objects via `.ciffy(polymer)`, and are differentiable.

## Installation

```bash
pip install -e .

# For the equivariant model (requires CUDA):
pip install -e '.[equivariant]'
```

## Pre-trained Weights

Download from [GitHub Releases](https://github.com/hmblair/sgnm/releases/latest):

| Model | Params | Val Correlation (SHAPE / DMS) | File |
|-------|--------|-------------------------------|------|
| GNM | 4,626 | +0.41 / +0.34 | `gnm-checkpoint.pth` |
| Equivariant | 78,798 | +0.70 / +0.60 | `equivariant-checkpoint.pth` |

## Usage

### Prediction

```python
import ciffy

# GNM model
from sgnm.models import SGNM
model = SGNM.load("gnm-checkpoint.pth")

# Or equivariant model (requires flash-eq + CUDA)
from sgnm.equivariant import EquivariantReactivityModel
model = EquivariantReactivityModel.load("equivariant-checkpoint.pth")

# Predict
poly = ciffy.load("structure.cif", backend="torch")
reactivity = model.ciffy(poly)  # (N, 2) for [SHAPE, DMS]
```

### Training

Training is configured via TOML. Include `[gnm]` and/or `[equivariant]` sections:

```toml
[data]
reactivity_path = "profiles.h5"
fasta_path = "ref.fasta"
structures_dir = "structures/"

[train]
learning_rate = 1e-3
max_epochs = 100
loss_type = "correlation"
wandb_project = "sgnm"

[gnm]
dim = 32
layers = 2

[equivariant]
embed_dim = 32
hidden_layers = 4
```

```bash
python scripts/train.py config.toml
```

Both models train in lockstep on the same data with per-channel SHAPE/DMS metrics logged to wandb.

### Ranking

Score and rank decoy structures against an experimental reactivity profile:

```bash
python scripts/rank.py decoys/ \
    --model gnm --weights checkpoints/gnm/best.pth \
    --profile profiles.h5 --fasta ref.fasta --id 8UYS_A
```

### Evaluation

Compare SGNM rankings against structural quality metrics (TM-score, RMSD):

```bash
python scripts/evaluate_ranking.py decoys/ \
    --model gnm --weights checkpoints/gnm/best.pth \
    --profile profiles.h5 --fasta ref.fasta --id 8UYS_A \
    --scores casp_scores.csv --metric tm_score --target R1149
```

## Contact

Email: `hmblair@stanford.edu`
