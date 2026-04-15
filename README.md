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
| GNM | 4,626 | +0.39 / +0.35 | `gnm-checkpoint.pth` |
| Equivariant | 78,798 | +0.63 / +0.75 | `equivariant-checkpoint.pth` |

## Usage

### Prediction

```python
import ciffy
import sgnm

# Load any model from a checkpoint (auto-detects model type)
model = sgnm.load("checkpoint.pth")

# Predict
poly = ciffy.load("structure.cif", backend="torch")
reactivity = model.ciffy(poly)  # (N, 2) for [SHAPE, DMS]
```

### Scoring

Score a structure against an experimental reactivity profile:

```python
import torch
import sgnm
from sgnm.config import ScoringConfig

model = sgnm.load("checkpoint.pth")
scorer = sgnm.StructureScorer(model, ScoringConfig(metric="correlation"))

poly = ciffy.load("structure.cif", backend="torch")
score = scorer.score(experimental_profile, poly)
```

Available metrics: `mae`, `mse`, `correlation`.

Use `channels` to score specific output channels (e.g. SHAPE only):

```python
scorer = sgnm.StructureScorer(model, ScoringConfig(metric="correlation", channels=[0]))
```

### Ranking

Rank decoy structures by agreement with experimental reactivity:

```python
import sgnm

model = sgnm.load("checkpoint.pth")
result = sgnm.rank(model, "decoys/", reactivity)

print(f"Best: {result.best.file} (score={result.best.score:.4f})")
result.to_csv("rankings.csv")
```

Or via the command line:

```bash
python scripts/rank.py decoys/ \
    --model gnm --weights checkpoint.pth \
    --profile profiles.h5 --fasta ref.fasta --id 8UYS_A
```

### Training

Training is configured via TOML files. See `configs/example.toml` for a template.

```bash
python scripts/train.py config.toml
```

#### Data format

Reactivity data is provided as one HDF5 file per condition, each containing
a `reactivity` dataset of shape `(N, L)`. Paths support an optional
`:key` suffix for datasets within a file (e.g. `profiles.h5:PDB130-2A3/reactivity`).

```toml
[data]
reactivity_paths = ["2a3.h5", "dms.h5"]
fasta_path = "sequences.fasta"
structures_dir = "structures/"

[train]
learning_rate = 1e-3
max_epochs = 100
loss_type = "correlation"

[gnm]
dim = 32
layers = 2
```

## Contact

Email: `hmblair@stanford.edu`
