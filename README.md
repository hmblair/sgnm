# SGNM

Predict RNA chemical probing reactivity from 3D structure.

## Overview

`sgnm` predicts SHAPE and DMS reactivity profiles from RNA tertiary structures. It provides two model architectures:

- **SGNM**: Gaussian Network Model with learnable radial basis function embeddings. Uses distance and base orientation pathways with GNM variance computation (delegated to [ciffy](https://github.com/hmblair/ciffy)).
- **EquivariantReactivityModel**: SO(3)-equivariant transformer using [flash-eq](https://github.com/hmblair/flash-eq). Operates at the atom level with k-NN graphs and reduces to per-residue predictions.

Both models are fast, differentiable, and accept [ciffy](https://github.com/hmblair/ciffy) `Polymer` objects directly.

## Installation

```bash
git clone https://github.com/hmblair/sgnm
cd sgnm
pip install -e .

# For the equivariant model (requires CUDA):
pip install -e '.[equivariant]'
```

## Quick Start

### SGNM Model

```python
from sgnm import SGNM

# Load pre-trained model
model = SGNM.load("weights.pth")

# Or use non-parametric baseline
model = SGNM.load()

# Predict from ciffy Polymer
profile = model.ciffy(polymer)

# Or from raw tensors (coords: (N, 3), frames: (N, 3, 3))
profile = model(coords, frames)
```

Frames are formed by the C2-C4-C6 atom triplet.

### Equivariant Model

```python
import sgnm

# Requires flash-eq (CUDA)
model = sgnm.equivariant(embed_dim=32, hidden_layers=4, k_neighbors=16)

# Takes Polymer directly, returns (n_residues, 2) for [SHAPE, DMS]
predictions = model(polymer)
```

### Scoring

Compute MAE between predicted and experimental SHAPE profiles:

```python
import sgnm

scorer = sgnm.score("weights.pth")

# From tensors
mae, pred = scorer(profile, coords, frames)

# From ciffy Polymer
mae, pred = scorer.ciffy(profile, polymer)
```

### Structure Relaxation

Optimize structure coordinates to match a target profile:

```python
import sgnm

scorer = sgnm.score("weights.pth")
relaxed = scorer.relax(target_profile, polymer, steps=100)
```

## Training

```python
from sgnm import train_sgnm, ModelConfig, DataConfig, TrainConfig

results = train_sgnm(
    model_config=ModelConfig(dim=32, layers=2),
    data_config=DataConfig(
        reactivity_path="profiles.h5",
        structures_dir="structures/",
    ),
    train_config=TrainConfig(
        learning_rate=1e-3,
        weight_decay=0.01,          # Use AdamW
        loss_type="correlation",    # "mae", "mse", or "correlation"
        warmup_epochs=1.0,          # Cosine schedule with warmup
        min_lr_ratio=0.1,
        accumulation_steps=4,       # Gradient accumulation
        max_epochs=100,
    ),
)
```

### Batch Scoring

Score folders of predicted structures (for synthetic data pipelines):

```python
from sgnm import (
    SGNM,
    StructureScorer,
    BatchScorer,
    BatchScoringConfig,
    FilterConfig,
)

# Setup
model = SGNM.load("weights.pth")
scorer = StructureScorer(model)

# Configure batch scoring
batch_scorer = BatchScorer(
    scorer,
    config=BatchScoringConfig(
        input_dir="./predictions",
        output_dir="./filtered",
        profile_path="profiles.h5",
    ),
    filter_config=FilterConfig(
        threshold=0.3,
        mode="below",  # Keep structures with MAE < 0.3
    ),
)

# Score and filter
results = batch_scorer.filter_and_copy()
print(results.summary())
# {'total': 1000, 'passed': 342, 'failed': 658, 'mean_score': 0.45, ...}
```

### New Scoring API

More flexible scoring with `StructureScorer`:

```python
from sgnm import SGNM, StructureScorer, ScoringConfig

model = SGNM.load("weights.pth")
scorer = StructureScorer(
    model,
    config=ScoringConfig(
        metric="mae",  # or "mse", "correlation"
        blank_start=5,
        blank_end=5,
    ),
)

result = scorer.score_polymer(target_profile, polymer)
print(f"Score: {result.score_value:.4f}")
```

### Structure Relaxation (New API)

```python
from sgnm import StructureScorer, StructureRelaxer, RelaxConfig

relaxer = StructureRelaxer(
    scorer,
    config=RelaxConfig(steps=100, lr=1e-3, alpha=1e-3),
)

result = relaxer.relax(target_profile, polymer)
result.save("relaxed.pdb")
print(f"Final score: {result.final_score:.4f}")
```

## Module Structure

```
sgnm/
├── models.py        # SGNM and BaseSGNM models
├── equivariant.py   # EquivariantReactivityModel (flash-eq)
├── gnm.py           # Geometry utilities (frames, orientation scoring)
├── nn.py            # Neural network layers (RBF, DenseNetwork)
├── losses.py        # MAE, MSE, correlation loss functions
├── schedulers.py    # Cosine schedule with warmup
├── scoring.py       # Scoring, batch scoring, relaxation
├── training.py      # Training infrastructure
├── data.py          # Dataset utilities
└── config.py        # Configuration dataclasses
```

## Version History

- **v2.0.0** - Refactored GNM math to use ciffy; added EquivariantReactivityModel; cosine schedule, correlation loss, AdamW, gradient accumulation
- **v1.1.0** - Training infrastructure, batch scoring, structure relaxation
- **v1.0.0** - Initial release

## Contact

Email: `hmblair@stanford.edu`
