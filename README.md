# SGNM

SHAPE Gaussian Network Model for predicting SHAPE profiles of RNA tertiary structures.

## Overview

`sgnm` is a PyTorch module for predicting SHAPE profiles of RNA tertiary structures in a fast and differentiable manner. It uses a Gaussian Network Model approach with learnable radial basis function embeddings.

## Installation

Clone the repo, install the requirements, and then install the module.
```bash
git clone https://github.com/hmblair/sgnm
cd sgnm
pip install -r requirements.txt
pip install .
```

Download the pre-trained weights:
```bash
curl -L "https://www.dropbox.com/scl/fi/5f808uvbfaxllnxov8cr5/weights.pth?rlkey=t8utsyfgplmfip1jnnggrd3y3&st=jxwukwj7&dl=0" --output weights.pth
```

## Quick Start

### Command Line

Predict SHAPE profile for a single molecule:
```bash
python -m sgnm --weights weights.pth mol.cif
```

Save output to HDF5:
```bash
python -m sgnm --weights weights.pth --out profile.h5 mol.cif
```

### Python API

```python
from sgnm import SGNM

# Load pre-trained model
model = SGNM.load("weights.pth")

# Or use non-parametric model
model = SGNM.load()

# Predict from coordinates and frames
profile = model(coords, frames)
```

## Usage

### Profile Prediction

```python
from sgnm import SGNM

model = SGNM.load("weights.pth")

# From raw tensors
# coords: shape (..., n, 3) - residue center coordinates
# frames: shape (..., n, 3, 3) - local coordinate frames
profile = model(coords, frames)

# From ciffy Polymer
profile = model.ciffy(polymer)
```

The frames are formed by the `C2-C4-C6` atom triplet, and coordinates are the center of these triplets.

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

## Advanced Usage (v1.1.0+)

### Configuration

Use dataclasses for type-safe configuration:

```python
from sgnm import ModelConfig, SGNM

# Configure model architecture
config = ModelConfig(dim=64, out_dim=1, layers=2)
model = SGNM(config)
```

### Training

Train models with the new `Trainer` class:

```python
from sgnm import (
    train_sgnm,
    ModelConfig,
    DataConfig,
    TrainConfig,
)

results = train_sgnm(
    model_config=ModelConfig(dim=32, layers=2),
    data_config=DataConfig(
        reactivity_path="profiles.h5",
        structures_dir="structures/",
    ),
    train_config=TrainConfig(
        learning_rate=1e-3,
        max_epochs=100,
        checkpoint_dir="./checkpoints",
    ),
)

print(f"Best validation loss: {results.best_val_loss:.4f}")
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
├── config.py      # Configuration dataclasses
├── nn.py          # Neural network layers (RBF, DenseNetwork)
├── models.py      # SGNM and BaseSGNM models
├── scoring.py     # Scoring, batch scoring, relaxation
├── training.py    # Training infrastructure
├── data.py        # Dataset utilities
└── gnm.py         # Core GNM mathematical functions
```

## Examples

See the `examples/` directory:
- `score1.py` - Scoring with raw coordinates
- `score2.py` - Scoring with ciffy Polymer
- `relax.py` - Structure relaxation

## Version History

- **v1.1.0** - Refactored for separation of concerns, added training infrastructure, batch scoring
- **v1.0.0** - Initial stable release

## Contact

Email: `hmblair@stanford.edu`
Slack: `@Hamish` (Stanford workspace)
