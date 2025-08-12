## Overview

`sgnm` (SHAPE Gaussian Network Model) is a `torch` module for predicting SHAPE profiles of RNA tertiary structures in a fast and differentiable manner.

## Installation

Clone the repo, install the requirements, and then install the module.
```
git clone https://github.com/hmblair/sgnm
cd sgnm
pip3 install -r requirements.txt
pip3 install .
```
Then, download the pre-trained weights to a suitable location.
```
curl -L "https://www.dropbox.com/scl/fi/5f808uvbfaxllnxov8cr5/weights.pth?rlkey=t8utsyfgplmfip1jnnggrd3y3&st=jxwukwj7&dl=0" --output weights.pth
```

# Usage

## Single-Molecule

To get the SHAPE profile of a single molecule `mol.cif`, run
```
python3 -m sgnm --weights weights.pth mol.cif
```
If a path to the weights is not passed, then a non-parametric model will be used instead. To save the profile to an HDF5 file, you may pass `--out profile.h5` as an argument.

## Large-Scale

To integrate the model into an existing `torch` pipeline, import and init the `SGNM` module. The same remarks about non-parametric models apply as above.
```
from sgnm import SGNM
module = SGNM.load(args.weights)
# OR
module = SGNM.load()
```
Predicting the SHAPE profile requires coordinates as inputs. Optionally (and suggested if using the parametric model), you may also provide local frames as well.
```
profile = module(coords, frames)
# OR
profile = module(coords)
```
The tensor `coords` must be of shape `(n, 3)`, and `frames` of shape `(n, 3, 3)`. The weights were trained with the latter being the frame formed by the C2-C4-C6 atoms, and the former the midpoint of these three, so likely it will work best if you use these as inputs.

## Scoring

`sgnm` provides a scoring module for directly computing the MAE between a SHAPE profile and an input structure.
```
import sgnm
objective = sgnm.score(args.weights)
# OR
objective = sgnm.score()
```
Scoring has the same syntax as profile prediction.
```
mae = objective(coords, frames)
# OR
mae = objective(coords)
```

# Help

See `examples/score1.py` for a MWE of scoring a structure given coordinates and a SHAPE profile. Also see `examples/score2.py` for the same problem using the `ciffy` module to load an example structure.

Email me at `hmblair@stanford.edu` or Slack me `@Hamish` in the Stanford workspace if things go wrong.
