"""Score and rank decoy structures against an experimental reactivity profile.

Usage:
    python scripts/rank.py structures/ \\
        --profile profile.h5 --fasta ref.fasta --id 8UYS_A \\
        --weights checkpoints/best.pth \\
        -o scores.csv
"""
import argparse
import sys
from pathlib import Path

import ciffy
import torch

from sgnm.data import load_reactivity_index
from sgnm.scoring import rank


def load_model(model_type: str, weights: str | None = None):
    if model_type == "gnm":
        from sgnm.models import SGNM
        return SGNM.load(weights)
    elif model_type == "equivariant":
        from sgnm.equivariant import EquivariantReactivityModel
        return EquivariantReactivityModel.load(weights)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(description="Rank decoy structures by reactivity score")
    parser.add_argument("structures", help="Directory of decoy .cif files")
    parser.add_argument("--model", default="gnm", choices=["gnm", "equivariant"])
    parser.add_argument("--weights", help="Path to model checkpoint", default=None)
    parser.add_argument("--profile", required=True, help="HDF5 reactivity file")
    parser.add_argument("--fasta", required=True, help="FASTA with probed sequences")
    parser.add_argument("--id", dest="sample_id", required=True, help="Sample ID in profile")
    parser.add_argument("--format", default="v2", choices=["v1", "v2"])
    parser.add_argument("--output", "-o", help="Output CSV path")
    args = parser.parse_args()

    model = load_model(args.model, args.weights)

    # Align reactivity to structures via sequence matching
    index = load_reactivity_index(args.profile, args.fasta, args.format)
    cif_files = sorted(Path(args.structures).glob("*.cif"))
    reactivity = None
    for path in cif_files:
        try:
            match = index.match(ciffy.load(str(path), backend="torch").strip())
            if match is not None:
                reactivity = match.reactivity
                break
        except Exception:
            continue

    if reactivity is None:
        print("Error: could not align reactivity to any structure", file=sys.stderr)
        sys.exit(1)

    result = rank(model, args.structures, reactivity)

    if args.output:
        result.to_csv(args.output)
        print(f"Wrote {len(result.entries)} scores to {args.output}")
    else:
        print("rank,file,score")
        for e in result.entries:
            print(f"{e.rank},{e.file},{e.score:.4f}")

    print(f"\nBest:  {result.best.file} (score={result.best.score:.4f})", file=sys.stderr)
    print(f"Worst: {result.worst.file} (score={result.worst.score:.4f})", file=sys.stderr)


if __name__ == "__main__":
    main()
