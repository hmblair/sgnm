"""Evaluate CASP ranking with multiple model checkpoints.

Usage:
    python scripts/evaluate_ranking_multi.py \
        /path/to/predictions/ \
        --profile /path/to/profiles.h5 \
        --scores /path/to/rna-scores.csv \
        --target R1149 \
        --checkpoints gnm-old:checkpoints/gnm-old/best.pth \
                      gnm-new:checkpoints/gnm/best.pth \
                      equivariant-old:checkpoints/equivariant-old/best.pth \
                      equivariant-new:checkpoints/equivariant/best.pth
"""
import argparse
import os
from pathlib import Path

import ciffy
import h5py
import numpy as np
import pandas as pd
import torch

from sgnm.config import ScoringConfig
from sgnm.scoring import pearsonr_np as pearsonr
from sgnm.scoring import rank


def load_model(label, path):
    if "equivariant" in label:
        from sgnm.equivariant import EquivariantReactivityModel
        return EquivariantReactivityModel.load(path)
    else:
        from sgnm.models import SGNM
        return SGNM.load(path)


def load_reactivity(profile_path, low=5, high=129):
    """Load SL5 reactivity from profiles.h5."""
    with h5py.File(profile_path, "r") as f:
        reactivity = f["reactivity"][0, low:high]
        reactivity[reactivity < 0.0] = 0.0
    r = torch.from_numpy(reactivity).float()
    # Normalize to [0, 1]
    r = (r - r.min()) / (r.max() - r.min()).clamp(min=1e-8)
    return r


def parse_casp_scores(csv_path, metric, target=None):
    with open(csv_path) as f:
        header = f.readline()
    if header.startswith("#") or "Model" in header:
        df = pd.read_csv(csv_path, sep=r"\s+", comment="#")
        return {f"{row['Model']}.cif": float(row[metric]) for _, row in df.iterrows() if metric in row}
    else:
        df = pd.read_csv(csv_path)
        if target:
            df = df[df["target"] == target]
        scores = {}
        for _, row in df.iterrows():
            filename = f"{row['target']}TS{int(row['gr_code']):03d}_{int(row['model'])}.cif"
            if metric in row:
                scores[filename] = float(row[metric])
        return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("structures", help="Directory of decoy .cif files")
    parser.add_argument("--profile", required=True, help="HDF5 reactivity file")
    parser.add_argument("--scores", required=True, help="CASP scores CSV")
    parser.add_argument("--metric", default="tm_score")
    parser.add_argument("--target", default=None)
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="label:path pairs")
    parser.add_argument("--channels", type=int, nargs="*", default=None,
                        help="Output channels to score (e.g. 0 for SHAPE only)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Directory for scatter plots (one per model)")
    args = parser.parse_args()

    reactivity = load_reactivity(args.profile)
    structural_scores = parse_casp_scores(args.scores, args.metric, args.target)
    scoring_config = ScoringConfig(metric="correlation", channels=args.channels)

    print(f"Decoys: {len(list(Path(args.structures).glob('*.cif')))}")
    print(f"Structural scores: {len(structural_scores)}")
    print(f"Reactivity length: {len(reactivity)}")
    if args.channels is not None:
        print(f"Scoring channels: {args.channels}")
    print()

    for spec in args.checkpoints:
        label, path = spec.split(":", 1)
        model = load_model(label, path)
        model.to(args.device)

        result = rank(model, args.structures, reactivity, config=scoring_config)
        sgnm_scores = {e.file: e.score for e in result.entries}

        if not sgnm_scores:
            print(f"[{label}] rank() returned 0 entries (all structures failed)")
            continue

        matched_sgnm, matched_metric = [], []
        for filename in sgnm_scores:
            if filename in structural_scores:
                matched_sgnm.append(sgnm_scores[filename])
                matched_metric.append(structural_scores[filename])

        if not matched_sgnm:
            print(f"[{label}] No filename overlap. "
                  f"rank files: {list(sgnm_scores.keys())[:3]}, "
                  f"score files: {list(structural_scores.keys())[:3]}")
            continue

        sgnm_arr = np.array(matched_sgnm)
        metric_arr = np.array(matched_metric)
        corr, pval = pearsonr(sgnm_arr, metric_arr)

        best_idx = np.argmax(sgnm_arr)
        sgnm_rank = int(np.sum(metric_arr >= metric_arr[best_idx]))

        print(f"[{label}] corr={corr:+.3f} (p={pval:.2e}) | "
              f"best rank={sgnm_rank}/{len(metric_arr)} | "
              f"matched={len(matched_sgnm)}")

        if args.output_dir:
            import hmbp
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            out_path = Path(args.output_dir) / f"ranking_{label}.png"
            hmbp.quick.scatter(
                sgnm_arr, metric_arr,
                title=f"{label} (r={corr:.2f}, rank={sgnm_rank}/{len(metric_arr)})",
                xlabel="SGNM Score",
                ylabel=args.metric.replace("_", " ").title(),
                path=str(out_path),
            )
            print(f"  Plot: {out_path}")


if __name__ == "__main__":
    main()
