"""Evaluate SGNM ranking against a structural quality metric.

Usage:
    python scripts/evaluate_ranking.py \\
        structures/ \\
        --profile profile.h5 --fasta ref.fasta --id 8UYS_A \\
        --scores casp15/rna-scores.csv --metric tm_score --target R1149 \\
        -o figures/ranking.png
"""
import argparse
import sys
from pathlib import Path

import ciffy
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

from sgnm.data import load_reactivity_index
from sgnm.scoring import rank
from rank import load_model


def parse_casp_scores(csv_path: str, metric: str, target: str | None = None) -> dict[str, float]:
    """Parse CASP scores CSV into {filename: metric_value} dict."""
    with open(csv_path) as f:
        header = f.readline()

    if header.startswith("#") or "Model" in header:
        # CASP16 format: whitespace-separated
        df = pd.read_csv(csv_path, sep=r"\s+", comment="#")
        return {f"{row['Model']}.cif": float(row[metric]) for _, row in df.iterrows() if metric in row}
    else:
        # CASP15 format: comma-separated
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
    parser = argparse.ArgumentParser(description="Evaluate SGNM ranking vs structural metric")
    parser.add_argument("structures", help="Directory of decoy .cif files")
    parser.add_argument("--model", default="gnm", choices=["gnm", "equivariant"])
    parser.add_argument("--weights", default=None)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--id", dest="sample_id", required=True)
    parser.add_argument("--format", default="v2", choices=["v1", "v2"])
    parser.add_argument("--scores", required=True, help="CASP scores CSV")
    parser.add_argument("--metric", default="tm_score")
    parser.add_argument("--target", default=None, help="CASP target ID (CASP15 format)")
    parser.add_argument("--output", "-o", default="figures/ranking.png")
    args = parser.parse_args()

    model = load_model(args.model, args.weights)

    # Align reactivity
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

    # Rank with SGNM
    result = rank(model, args.structures, reactivity)
    sgnm_scores = {e.file: e.score for e in result.entries}

    # Load structural metric
    structural_scores = parse_casp_scores(args.scores, args.metric, args.target)

    # Match
    matched_sgnm, matched_metric, matched_names = [], [], []
    for filename in sgnm_scores:
        if filename in structural_scores:
            matched_sgnm.append(sgnm_scores[filename])
            matched_metric.append(structural_scores[filename])
            matched_names.append(filename)

    if not matched_sgnm:
        print("Error: no matching structures between SGNM scores and CSV", file=sys.stderr)
        sys.exit(1)

    sgnm_arr = np.array(matched_sgnm)
    metric_arr = np.array(matched_metric)
    corr, pval = pearsonr(sgnm_arr, metric_arr)

    best_idx = np.argmax(sgnm_arr)
    best_metric_idx = np.argmax(metric_arr)
    sgnm_rank = int(np.sum(metric_arr >= metric_arr[best_idx]))

    print(f"Matched: {len(matched_sgnm)} structures")
    print(f"Pearson (SGNM vs {args.metric}): {corr:.3f} (p={pval:.2e})")
    print(f"SGNM best: {matched_names[best_idx]} ({args.metric}={metric_arr[best_idx]:.3f}, rank {sgnm_rank}/{len(metric_arr)})")
    print(f"True best:  {matched_names[best_metric_idx]} ({args.metric}={metric_arr[best_metric_idx]:.3f})")

    # Plot
    import hmbp
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    hmbp.quick.scatter(
        sgnm_arr, metric_arr,
        xlabel="SGNM Score",
        ylabel=args.metric.replace("_", " ").title(),
        title=f"SGNM vs {args.metric} (r={corr:.2f}, n={len(matched_sgnm)})",
        path=args.output,
    )
    print(f"Plot: {args.output}")


if __name__ == "__main__":
    main()
