"""Evaluate how ranking performance degrades with structural noise.

For each noise level, predicts reactivity from a noised copy of the ground
truth structure and uses that as the reference profile for ranking decoys.
This simulates the scenario where the "true" structure used for scoring
has varying levels of error.

Usage:
    python scripts/ranking_noise_sweep.py \
        --decoys data/casp/casp15/sl5/predictions/ \
        --ground-truth data/casp/casp15/sl5/structure/8UYS.cif \
        --scores data/casp/casp15/rna-scores.csv \
        --target R1149 --metric tm_score \
        --weights checkpoints/gnm/best.pth --model gnm \
        --noise 0 0.5 1 2 5 10 20 \
        --repeats 5 \
        -o figures/ranking_noise_sweep_gnm.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import ciffy
from sgnm.losses import pearsonr_np as pearsonr
from sgnm.scoring import _normalize, _correlation


def parse_casp_scores(csv_path, metric, target=None):
    """Parse CASP scores CSV into {filename: metric_value} dict."""
    import pandas as pd

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


def rank_with_noised_reference(model, decoy_dir, gt_poly, noise_std, repeats=1):
    """Rank decoys using reactivity predicted from a noised ground truth.

    Returns list of (sgnm_scores_dict, one per repeat).
    """
    model.eval()
    cif_files = sorted(Path(decoy_dir).glob("*.cif"))
    all_runs = []

    for _ in range(repeats):
        # Predict reference profile from noised ground truth
        gt = gt_poly.copy()
        if noise_std > 0:
            gt.coordinates = gt.coordinates + torch.randn_like(gt.coordinates) * noise_std

        with torch.no_grad():
            ref_profile = model.ciffy(gt)  # (N, C)

        # Score each decoy against this reference
        scores = {}
        with torch.no_grad():
            for path in cif_files:
                try:
                    poly = ciffy.load(str(path), backend="torch")
                    pred = model.ciffy(poly)

                    # Correlation between decoy prediction and noised-GT prediction
                    ref = _normalize(ref_profile)
                    p = _normalize(pred)

                    if ref.dim() > 1:
                        corr = _correlation(p, ref)
                        score = (corr ** 2 * corr.sign()).sum().item()
                    else:
                        score = _correlation(p, ref).item()

                    scores[path.name] = score
                except Exception:
                    continue

        all_runs.append(scores)

    return all_runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoys", required=True, help="Directory of decoy .cif files")
    parser.add_argument("--ground-truth", required=True, help="Ground truth structure .cif")
    parser.add_argument("--scores", required=True, help="CASP scores CSV")
    parser.add_argument("--target", default=None, help="CASP target ID (for CASP15 format)")
    parser.add_argument("--metric", default="tm_score")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--model", default="gnm", choices=["gnm", "equivariant"])
    parser.add_argument("--noise", nargs="+", type=float, default=[0, 0.5, 1, 2, 5, 10, 20])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", "-o", default="figures/ranking_noise_sweep.png")
    args = parser.parse_args()

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model == "gnm":
        from sgnm.models import SGNM
        model = SGNM.load(args.weights).to(device)
    else:
        from sgnm.equivariant import EquivariantReactivityModel
        model = EquivariantReactivityModel.load(args.weights).to(device)

    # Load ground truth structure
    gt_poly = ciffy.load(args.ground_truth, backend="torch").strip()

    # Load structural quality scores
    structural_scores = parse_casp_scores(args.scores, args.metric, args.target)

    print(f"{'Noise (A)':>10} {'Rank Corr':>10} {'±':>6} {'Best Pick Rank':>15} {'±':>6}")
    print("-" * 55)

    results = []
    for noise in args.noise:
        all_runs = rank_with_noised_reference(
            model, args.decoys, gt_poly, noise, args.repeats
        )

        corrs = []
        best_ranks = []
        for sgnm_scores in all_runs:
            # Match SGNM scores with structural scores
            matched_sgnm, matched_metric = [], []
            for fn in sgnm_scores:
                if fn in structural_scores:
                    matched_sgnm.append(sgnm_scores[fn])
                    matched_metric.append(structural_scores[fn])

            if len(matched_sgnm) < 5:
                continue

            sgnm_arr = np.array(matched_sgnm)
            metric_arr = np.array(matched_metric)

            corr, _ = pearsonr(sgnm_arr, metric_arr)
            corrs.append(corr)

            # Rank of SGNM's best pick by structural metric
            best_idx = np.argmax(sgnm_arr)
            best_rank = int(np.sum(metric_arr >= metric_arr[best_idx]))
            best_ranks.append(best_rank)

        if corrs:
            results.append((
                noise,
                np.mean(corrs), np.std(corrs),
                np.mean(best_ranks), np.std(best_ranks),
                len(matched_sgnm),
            ))
            print(f"{noise:10.1f} {np.mean(corrs):10.4f} {np.std(corrs):6.4f} "
                  f"{np.mean(best_ranks):15.1f} {np.std(best_ranks):6.1f}")
        else:
            results.append((noise, 0, 0, 0, 0, 0))
            print(f"{noise:10.1f} {'FAILED':>10}")

    # Plot
    try:
        import hmbp
        noise_vals = [r[0] for r in results]
        corr_means = [r[1] for r in results]

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        hmbp.quick.line(
            corr_means, noise_vals,
            xlabel="Ground Truth Noise (Angstroms)",
            ylabel="Ranking Correlation (Pearson)",
            title=f"Ranking Performance vs GT Noise ({args.model.upper()})",
            path=args.output,
        )
        print(f"\nPlot saved to {args.output}")
    except ImportError:
        print("\nhmbp not available, skipping plot")


if __name__ == "__main__":
    main()
