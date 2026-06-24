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
        --weights checkpoints/gnm/best.pth \
        --noise 0 0.5 1 2 5 10 20 --repeats 5 \
        -o figures/ranking_noise_sweep.png
"""
import argparse
from pathlib import Path

import ciffy
import numpy as np
import torch

import sgnm
from sgnm.config import ScoringConfig
from sgnm.data import parse_casp_scores
from sgnm.scoring import StructureScorer, pearsonr_np


def rank_with_noised_reference(scorer, decoy_dir, gt_poly, noise_std, repeats=1):
    """Rank decoys against a reference predicted from a noised ground truth.

    Returns a list (one dict per repeat) of {filename: score}.
    """
    cif_files = sorted(Path(decoy_dir).glob("*.cif"))
    runs = []
    for _ in range(repeats):
        gt = gt_poly.copy()
        if noise_std > 0:
            gt.coordinates = gt.coordinates + torch.randn_like(gt.coordinates) * noise_std
        ref = scorer.model.ciffy(gt)  # predicted reference profile (N, C)

        scores = {}
        for path in cif_files:
            try:
                poly = ciffy.load(str(path), backend="torch")
                scores[path.name] = scorer.score(ref, poly)
            except Exception:
                continue
        runs.append(scores)
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoys", required=True, help="Directory of decoy .cif files")
    parser.add_argument("--ground-truth", required=True, help="Ground truth structure .cif")
    parser.add_argument("--scores", required=True, help="CASP scores CSV")
    parser.add_argument("--target", default=None, help="CASP target ID")
    parser.add_argument("--metric", default="tm_score")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--noise", nargs="+", type=float, default=[0, 0.5, 1, 2, 5, 10, 20])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", "-o", default="figures/ranking_noise_sweep.png")
    args = parser.parse_args()

    model = sgnm.load(args.weights).to(args.device)
    scorer = StructureScorer(model, ScoringConfig(metric="correlation"))
    gt_poly = ciffy.load(args.ground_truth, backend="torch").strip()
    structural_scores = parse_casp_scores(args.scores, args.metric, args.target)

    print(f"{'Noise (A)':>10} {'Rank Corr':>10} {'±':>6} {'Best Pick Rank':>15} {'±':>6}")
    print("-" * 55)

    results = []
    for noise in args.noise:
        runs = rank_with_noised_reference(scorer, args.decoys, gt_poly, noise, args.repeats)
        corrs, best_ranks = [], []
        for scores in runs:
            matched = [(scores[fn], structural_scores[fn]) for fn in scores if fn in structural_scores]
            if len(matched) < 5:
                continue
            sgnm_arr = np.array([m[0] for m in matched])
            metric_arr = np.array([m[1] for m in matched])
            corrs.append(pearsonr_np(sgnm_arr, metric_arr)[0])
            best_idx = np.argmax(sgnm_arr)
            best_ranks.append(int(np.sum(metric_arr >= metric_arr[best_idx])))

        if corrs:
            results.append((noise, np.mean(corrs), np.std(corrs)))
            print(f"{noise:10.1f} {np.mean(corrs):10.4f} {np.std(corrs):6.4f} "
                  f"{np.mean(best_ranks):15.1f} {np.std(best_ranks):6.1f}")
        else:
            results.append((noise, 0.0, 0.0))
            print(f"{noise:10.1f} {'FAILED':>10}")

    try:
        import hmbp
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        hmbp.quick.line(
            [r[1] for r in results], [r[0] for r in results],
            xlabel="Ground Truth Noise (Angstroms)",
            ylabel="Ranking Correlation (Pearson)",
            title="Ranking Performance vs GT Noise",
            path=args.output,
        )
        print(f"\nPlot saved to {args.output}")
    except ImportError:
        print("\nhmbp not available, skipping plot")


if __name__ == "__main__":
    main()
