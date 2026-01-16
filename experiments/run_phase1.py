#!/usr/bin/env python3
"""
Phase 1: Single Method Ablation Study.

This script runs all Phase 1 experiments to evaluate different dimensionality
reduction and feature selection methods. Each method is tested with fixed
LightGBM hyperparameters and 5-fold species-stratified CV.

Usage:
    uv run python experiments/run_phase1.py

Expected runtime: ~30-60 minutes
Output: JSON files in outputs/experiments/phase1_*.json
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Ensure proper imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sklearn.decomposition import TruncatedSVD, NMF, PCA, KernelPCA
from sklearn.cross_decomposition import PLSRegression, PLSCanonical
from sklearn.feature_selection import VarianceThreshold, f_classif
from sklearn.pipeline import Pipeline

from controlled_experiment import ControlledExperiment, OUTPUT_DIR
from features.reducers import (
    MultiTargetFeatureUnion,
    LGBImportanceSelector,
    MultiStageSelector
)

# Phase 1 Experiments
# Format: (name, transformer, use_scaler)
EXPERIMENTS = [
    # =========================================================================
    # Tier 1: Direct sklearn (no custom code needed)
    # =========================================================================

    # 1B: Modern DR Methods - TruncatedSVD (sparse-native)
    ("1B1_svd_50", TruncatedSVD(n_components=50, random_state=42), False),
    ("1B2_svd_100", TruncatedSVD(n_components=100, random_state=42), False),
    ("1B3_svd_200", TruncatedSVD(n_components=200, random_state=42), False),

    # 1B: Modern DR Methods - NMF (MALDI-proven)
    ("1B4_nmf_50", NMF(n_components=50, init='nndsvd', max_iter=500, random_state=42), False),
    ("1B5_nmf_100", NMF(n_components=100, init='nndsvd', max_iter=500, random_state=42), False),

    # 1A: Course Methods - PLS
    ("1A1_pls_7", PLSRegression(n_components=7), True),
    ("1A2_pls_3", PLSRegression(n_components=3), True),

    # 1C: Kernel Methods
    ("1C1_kpca_cosine", KernelPCA(n_components=100, kernel='cosine'), True),

    # 1D: Feature Selection - Variance Threshold
    ("1D1_variance", VarianceThreshold(threshold=0.01), False),

    # =========================================================================
    # Tier 2: Simple custom transformers
    # =========================================================================

    # 1A: Course Methods - PLSCanonical (multi-target)
    ("1A3_plscanonical_7", PLSCanonical(n_components=7), True),

    # 1D: Feature Selection - Univariate (f_classif)
    ("1D2_f_classif_500", MultiTargetFeatureUnion(f_classif, k=500), False),

    # =========================================================================
    # Tier 3: Complex transformers
    # =========================================================================

    # 1A: Course Methods - PCA + PLSCanonical Pipeline
    ("1A4_pca200_plscanonical7", Pipeline([
        ('pca', PCA(n_components=200, random_state=42)),
        ('pls', PLSCanonical(n_components=7))
    ]), True),

    # 1D: Feature Selection - LGB Importance
    ("1D3_lgb_importance_500", LGBImportanceSelector(k=500), False),

    # 1D: Feature Selection - Multi-stage
    ("1D4_multistage", MultiStageSelector(var_threshold=0.001, univariate_k=1000, lgb_k=500), False),
]


def run_all_experiments() -> List[Dict[str, Any]]:
    """Run all Phase 1 experiments and return results."""
    results = []
    n_experiments = len(EXPERIMENTS)

    print("=" * 70)
    print("PHASE 1: SINGLE METHOD ABLATION STUDY")
    print("=" * 70)
    print(f"\nRunning {n_experiments} experiments...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    for idx, (name, transformer, use_scaler) in enumerate(EXPERIMENTS, 1):
        print(f"\n[{idx}/{n_experiments}] Starting experiment: {name}")
        print("-" * 50)

        try:
            exp = ControlledExperiment(
                name=name,
                transformer=transformer,
                use_scaler=use_scaler
            )
            result = exp.run()
            exp.print_summary()

            # Save individual result
            output_path = OUTPUT_DIR / f"phase1_{name}.json"
            exp.save_results(output_path)

            results.append(result.to_dict())

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'name': name,
                'error': str(e),
                'mean_auc': 0,
                'K.pneumoniae_auc': 0
            })

    return results


def print_ranking(results: List[Dict[str, Any]]):
    """Print ranking table sorted by K.pneumoniae AUC."""
    print("\n" + "=" * 70)
    print("PHASE 1 RANKING (by K.pneumoniae AUC)")
    print("=" * 70)

    # Load baseline for comparison
    baseline_path = OUTPUT_DIR / "phase0_baseline.json"
    baseline_kpn = 0.6946
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
            baseline_kpn = baseline.get('K.pneumoniae_auc', 0.6946)

    # Sort by K.pneumoniae AUC
    ranking = sorted(
        results,
        key=lambda x: x.get('K.pneumoniae_auc', 0) or 0,
        reverse=True
    )

    print(f"\nBaseline K.pn AUC: {baseline_kpn:.4f}")
    print()
    print(f"{'Rank':<5} {'Method':<30} {'K.pn AUC':>10} {'Δ Baseline':>12} {'Mean AUC':>10}")
    print("-" * 70)

    for rank, result in enumerate(ranking, 1):
        name = result.get('name', 'unknown')
        kpn_auc = result.get('K.pneumoniae_auc', 0) or 0
        mean_auc = result.get('mean_auc', 0) or 0
        delta = kpn_auc - baseline_kpn

        delta_str = f"{delta:+.4f}" if kpn_auc > 0 else "N/A"
        kpn_str = f"{kpn_auc:.4f}" if kpn_auc > 0 else "FAILED"
        mean_str = f"{mean_auc:.4f}" if mean_auc > 0 else "N/A"

        marker = " ✓" if delta > 0.02 else ""  # Mark improvements > 2%
        print(f"{rank:<5} {name:<30} {kpn_str:>10} {delta_str:>12} {mean_str:>10}{marker}")

    # Summary
    improvements = [r for r in ranking if (r.get('K.pneumoniae_auc', 0) or 0) > baseline_kpn]
    print("\n" + "-" * 70)
    print(f"Methods that improved K.pn AUC: {len(improvements)}/{len(results)}")

    if improvements:
        best = improvements[0]
        best_delta = (best.get('K.pneumoniae_auc', 0) or 0) - baseline_kpn
        print(f"Best improvement: {best['name']} (+{best_delta:.4f})")


def save_combined_results(results: List[Dict[str, Any]]):
    """Save combined results and ranking to a single JSON file."""
    output = {
        'phase': 1,
        'timestamp': datetime.now().isoformat(),
        'n_experiments': len(results),
        'baseline_K.pneumoniae_auc': 0.6946,
        'baseline_mean_auc': 0.8998,
        'results': results,
        'ranking_by_kpn': sorted(
            [r['name'] for r in results if r.get('K.pneumoniae_auc')],
            key=lambda n: next(
                (r.get('K.pneumoniae_auc', 0) for r in results if r['name'] == n),
                0
            ),
            reverse=True
        )
    }

    output_path = OUTPUT_DIR / "phase1_combined_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nCombined results saved to: {output_path}")


def main():
    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Run all experiments
    results = run_all_experiments()

    # Print ranking
    print_ranking(results)

    # Save combined results
    save_combined_results(results)

    # Print timing
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\nTotal runtime: {duration}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
