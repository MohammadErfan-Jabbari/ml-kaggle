#!/usr/bin/env python3
"""
Phase 3: Global vs Per-Species PLS Regression.

This script compares PLSRegression(n=20) trained globally vs per-species
to determine if per-species training avoids "P. aeruginosa pattern leakage"
and improves K.pneumoniae predictions.

Usage:
    uv run python experiments/run_phase3.py

Expected runtime: ~5 minutes
Output: JSON file in outputs/experiments/phase3_combined_results.json
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List
import numpy as np
from scipy.stats import ttest_rel

# Ensure proper imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import controlled_experiment as ce
from controlled_experiment import (
    ControlledExperiment,
    OUTPUT_DIR,
    load_data,
    SPECIES_NAMES,
    N_FOLDS,
)
from features.reducers import PerSpeciesPLS

# Global baseline from Phase 2
GLOBAL_BASELINE = {
    "name": "2B5_pls_n20",
    "kpn_auc": 0.7431,
    "source_file": "phase2_2B5_pls_n20.json"
}


class AugmentedSpeciesExperiment(ControlledExperiment):
    """
    Controlled experiment with species-augmented features.

    Overrides the run() method to augment X with species_id as the last
    feature before processing. This allows PerSpeciesPLS to extract
    species information during fit/transform.
    """

    def run(self):
        """
        Override run to augment X with species_id before processing.

        Returns
        -------
        ExperimentResult
            Experiment results with per-species PLS transformation.
        """
        # Load original data (this populates _DATA_CACHE)
        X, X_test, y, species, species_test = load_data()

        # Augment with species as last feature (feature 6001)
        X_aug = np.hstack([X, species.reshape(-1, 1).astype(np.float32)])
        X_test_aug = np.hstack([X_test, species_test.reshape(-1, 1).astype(np.float32)])

        # Monkey-patch global cache for this run
        # This is the cleanest way to inject augmented data without
        # modifying ControlledExperiment
        original_cache = ce._DATA_CACHE
        ce._DATA_CACHE = (X_aug, X_test_aug, y, species, species_test)

        try:
            # Run parent experiment with augmented data
            result = super().run()
        finally:
            # Reset cache to original state
            ce._DATA_CACHE = original_cache

        return result


def load_global_baseline() -> Dict[str, Any]:
    """Load global PLS(n=20) result from Phase 2."""
    baseline_path = OUTPUT_DIR / GLOBAL_BASELINE["source_file"]

    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Phase 2 baseline not found: {baseline_path}\n"
            f"Please run Phase 2 first: uv run python experiments/run_phase2.py"
        )

    with open(baseline_path) as f:
        return json.load(f)


def extract_fold_aucs(result: Dict[str, Any], species: str = "K.pneumoniae") -> List[float]:
    """Extract fold-level AUCs for a species from result dict."""
    if 'per_species' in result and species in result['per_species']:
        fold_data = result['per_species'][species]
        if 'folds' in fold_data:
            return [f for f in fold_data['folds'] if not np.isnan(f)]
    return []


def compute_statistical_comparison(
    global_folds: List[float],
    per_species_folds: List[float]
) -> Dict[str, Any]:
    """Compute paired t-test comparing per-species vs global."""
    if len(global_folds) != len(per_species_folds):
        return {
            't_statistic': None,
            'p_value': None,
            'significant': False,
            'note': f'Fold count mismatch: global={len(global_folds)}, per_species={len(per_species_folds)}'
        }

    if len(global_folds) < 2:
        return {
            't_statistic': None,
            'p_value': None,
            'significant': False,
            'note': 'Insufficient folds for statistical testing'
        }

    stat, p_value = ttest_rel(per_species_folds, global_folds)
    return {
        't_statistic': float(stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'note': None
    }


def print_phase3_summary(
    global_result: Dict[str, Any],
    per_species_result: Any,
    comparison: Dict[str, Any]
):
    """Print Phase 3 summary and recommendation."""
    global_kpn = global_result['K.pneumoniae_auc']
    global_kpn_std = global_result.get('K.pneumoniae_auc_std', 0)
    per_species_kpn = per_species_result.per_species['K.pneumoniae']['mean']
    per_species_kpn_std = per_species_result.per_species['K.pneumoniae'].get('std', 0)

    delta = per_species_kpn - global_kpn
    delta_percent = (delta / global_kpn) * 100

    print("\n" + "=" * 70)
    print("PHASE 3 SUMMARY: GLOBAL VS PER-SPECIES PLS(n=20)")
    print("=" * 70)

    print("\nK.pneumoniae AUC Comparison:")
    print(f"{'Method':<20} {'AUC':>10} {'Std':>10} {'Delta':>12}")
    print("-" * 70)
    print(f"{'Global PLS(n=20)':<20} {global_kpn:>10.4f} {global_kpn_std:>10.4f} {'':>12}")
    print(f"{'Per-species PLS(n=20)':<20} {per_species_kpn:>10.4f} {per_species_kpn_std:>10.4f} {delta:>+11.4f}")
    print("-" * 70)
    print(f"{'Delta':<20} {delta_percent:>+10.2f}%")

    # Statistical significance
    p_value = comparison.get('p_value')
    if p_value is not None:
        print(f"\nStatistical Test (paired t-test):")
        print(f"  t-statistic: {comparison['t_statistic']:.4f}")
        print(f"  p-value: {p_value:.4f}")
        if comparison['significant']:
            print(f"  ✓ Statistically significant (α=0.05)")
        else:
            print(f"  ✗ Not statistically significant")
    else:
        print(f"\nStatistical Test: {comparison.get('note', 'N/A')}")

    # Per-species model details
    print("\nPer-Species Models:")
    print(f"{'Species':<15} {'Trained':>10} {'N Samples':>12}")
    print("-" * 70)
    for species_id, species_name in SPECIES_NAMES.items():
        stats = per_species_result.config.get('species_stats', {}).get(species_id, {})
        trained = stats.get('trained', False)
        n_samples = stats.get('n_fully_labeled', 0)
        status = "✓" if trained else "✗"
        print(f"  {species_name:<15} {status:>10} {n_samples:>12}")

    # Recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION FOR PHASE 4")
    print("=" * 70)

    if delta > 0.01 and comparison.get('significant', False):
        recommendation = (
            f"✓ Per-species PLS significantly better by +{delta_percent:.2f}%\n"
            f"  Include per-species variants in Phase 4 combinations."
        )
    elif delta > 0.005:
        recommendation = (
            f"~ Per-species PLS slightly better (+{delta_percent:.2f}%) but not significant\n"
            f"  Consider per-species variants in Phase 4 if convenient."
        )
    elif delta > -0.005:
        recommendation = (
            f"≈ Per-species PLS similar to global (Δ={delta_percent:+.2f}%)\n"
            f"  Either approach acceptable. Default to global for simplicity."
        )
    else:
        recommendation = (
            f"✗ Per-species PLS worse than global by {delta_percent:.2f}%\n"
            f"  Use global PLS for Phase 4."
        )

    print(recommendation)
    print("=" * 70)


def save_combined_results(
    global_result: Dict[str, Any],
    per_species_result: Any,
    comparison: Dict[str, Any],
    recommendation: str
):
    """Save combined Phase 3 results to JSON."""
    global_kpn = global_result['K.pneumoniae_auc']
    per_species_kpn = per_species_result.per_species['K.pneumoniae']['mean']
    delta = per_species_kpn - global_kpn
    delta_percent = (delta / global_kpn) * 100

    output = {
        "phase": 3,
        "timestamp": datetime.now().isoformat(),
        "global_baseline": {
            "name": global_result['name'],
            "kpn_auc": global_kpn,
            "kpn_auc_std": global_result.get('K.pneumoniae_auc_std', 0),
            "folds": global_result['per_species']['K.pneumoniae']['folds'],
            "source": GLOBAL_BASELINE["source_file"]
        },
        "per_species_result": per_species_result.to_dict(),
        "comparison": {
            "delta_kpn_auc": float(delta),
            "delta_percent": float(delta_percent),
            "p_value": comparison.get('p_value'),
            "significant": comparison.get('significant', False),
            "t_statistic": comparison.get('t_statistic')
        },
        "recommendation": recommendation
    }

    output_path = OUTPUT_DIR / "phase3_combined_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nCombined results saved to: {output_path}")


def get_recommendation_text(
    delta: float,
    delta_percent: float,
    significant: bool
) -> str:
    """Generate recommendation text based on results."""
    if delta > 0.01 and significant:
        return f"Per-species wins by +{delta_percent:.2f}% [p<0.05]"
    elif delta > 0.005:
        return f"Per-species slightly better (+{delta_percent:.2f}%) but not significant"
    elif delta > -0.005:
        return f"Global and per-species equivalent (Δ={delta_percent:+.2f}%)"
    else:
        return f"Global preferred (per-species is {delta_percent:.2f}% worse)"


def main():
    """Run Phase 3 experiments."""
    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    print("=" * 70)
    print("PHASE 3: GLOBAL VS PER-SPECIES PLS REGRESSION")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Step 1: Load global baseline from Phase 2
    print("\n[Step 1] Loading global baseline from Phase 2...")
    global_result = load_global_baseline()
    print(f"  Loaded: {GLOBAL_BASELINE['source_file']}")
    print(f"  Global K.pn AUC: {global_result['K.pneumoniae_auc']:.4f}")

    # Step 2: Run per-species experiment
    print("\n[Step 2] Running per-species PLS experiment...")
    per_species_exp = AugmentedSpeciesExperiment(
        name="3A2_pls_n20_per_species",
        transformer=PerSpeciesPLS(n_components=20, min_samples=20),
        use_scaler=True
    )
    per_species_result = per_species_exp.run()
    per_species_exp.print_summary()

    # Save individual result
    individual_output_path = OUTPUT_DIR / "phase3_3A2_pls_n20_per_species.json"
    per_species_exp.save_results(individual_output_path)

    # Step 3: Statistical comparison
    print("\n[Step 3] Computing statistical comparison...")
    global_folds = extract_fold_aucs(global_result, "K.pneumoniae")
    per_species_folds = extract_fold_aucs(per_species_result.to_dict(), "K.pneumoniae")

    print(f"  Global fold AUCs: {[f'{f:.4f}' for f in global_folds]}")
    print(f"  Per-species fold AUCs: {[f'{f:.4f}' for f in per_species_folds]}")

    comparison = compute_statistical_comparison(global_folds, per_species_folds)

    # Step 4: Generate recommendation
    print("\n[Step 4] Generating recommendation...")
    global_kpn = global_result['K.pneumoniae_auc']
    per_species_kpn = per_species_result.per_species['K.pneumoniae']['mean']
    delta = per_species_kpn - global_kpn
    delta_percent = (delta / global_kpn) * 100
    recommendation = get_recommendation_text(
        delta,
        delta_percent,
        comparison.get('significant', False)
    )

    # Step 5: Save combined results
    print("\n[Step 5] Saving combined results...")
    save_combined_results(
        global_result,
        per_species_result,
        comparison,
        recommendation
    )

    # Step 6: Print summary
    print_phase3_summary(global_result, per_species_result, comparison)

    # Print timing
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\nTotal runtime: {duration}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
