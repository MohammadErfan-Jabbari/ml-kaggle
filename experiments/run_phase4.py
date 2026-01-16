#!/usr/bin/env python3
"""
Phase 4: Combination Experiments.

Tests two-stage pipelines combining feature selection (FS) and dimensionality
reduction (DR) methods to determine if combinations outperform the best
single method (PLS n=20, K.pn AUC=0.7431).

Usage:
    uv run python experiments/run_phase4.py

Expected runtime: ~3-4 minutes
Output: JSON files in outputs/experiments/phase4_*.json
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.stats import ttest_rel

# Ensure proper imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.cross_decomposition import PLSRegression

from controlled_experiment import (
    ControlledExperiment,
    OUTPUT_DIR,
    load_data,
    SPECIES_NAMES,
    N_FOLDS,
)
from features.reducers import LGBImportanceSelector, NMFScaler

# Phase 4 Experiments (5 total)
# Format: (name, transformer_pipeline, use_scaler)
EXPERIMENTS = [
    # =========================================================================
    # 4A: FS → DR Pipelines (Feature Selection then Dimensionality Reduction)
    # =========================================================================
    ("4A1_lgb500_pls20",
     Pipeline([
         ('fs', LGBImportanceSelector(k=500)),
         ('dr', PLSRegression(n_components=20))
     ]),
     True),  # use_scaler for PLS

    ("4A2_lgb500_nmf50",
     Pipeline([
         ('fs', LGBImportanceSelector(k=500)),
         ('dr', NMFScaler(n_components=50))
     ]),
     False),  # NMFScaler handles its own shifting

    ("4A3_lgb500_svd50",
     Pipeline([
         ('fs', LGBImportanceSelector(k=500)),
         ('dr', TruncatedSVD(n_components=50, random_state=42))
     ]),
     False),  # SVD doesn't need scaling

    # =========================================================================
    # 4B: DR → DR Pipelines (Two-stage Dimensionality Reduction)
    # =========================================================================
    ("4B1_svd200_pls20",
     Pipeline([
         ('dr1', TruncatedSVD(n_components=200, random_state=42)),
         ('dr2', PLSRegression(n_components=20))
     ]),
     True),  # use_scaler for PLS

    ("4B2_nmf100_pls20",
     Pipeline([
         ('dr1', NMFScaler(n_components=100)),
         ('dr2', PLSRegression(n_components=20))
     ]),
     True),  # use_scaler for PLS
]

# Baseline from Phase 2
BASELINE_KPN_AUC = 0.7431
BASELINE_MEAN_AUC = 0.9040
BASELINE_FILE = "phase2_2B5_pls_n20.json"


def load_baseline_folds() -> List[float]:
    """Load baseline fold AUCs from Phase 2 results."""
    baseline_path = OUTPUT_DIR / BASELINE_FILE
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Phase 2 baseline not found: {baseline_path}\n"
            f"Please run Phase 2 first: uv run python experiments/run_phase2.py"
        )

    with open(baseline_path) as f:
        baseline_data = json.load(f)

    # Extract K.pneumoniae fold AUCs
    if 'per_species' in baseline_data and 'K.pneumoniae' in baseline_data['per_species']:
        folds = baseline_data['per_species']['K.pneumoniae'].get('folds', [])
        valid_folds = [f for f in folds if not np.isnan(f)]
        if valid_folds:
            return valid_folds

    raise ValueError(f"Could not extract baseline fold AUCs from {baseline_path}")


def run_phase4_with_checkpointing() -> Tuple[List[Dict[str, Any]], List[str]]:
    """Run Phase 4 with checkpoint recovery."""
    checkpoint_file = OUTPUT_DIR / "phase4_checkpoint.json"

    # Load checkpoint if exists
    completed = set()
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            checkpoint_data = json.load(f)
            completed = set(checkpoint_data.get('completed_experiments', []))
        print(f"Resuming from checkpoint: {len(completed)} experiments already completed")
    else:
        print("No checkpoint found, starting fresh")

    results = []
    skipped = []

    for idx, (name, transformer, use_scaler) in enumerate(EXPERIMENTS, 1):
        if name in completed:
            print(f"\n[{idx}/{len(EXPERIMENTS)}] Skipping {name} (already completed)")
            # Load existing result
            result_file = OUTPUT_DIR / f"phase4_{name}.json"
            if result_file.exists():
                with open(result_file) as f:
                    results.append(json.load(f))
            continue

        print(f"\n[{idx}/{len(EXPERIMENTS)}] Starting experiment: {name}")
        print("-" * 60)

        try:
            exp = ControlledExperiment(
                name=name,
                transformer=transformer,
                use_scaler=use_scaler
            )
            result = exp.run()
            exp.print_summary()

            # Save individual result
            output_path = OUTPUT_DIR / f"phase4_{name}.json"
            exp.save_results(output_path)

            results.append(result.to_dict())

            # Update checkpoint
            completed.add(name)
            with open(checkpoint_file, 'w') as f:
                json.dump({'completed_experiments': list(completed)}, f)

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

    return results, skipped


def extract_fold_aucs(result: Dict[str, Any], species: str = "K.pneumoniae") -> List[float]:
    """Extract fold-level AUCs for a species from result dict."""
    if 'per_species' in result and species in result['per_species']:
        fold_data = result['per_species'][species]
        if 'folds' in fold_data:
            return [f for f in fold_data['folds'] if not np.isnan(f)]
    return []


def compute_statistical_tests(
    results: List[Dict[str, Any]],
    baseline_folds: List[float]
) -> Dict[str, Dict]:
    """Compute statistical tests for combinations vs baseline."""
    tests = {}

    for result in results:
        if 'error' in result:
            continue

        name = result['name']
        combo_folds = extract_fold_aucs(result, "K.pneumoniae")

        if len(combo_folds) != len(baseline_folds):
            tests[f"{name}_vs_baseline"] = {
                'p_value': None,
                'significant': False,
                'note': f'Fold count mismatch: combo={len(combo_folds)}, baseline={len(baseline_folds)}'
            }
            continue

        # Paired t-test
        stat, p_value = ttest_rel(combo_folds, baseline_folds)
        mean_diff = np.mean(combo_folds) - np.mean(baseline_folds)

        tests[f"{name}_vs_baseline"] = {
            't_statistic': float(stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'mean_difference': float(mean_diff)
        }

    return tests


def compute_effect_size(combo_folds: List[float], baseline_folds: List[float]) -> Dict[str, Any]:
    """Compute Cohen's d for effect size."""
    mean_diff = np.mean(combo_folds) - np.mean(baseline_folds)
    pooled_std = np.sqrt((np.std(combo_folds)**2 + np.std(baseline_folds)**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

    interpretation = 'small' if abs(cohens_d) < 0.2 else 'medium' if abs(cohens_d) < 0.8 else 'large'

    return {
        'cohens_d': float(cohens_d),
        'interpretation': interpretation
    }


def compute_confidence_interval_difference(
    combo_folds: List[float],
    baseline_folds: List[float],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """Compute CI for mean difference."""
    from scipy.stats import t

    diff = np.array(combo_folds) - np.array(baseline_folds)
    mean_diff = np.mean(diff)
    se_diff = np.std(diff, ddof=1) / np.sqrt(len(diff))

    ci = t.interval(1 - alpha, df=len(diff) - 1, loc=mean_diff, scale=se_diff)

    return {
        'mean_difference': float(mean_diff),
        'ci_lower': float(ci[0]),
        'ci_upper': float(ci[1]),
        'includes_zero': ci[0] <= 0 <= ci[1],
        'confidence_level': 1 - alpha
    }


def compute_all_statistics(
    results: List[Dict[str, Any]],
    baseline_folds: List[float]
) -> Dict[str, Dict]:
    """Compute comprehensive statistical analysis."""
    statistics = {}

    for result in results:
        if 'error' in result:
            continue

        name = result['name']
        combo_folds = extract_fold_aucs(result, "K.pneumoniae")

        if len(combo_folds) != len(baseline_folds):
            continue

        stats = {}
        stats['statistical_test'] = compute_statistical_tests([result], baseline_folds)[f"{name}_vs_baseline"]
        stats['effect_size'] = compute_effect_size(combo_folds, baseline_folds)
        stats['confidence_interval'] = compute_confidence_interval_difference(combo_folds, baseline_folds)

        statistics[name] = stats

    return statistics


def print_phase4_summary(
    results: List[Dict[str, Any]],
    statistics: Dict[str, Dict]
):
    """Print Phase 4 summary and recommendations."""
    print("\n" + "=" * 80)
    print("PHASE 4 SUMMARY: COMBINATION EXPERIMENTS")
    print("=" * 80)

    print(f"\nBaseline (PLS n=20): Mean AUC = {BASELINE_MEAN_AUC:.4f}")
    print(f"Results from {len([r for r in results if 'error' not in r])} combination experiments:")

    # Rank by Mean AUC (primary metric - matches leaderboard scoring)
    valid_results = [r for r in results if 'error' not in r]
    sorted_results = sorted(valid_results, key=lambda x: x.get('mean_auc', 0), reverse=True)

    print(f"\n{'Experiment':<20} {'Mean AUC':>10} {'Std':>10} {'Δ Baseline':>12} {'p-value':>10} {'Effect':>10}")
    print("-" * 80)

    for result in sorted_results:
        name = result['name']
        mean = result.get('mean_auc', 0)
        std = result.get('mean_auc_std', 0)
        delta = mean - BASELINE_MEAN_AUC

        stats = statistics.get(name, {})
        pval = stats.get('statistical_test', {}).get('p_value')
        effect = stats.get('effect_size', {}).get('interpretation', 'N/A')

        pval_str = f"{pval:.4f}" if pval is not None else "N/A"
        marker = " ★" if mean == sorted_results[0].get('mean_auc', 0) else ""

        print(f"{name:<20} {mean:>10.4f} {std:>10.4f} {delta:>+11.4f} {pval_str:>10} {effect:>8}{marker}")

    # Find best by Mean AUC
    best = max(valid_results, key=lambda r: r.get('mean_auc', 0))
    best_name = best['name']
    best_mean = best.get('mean_auc', 0)
    best_delta = best_mean - BASELINE_MEAN_AUC

    # Statistical significance
    best_stats = statistics.get(best_name, {})
    best_pval = best_stats.get('statistical_test', {}).get('p_value')
    best_sig = best_stats.get('statistical_test', {}).get('significant', False)
    best_effect = best_stats.get('effect_size', {}).get('cohens_d', 0)
    best_ci = best_stats.get('confidence_interval', {})

    print("\n" + "=" * 80)
    print("BEST COMBINATION ANALYSIS")
    print("=" * 80)
    print(f"\nBest: {best_name}")
    print(f"Mean AUC: {best_mean:.4f} (Δ = {best_delta:+.4f})")

    if best_pval is not None:
        print(f"Statistical test: t={best_stats['statistical_test']['t_statistic']:.4f}, p={best_pval:.4f}")
        if best_sig:
            print(f"✓ Statistically significant improvement over baseline (α=0.05)")
        else:
            print(f"✗ Not statistically significant (p > 0.05)")

    print(f"Effect size (Cohen's d): {best_effect:.4f} ({best_stats['effect_size']['interpretation']})")

    if best_ci:
        print(f"95% CI for difference: [{best_ci['ci_lower']:.4f}, {best_ci['ci_upper']:.4f}]")
        if best_ci['includes_zero']:
            print(f"  CI includes zero → cannot rule out no effect")

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION FOR PHASE 5")
    print("=" * 80)

    if best_delta > 0.01 and best_sig:
        recommendation = (
            f"✓ {best_name} significantly improves over baseline (+{best_delta:.2%})\n"
            f"  Use this method for Phase 5 ensembling.\n"
            f"  Include top 3 combinations in ensemble."
        )
    elif best_delta > 0.005:
        recommendation = (
            f"~ {best_name} shows improvement (+{best_delta:.2%}) but not significant\n"
            f"  Consider including in Phase 5 ensemble with baseline.\n"
            f"  Ensemble may stabilize small improvements."
        )
    elif best_delta > -0.005:
        recommendation = (
            f"≈ {best_name} similar to baseline (Δ={best_delta:+.2%})\n"
            f"  Combinations don't show clear advantage.\n"
            f"  Phase 5 ensemble should focus on best single methods from Phases 1-2."
        )
    else:
        recommendation = (
            f"✗ {best_name} worse than baseline by {best_delta:.2%}\n"
            f"  Combinations underperform.\n"
            f"  Phase 5 should use single methods only (PLS n=20, LGB FS, etc.)."
        )

    print(recommendation)
    print("=" * 80)


def save_combined_results(
    results: List[Dict[str, Any]],
    statistics: Dict[str, Dict],
    skipped: List[str]
):
    """Save combined Phase 4 results to JSON."""
    best = max([r for r in results if 'error' not in r], key=lambda r: r.get('mean_auc', 0))
    best_name = best['name']
    best_stats = statistics.get(best_name, {})

    # Generate recommendation text
    best_delta = best.get('mean_auc', 0) - BASELINE_MEAN_AUC
    best_sig = best_stats.get('statistical_test', {}).get('significant', False)

    if best_delta > 0.01 and best_sig:
        rec = f"{best_name} significantly improves over baseline (+{best_delta:.2%})"
    elif best_delta > 0.005:
        rec = f"{best_name} shows improvement (+{best_delta:.2%}) but not significant"
    elif best_delta > -0.005:
        rec = f"{best_name} similar to baseline (Δ={best_delta:+.2%})"
    else:
        rec = f"{best_name} worse than baseline by {best_delta:.2%}%"

    output = {
        'phase': 4,
        'timestamp': datetime.now().isoformat(),
        'n_experiments_total': len(EXPERIMENTS),
        'n_experiments_completed': len([r for r in results if 'error' not in r]),
        'n_experiments_skipped': len(skipped),
        'baseline_comparison': {
            'kpn_auc': BASELINE_KPN_AUC,
            'mean_auc': BASELINE_MEAN_AUC,
            'source_file': BASELINE_FILE
        },
        'results': results,
        'statistics': statistics,
        'recommendations': rec,
        'best_experiment': {
            'name': best_name,
            'mean_auc': best.get('mean_auc'),
            'delta_from_baseline': best_delta,
            'statistically_significant': best_sig
        }
    }

    output_path = OUTPUT_DIR / "phase4_combined_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nCombined results saved to: {output_path}")


def main():
    """Run Phase 4 experiments."""
    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    print("=" * 80)
    print("PHASE 4: COMBINATION EXPERIMENTS")
    print("=" * 80)
    print(f"\nRunning {len(EXPERIMENTS)} experiments...")
    print(f"Output directory: {OUTPUT_DIR}")

    # Load baseline fold AUCs for statistical testing
    print("\n[Loading baseline from Phase 2...]")
    baseline_folds = load_baseline_folds()
    print(f"  Baseline: PLS(n=20), K.pn fold AUCs = {[f'{f:.4f}' for f in baseline_folds]}")

    # Run experiments with checkpointing
    results, skipped = run_phase4_with_checkpointing()

    # Compute statistics
    print("\n" + "=" * 80)
    print("Computing statistical analysis...")
    statistics = compute_all_statistics(results, baseline_folds)

    # Save combined results
    save_combined_results(results, statistics, skipped)

    # Print summary
    print_phase4_summary(results, statistics)

    # Print timing
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\nTotal runtime: {duration}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
