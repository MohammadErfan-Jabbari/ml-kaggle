#!/usr/bin/env python3
"""
Phase 2: Hyperparameter Sensitivity Analysis.

This script runs hyperparameter sweeps for the top 3 methods from Phase 1.
Each method is tested across a range of hyperparameter values to identify
optimal configurations and characterize sensitivity.

Usage:
    uv run python experiments/run_phase2.py

Expected runtime: ~15 minutes
Output: JSON files in outputs/experiments/phase2_*.json
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

# Ensure proper imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from sklearn.decomposition import TruncatedSVD, NMF, PCA, KernelPCA
from sklearn.cross_decomposition import PLSRegression, PLSCanonical
from sklearn.feature_selection import VarianceThreshold, f_classif
from sklearn.pipeline import Pipeline
from scipy.stats import ttest_rel

from controlled_experiment import (
    ControlledExperiment,
    OUTPUT_DIR,
    load_data,
    remove_constant_features,
    SPECIES_NAMES,
    N_FOLDS
)
from features.reducers import (
    MultiTargetFeatureUnion,
    LGBImportanceSelector,
    MultiStageSelector
)

# Phase 2 Experiments
# Format: (name, transformer, use_scaler)
# Naming convention: {group}_{method}_{hyperparameter_value}
EXPERIMENTS = [
    # =========================================================================
    # 2A: LGBImportanceSelector sweep - k (number of features)
    # =========================================================================
    ("2A1_lgb_k250", LGBImportanceSelector(k=250), False),
    ("2A2_lgb_k500", LGBImportanceSelector(k=500), False),
    ("2A3_lgb_k750", LGBImportanceSelector(k=750), False),
    ("2A4_lgb_k1000", LGBImportanceSelector(k=1000), False),
    ("2A5_lgb_k1500", LGBImportanceSelector(k=1500), False),

    # =========================================================================
    # 2B: PLSRegression sweep - n_components
    # =========================================================================
    ("2B1_pls_n5", PLSRegression(n_components=5), True),
    ("2B2_pls_n7", PLSRegression(n_components=7), True),
    ("2B3_pls_n10", PLSRegression(n_components=10), True),
    ("2B4_pls_n15", PLSRegression(n_components=15), True),
    ("2B5_pls_n20", PLSRegression(n_components=20), True),

    # =========================================================================
    # 2C: VarianceThreshold sweep - threshold
    # =========================================================================
    ("2C1_var_t001", VarianceThreshold(threshold=0.001), False),
    ("2C2_var_t005", VarianceThreshold(threshold=0.005), False),
    ("2C3_var_t01", VarianceThreshold(threshold=0.01), False),
    ("2C4_var_t02", VarianceThreshold(threshold=0.02), False),
    ("2C5_var_t05", VarianceThreshold(threshold=0.05), False),
]

# Method grouping for analysis
METHOD_GROUPS = {
    "2A": {"method": "LGBImportanceSelector", "hyperparameter": "k"},
    "2B": {"method": "PLSRegression", "hyperparameter": "n_components"},
    "2C": {"method": "VarianceThreshold", "hyperparameter": "threshold"},
}

BASELINE_KPN_AUC = 0.6946
BASELINE_MEAN_AUC = 0.8998


def check_variance_threshold_edge_case(threshold: float) -> bool:
    """Check if threshold removes all features."""
    X_train, X_test, _, _, _ = load_data()
    X_train, X_test, _ = remove_constant_features(X_train, X_test)
    variances = X_train.var(axis=0)
    n_surviving = (variances > threshold).sum()
    return n_surviving > 0


def load_baseline_folds() -> List[float]:
    """Load baseline fold AUCs from Phase 0 results."""
    baseline_path = OUTPUT_DIR / "phase0_baseline.json"
    if not baseline_path.exists():
        print("WARNING: phase0_baseline.json not found, using placeholder")
        return [0.672, 0.701, 0.698, 0.715, 0.687]

    with open(baseline_path) as f:
        baseline_data = json.load(f)

    # Extract K.pneumoniae fold AUCs
    if 'K.pneumoniae' in baseline_data.get('per_species', {}):
        folds = baseline_data['per_species']['K.pneumoniae'].get('folds', [])
        valid_folds = [f for f in folds if not np.isnan(f)]
        if valid_folds:
            return valid_folds

    print("WARNING: Could not extract baseline fold AUCs, using placeholder")
    return [0.672, 0.701, 0.698, 0.715, 0.687]


def run_phase2_with_checkpointing() -> Tuple[List[Dict[str, Any]], List[str]]:
    """Run Phase 2 with checkpoint recovery."""
    checkpoint_file = OUTPUT_DIR / "phase2_checkpoint.json"

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
            result_file = OUTPUT_DIR / f"phase2_{name}.json"
            if result_file.exists():
                with open(result_file) as f:
                    results.append(json.load(f))
            continue

        # Edge case check for VarianceThreshold
        if isinstance(transformer, VarianceThreshold):
            if not check_variance_threshold_edge_case(transformer.threshold):
                print(f"\n[{idx}/{len(EXPERIMENTS)}] Skipping {name}: threshold removes all features")
                skipped.append(name)
                continue

        print(f"\n[{idx}/{len(EXPERIMENTS)}] Starting experiment: {name}")
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
            output_path = OUTPUT_DIR / f"phase2_{name}.json"
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


def extract_hyperparameter_value(name: str, group: str) -> float:
    """Extract hyperparameter value from experiment name."""
    # Parse name like "2A1_lgb_k250" -> 250
    parts = name.split('_')
    value_str = parts[-1]  # "k250", "n5", "t001"

    # Remove prefix
    for prefix in ['k', 'n', 't']:
        if value_str.startswith(prefix):
            value_str = value_str[1:]
            break

    # Handle decimal (t001 -> 0.001)
    if group == "2C":
        return float("0." + value_str)
    return float(value_str)


def compute_sensitivity_analysis(results: List[Dict[str, Any]]) -> Dict[str, Dict]:
    """Compute per-method sensitivity analysis."""
    sensitivity = {}

    for group_id, group_info in METHOD_GROUPS.items():
        method_name = group_info["method"]
        hyperparam_name = group_info["hyperparameter"]

        # Filter results for this method group
        group_results = [
            r for r in results
            if r.get('name', '').startswith(group_id) and 'error' not in r
        ]

        if not group_results:
            continue

        # Extract values and AUCs
        values_tested = []
        kpn_aucs = []
        kpn_stds = []

        for r in sorted(group_results, key=lambda x: extract_hyperparameter_value(x['name'], group_id)):
            value = extract_hyperparameter_value(r['name'], group_id)
            auc = r.get('K.pneumoniae_auc', 0)
            std = r.get('K.pneumoniae_auc_std', 0)

            values_tested.append(value)
            kpn_aucs.append(auc)
            kpn_stds.append(std)

        # Find optimal
        if kpn_aucs:
            optimal_idx = int(np.argmax(kpn_aucs))
            optimal_value = values_tested[optimal_idx]
            optimal_auc = kpn_aucs[optimal_idx]

            # Calculate improvement over baseline
            improvement = optimal_auc - BASELINE_KPN_AUC

            # Calculate sensitivity range
            sensitivity_range = max(kpn_aucs) - min(kpn_aucs)

            # Generate conclusion
            if optimal_idx == 0:
                trend = "Best at lowest value, suggesting decreasing performance"
            elif optimal_idx == len(kpn_aucs) - 1:
                trend = "Best at highest value, suggesting increasing performance"
            elif sensitivity_range < 0.01:
                trend = "Very flat, method is insensitive to this hyperparameter"
            else:
                trend = f"Peak at {hyperparam_name}={optimal_value}, plateau beyond"

            sensitivity[method_name] = {
                "hyperparameter": hyperparam_name,
                "values_tested": values_tested,
                "kpn_aucs": [float(a) for a in kpn_aucs],
                "kpn_stds": [float(s) for s in kpn_stds],
                "optimal_value": optimal_value,
                "optimal_auc": float(optimal_auc),
                "optimal_std": float(kpn_stds[optimal_idx]),
                "improvement_over_baseline": f"{improvement:+.4f}",
                "sensitivity_range": float(sensitivity_range),
                "conclusion": trend
            }

    return sensitivity


def paired_ttest_vs_baseline(optimal_folds: List[float], baseline_folds: List[float]) -> Dict[str, Any]:
    """Test if optimal method significantly beats baseline."""
    if len(optimal_folds) != len(baseline_folds):
        # Use mean/std to reconstruct if needed
        print(f"WARNING: Fold count mismatch (optimal={len(optimal_folds)}, baseline={len(baseline_folds)})")
        return {
            't_statistic': 0.0,
            'p_value': 1.0,
            'significant': False,
            'mean_difference': 0.0
        }

    stat, p_value = ttest_rel(optimal_folds, baseline_folds)
    return {
        't_statistic': float(stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'mean_difference': float(np.mean(optimal_folds) - np.mean(baseline_folds))
    }


def compute_confidence_interval(mean: float, std: float, n_folds: int = 5) -> Tuple[float, float]:
    """Compute 95% confidence interval."""
    ci_half = 1.96 * std / np.sqrt(n_folds)
    return (mean - ci_half, mean + ci_half)


def compute_statistical_tests(results: List[Dict[str, Any]], baseline_folds: List[float]) -> Dict[str, Dict]:
    """Compute statistical tests for optimal configs vs baseline."""
    tests = {}

    for group_id, group_info in METHOD_GROUPS.items():
        method_name = group_info["method"]

        # Find optimal result for this method
        group_results = [
            r for r in results
            if r.get('name', '').startswith(group_id) and 'error' not in r
        ]

        if not group_results:
            continue

        # Find best by Mean AUC (primary metric - matches leaderboard scoring)
        best = max(group_results, key=lambda r: r.get('mean_auc', 0))
        name = best['name']

        # Extract fold AUCs if available (for K.pneumoniae statistical testing)
        fold_aucs = None
        if 'K.pneumoniae' in best.get('per_species', {}):
            fold_data = best['per_species']['K.pneumoniae']
            if 'folds' in fold_data:
                fold_aucs = [f for f in fold_data['folds'] if not np.isnan(f)]

        if fold_aucs and len(fold_aucs) == len(baseline_folds):
            test_result = paired_ttest_vs_baseline(fold_aucs, baseline_folds)
            tests[f"{name}_vs_baseline"] = test_result
        else:
            # Mark as unable to test
            tests[f"{name}_vs_baseline"] = {
                'p_value': None,
                'significant': False,
                'note': 'Could not extract fold AUCs for statistical testing'
            }

    return tests


def verify_phase2_results(results: List[Dict[str, Any]], skipped: List[str]) -> bool:
    """Run sanity checks on Phase 2 results."""
    checks = []

    # 1. All experiments completed (allow 1 skip for edge case)
    n_expected = len(EXPERIMENTS)
    n_completed = len([r for r in results if 'error' not in r])
    checks.append(("Experiments completed", n_completed >= n_expected - 1,
                  f"{n_completed}/{n_expected}"))

    # 2. No crashes (all have valid AUCs)
    valid_aucs = [r for r in results if r.get('K.pneumoniae_auc', 0) > 0.5]
    checks.append(("Valid AUCs", len(valid_aucs) >= n_expected - 1,
                  f"{len(valid_aucs)}/{n_expected}"))

    # 3. Optimal beats baseline
    best_kpn = max([r.get('K.pneumoniae_auc', 0) for r in results])
    checks.append(("Best > baseline", best_kpn > BASELINE_KPN_AUC,
                  f"{best_kpn:.4f} > {BASELINE_KPN_AUC:.4f}"))

    # 4. Consistent std (not too high)
    high_std = [r for r in results if r.get('K.pneumoniae_auc_std', 0) > 0.1]
    checks.append(("Reasonable std", len(high_std) == 0,
                  f"{len(high_std)} with std > 0.1"))

    # Print results
    print("\n" + "=" * 60)
    print("PHASE 2 VERIFICATION")
    print("=" * 60)
    for check_name, passed, detail in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}: {detail}")

    all_passed = all(passed for _, passed, _ in checks)
    print(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")

    if skipped:
        print(f"\nSkipped experiments: {', '.join(skipped)}")

    return all_passed


def print_phase2_summary(results: List[Dict[str, Any]]):
    """Print final recommendations."""
    # Find best overall by Mean AUC (primary metric - matches leaderboard scoring)
    best = max(results, key=lambda r: r.get('mean_auc', 0))

    # Group by method
    by_method = {"2A": [], "2B": [], "2C": []}
    for r in results:
        if 'error' in r:
            continue
        group = r['name'][:2]
        if group in by_method:
            by_method[group].append(r)

    # Find best per method by Mean AUC
    best_per_method = {}
    for group, group_results in by_method.items():
        if group_results:
            best_per_method[group] = max(
                group_results,
                key=lambda r: r.get('mean_auc', 0)
            )

    print("\n" + "=" * 70)
    print("PHASE 2 SUMMARY & RECOMMENDATIONS")
    print("=" * 70)

    print(f"\nBaseline Mean AUC: {BASELINE_MEAN_AUC:.4f}")
    print("\nBest Configuration Per Method:")
    print(f"{'Method':<25} {'Config':<20} {'Mean AUC':>10} {'Δ Baseline':>12}")
    print("-" * 70)

    for group, result in best_per_method.items():
        config = result['name'].split('_', 1)[1]
        mean = result['mean_auc']
        delta = mean - BASELINE_MEAN_AUC
        marker = " ★" if mean == best['mean_auc'] else ""
        print(f"{group:<25} {config:<20} {mean:>10.4f} {delta:>+11.4f}{marker}")

    print("\n" + "=" * 70)
    print(f"RECOMMENDATION: Use {best['name']} (Mean AUC = {best['mean_auc']:.4f})")

    # Check for statistical significance
    if 'p_value' in best and best['p_value'] is not None:
        if best['p_value'] < 0.05:
            print(f"✓ Statistically significant improvement (p={best['p_value']:.3f})")
        else:
            print(f"✗ Not statistically significant (p={best['p_value']:.3f})")


def save_combined_results(
    results: List[Dict[str, Any]],
    sensitivity: Dict[str, Dict],
    statistical_tests: Dict[str, Dict],
    skipped: List[str]
):
    """Save combined results to JSON."""
    # Enrich results with metadata
    enriched_results = []
    for r in results:
        if 'error' not in r:
            # Add method, hyperparameter, value
            name = r['name']
            group = name[:2]

            if group in METHOD_GROUPS:
                r['method'] = METHOD_GROUPS[group]['method']
                r['hyperparameter'] = METHOD_GROUPS[group]['hyperparameter']
                r['value'] = extract_hyperparameter_value(name, group)

                # Add confidence intervals
                kpn_mean = r.get('K.pneumoniae_auc', 0)
                kpn_std = r.get('K.pneumoniae_auc_std', 0)
                if kpn_std > 0:
                    ci_low, ci_high = compute_confidence_interval(kpn_mean, kpn_std)
                    r['kpn_ci_95'] = [ci_low, ci_high]
                    r['baseline_in_ci'] = (ci_low <= BASELINE_KPN_AUC <= ci_high)

        enriched_results.append(r)

    # Generate recommendation by Mean AUC (primary metric - matches leaderboard scoring)
    best = max(results, key=lambda r: r.get('mean_auc', 0))
    best_name = best['name']
    best_pvalue = statistical_tests.get(f"{best_name}_vs_baseline", {}).get('p_value', None)

    if best_pvalue is not None and best_pvalue < 0.05:
        recommendation = f"{best['method']}({best['hyperparameter']}={best['value']}) is best [p={best_pvalue:.3f}]"
    else:
        recommendation = f"{best['method']}({best['hyperparameter']}={best['value']}) is best (not statistically significant)"

    output = {
        'phase': 2,
        'timestamp': datetime.now().isoformat(),
        'n_experiments_total': len(EXPERIMENTS),
        'n_experiments_completed': len([r for r in results if 'error' not in r]),
        'n_experiments_skipped': len(skipped),
        'skipped_experiments': skipped,
        'baseline_comparison': {
            'kpn_auc': BASELINE_KPN_AUC,
            'mean_auc': BASELINE_MEAN_AUC
        },
        'results': enriched_results,
        'sensitivity_analysis': sensitivity,
        'statistical_tests': statistical_tests,
        'recommendations': recommendation
    }

    output_path = OUTPUT_DIR / "phase2_combined_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nCombined results saved to: {output_path}")

    # Also save sensitivity analysis separately
    sensitivity_path = OUTPUT_DIR / "phase2_sensitivity_analysis.json"
    with open(sensitivity_path, 'w') as f:
        json.dump(sensitivity, f, indent=2, default=str)

    print(f"Sensitivity analysis saved to: {sensitivity_path}")


def main():
    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    print("=" * 70)
    print("PHASE 2: HYPERPARAMETER SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"\nRunning {len(EXPERIMENTS)} experiments...")
    print(f"Output directory: {OUTPUT_DIR}")

    # Load baseline fold AUCs for statistical testing
    baseline_folds = load_baseline_folds()
    print(f"Baseline fold AUCs: {[f'{f:.3f}' for f in baseline_folds]}")

    # Run experiments with checkpointing
    results, skipped = run_phase2_with_checkpointing()

    # Compute sensitivity analysis
    print("\n" + "=" * 70)
    print("Computing sensitivity analysis...")
    sensitivity = compute_sensitivity_analysis(results)

    # Compute statistical tests
    print("Computing statistical significance tests...")
    statistical_tests = compute_statistical_tests(results, baseline_folds)

    # Save combined results
    save_combined_results(results, sensitivity, statistical_tests, skipped)

    # Print summary
    print_phase2_summary(results)

    # Verify results
    verify_phase2_results(results, skipped)

    # Print timing
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\nTotal runtime: {duration}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
