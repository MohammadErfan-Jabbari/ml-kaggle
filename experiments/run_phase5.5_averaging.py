#!/usr/bin/env python3
"""
Phase 5.5b: Ensemble Averaging with Validation.

This script creates ensemble predictions from Phase 2 top methods and
evaluates them on the VALIDATION SET (not OOF).

Key difference from original Phase 5:
- Original: Evaluated on OOF (overfits)
- New: Evaluate on Val (672 samples)

Usage:
    uv run python experiments/run_phase5.5_averaging.py
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add parent and src directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.controlled_experiment import (
    load_data,
    OUTPUT_DIR,
    compute_metrics,
    apply_intrinsic_rules,
    ANTIBIOTICS
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Top 3 methods from Phase 2 with validation
TOP3_METHODS = [
    "pls_n20_val",      # Val K.pn AUC = 0.6905
    "lgb_k500_val",     # Val K.pn AUC = 0.6805
    "var_t005_val",     # Val K.pn AUC = 0.6532
]

BASELINE_VAL_KPN = 0.6595  # From Phase 5.5 baseline
BASELINE_VAL_MEAN = 0.8030  # From Phase 5.5 baseline

# =============================================================================
# LOAD VALIDATION DATA AND PREDICTIONS
# =============================================================================

def load_validation_data():
    """Load validation split data."""
    from src.data.dataset import load_validation_split
    X, X_val, y, y_val, species, species_val = load_validation_split()
    return X, X_val, y, y_val, species, species_val


def load_experiment_results(method_name: str) -> dict:
    """Load experiment results from JSON file."""
    result_path = OUTPUT_DIR / f"phase5.5_{method_name}.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Results not found: {result_path}")

    with open(result_path, 'r') as f:
        return json.load(f)


def load_val_predictions(method_names: list) -> dict:
    """
    Load val_predictions from experiment results.

    CRITICAL: We're loading val_predictions (672 samples), NOT oof_predictions (2688 samples).
    """
    predictions = {}

    for method_name in method_names:
        print(f"\nLoading: {method_name}")
        result = load_experiment_results(method_name)

        # Check if val_predictions exists (it should after our fix)
        # Note: val_predictions is not directly saved in JSON (too large)
        # We need to regenerate predictions or load from a different source

        # WORKAROUND: Since val_predictions is a numpy array and not saved in JSON,
        # we need to regenerate predictions for each method.
        # For now, let's just extract the val metrics

        predictions[method_name] = {
            'val_K_pneumoniae_auc': result.get('val_K_pneumoniae_auc'),
            'val_mean_auc': result.get('val_mean_auc'),
            'name': method_name
        }

    return predictions


# =============================================================================
# GENERATE VALIDATION PREDICTIONS
# =============================================================================

def generate_val_predictions(method_configs: list) -> dict:
    """
    Generate validation predictions for each method.

    Since val_predictions are not saved in JSON, we need to regenerate them.
    This runs ControlledExperiment for each method with use_validation=True.
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import StandardScaler
    from src.features.reducers import LGBImportanceSelector
    from experiments.controlled_experiment import (
        ControlledExperiment,
        remove_constant_features,
        compute_sample_weights
    )
    import lightgbm as lgb
    from sklearn.base import clone

    # Load validation data
    X, X_val, y, y_val, species, species_val = load_validation_data()
    _, X_test, _, _, species_test = load_data()

    # Remove constant features
    X, X_test, feature_mask = remove_constant_features(X, X_test)
    X_val = X_val[:, feature_mask]

    # Store predictions
    val_predictions = {}
    test_predictions = {}

    method_transformers = {
        'pls_n20_val': ('PLS(n=20)', PLSRegression(n_components=20), True),
        'lgb_k500_val': ('LGBImportanceSelector(k=500)', LGBImportanceSelector(k=500), False),
        'var_t005_val': ('VarianceThreshold(t=0.005)', VarianceThreshold(threshold=0.005), False),
    }

    for method_name, transformer_config in method_transformers.items():
        display_name, transformer, use_scaler = transformer_config

        print(f"\nGenerating predictions for: {display_name}")

        # Apply scaler if needed
        if use_scaler:
            scaler = StandardScaler()
            X_t = scaler.fit_transform(X)
            X_val_t = scaler.transform(X_val)
            X_test_t = scaler.transform(X_test)
        else:
            X_t = X.copy()
            X_val_t = X_val.copy()
            X_test_t = X_test.copy()

        # Apply transformer
        if transformer is not None:
            # Check if supervised DR
            from experiments.controlled_experiment import ControlledExperiment as CE
            temp_exp = CE(name="temp", transformer=transformer, use_scaler=False)
            if temp_exp._is_supervised_dr():
                # Fit on fully-labeled samples
                full_label_mask = ~np.isnan(y).any(axis=1)
                transformer.fit(X_t[full_label_mask], y[full_label_mask])
            else:
                transformer.fit(X_t, y)
            X_t = transformer.transform(X_t)
            X_val_t = transformer.transform(X_val_t)
            X_test_t = transformer.transform(X_test_t)

        # Generate val predictions
        val_preds = np.zeros((len(X_val), len(ANTIBIOTICS)))
        test_preds = np.zeros((len(X_test), len(ANTIBIOTICS)))

        for ab_idx, antibiotic in enumerate(ANTIBIOTICS):
            train_mask = ~np.isnan(y[:, ab_idx])

            if train_mask.sum() < 50:
                continue

            # Train model
            model = lgb.LGBMClassifier(**ControlledExperiment.LGB_PARAMS)
            weights = compute_sample_weights(species[train_mask])
            model.fit(X_t[train_mask], y[train_mask, ab_idx], sample_weight=weights)

            # Generate predictions
            val_preds[:, ab_idx] = model.predict_proba(X_val_t)[:, 1]
            test_preds[:, ab_idx] = model.predict_proba(X_test_t)[:, 1]

        val_predictions[method_name] = val_preds
        test_predictions[method_name] = test_preds

        # Compute val metrics to verify
        val_metrics = compute_metrics(y_val, val_preds, species_val)
        print(f"  Val K.pn AUC: {val_metrics.get('K.pneumoniae_auc', 0):.4f}")
        print(f"  Val Mean AUC: {val_metrics.get('mean_auc', 0):.4f}")

    return val_predictions, test_predictions


# =============================================================================
# ENSEMBLE AVERAGING
# =============================================================================

def average_predictions(val_preds_list: list, test_preds_list: list, weights: list = None) -> tuple:
    """
    Average predictions from multiple methods.

    Args:
        val_preds_list: List of val prediction arrays
        test_preds_list: List of test prediction arrays
        weights: Optional weights (default: equal)

    Returns:
        (val_avg, test_avg): Averaged predictions
    """
    n_models = len(val_preds_list)

    if weights is None:
        weights = [1.0 / n_models] * n_models

    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()

    # Initialize with first model
    val_avg = val_preds_list[0].copy() * weights[0]
    test_avg = test_preds_list[0].copy() * weights[0]

    # Add remaining models
    for i in range(1, n_models):
        val_avg += val_preds_list[i] * weights[i]
        test_avg += test_preds_list[i] * weights[i]

    return val_avg, test_avg


def evaluate_ensemble_on_val(val_preds: np.ndarray, y_val: np.ndarray,
                              species_val: np.ndarray, ensemble_name: str) -> dict:
    """Evaluate ensemble predictions on validation set."""
    metrics = compute_metrics(y_val, val_preds, species_val)

    return {
        'name': ensemble_name,
        'val_mean_auc': metrics.get('mean_auc'),
        'val_K_pneumoniae_auc': metrics.get('K.pneumoniae_auc'),
        'val_per_species': {
            'E.coli': metrics.get('E.coli_auc'),
            'K.pneumoniae': metrics.get('K.pneumoniae_auc'),
            'P.mirabilis': metrics.get('P.mirabilis_auc'),
            'P.aeruginosa': metrics.get('P.aeruginosa_auc'),
        }
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run Phase 5.5b averaging ensemble experiments."""
    start_time = datetime.now()

    print("=" * 80)
    print("PHASE 5.5B: ENSEMBLE AVERAGING WITH VALIDATION")
    print("=" * 80)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Load validation data
    print("\nLoading validation data...")
    X, X_val, y, y_val, species, species_val = load_validation_data()
    _, X_test, _, _, species_test = load_data()
    print(f"  Train: {X.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Generate predictions for top 3 methods
    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS FOR TOP 3 METHODS")
    print("=" * 80)

    method_configs = [
        ('pls_n20_val', 'PLS(n=20)'),
        ('lgb_k500_val', 'LGBImportanceSelector(k=500)'),
        ('var_t005_val', 'VarianceThreshold(t=0.005)'),
    ]

    val_predictions, test_predictions = generate_val_predictions(method_configs)

    # Extract val K.pn AUCs for weighted averaging
    val_kpn_aucs = [
        0.6905,  # PLS(n=20)
        0.6805,  # LGBImportanceSelector(k=500)
        0.6532,  # VarianceThreshold(t=0.005)
    ]

    # =============================================================================
    # ENSEMBLE EXPERIMENTS
    # =============================================================================

    print("\n" + "=" * 80)
    print("RUNNING ENSEMBLE EXPERIMENTS")
    print("=" * 80)

    results = {}

    # 5A1: Equal weights averaging
    print("\n[5A1] Equal weights averaging")
    val_preds_list = [val_predictions[m[0]] for m in method_configs]
    test_preds_list = [test_predictions[m[0]] for m in method_configs]

    val_avg, test_avg = average_predictions(val_preds_list, test_preds_list, weights=None)
    result_5a1 = evaluate_ensemble_on_val(val_avg, y_val, species_val, "5A1_Equal_Weights")
    results['5A1'] = result_5a1

    # Sanity check - ensemble should improve over worst (not necessarily bounded by best)
    min_kpn = min(val_kpn_aucs)
    max_kpn = max(val_kpn_aucs)
    avg_kpn = result_5a1['val_K_pneumoniae_auc']
    assert avg_kpn >= min_kpn, f"Averaging failed: {avg_kpn:.4f} < worst {min_kpn:.4f}"
    # Ensemble can be better than best! That's the point of ensembling.
    if avg_kpn > max_kpn:
        print(f"  ★ Ensemble improves over best single: {avg_kpn:.4f} > {max_kpn:.4f}")
    else:
        print(f"  Sanity check passed: {avg_kpn:.4f} >= {min_kpn:.4f}")

    # 5A2: Weighted by Val K.pn AUC
    print("\n[5A2] Weighted by Val K.pn AUC")
    total = sum(val_kpn_aucs)
    weights = [auc / total for auc in val_kpn_aucs]
    print(f"  Weights: {[f'{w:.3f}' for w in weights]}")

    val_avg, test_avg = average_predictions(val_preds_list, test_preds_list, weights=weights)
    result_5a2 = evaluate_ensemble_on_val(val_avg, y_val, species_val, "5A2_Weighted_By_KpN")
    results['5A2'] = result_5a2

    # Sanity check - ensemble should improve over worst
    avg_kpn = result_5a2['val_K_pneumoniae_auc']
    assert avg_kpn >= min_kpn, f"Weighted averaging failed: {avg_kpn:.4f} < worst {min_kpn:.4f}"
    if avg_kpn > max_kpn:
        print(f"  ★ Ensemble improves over best single: {avg_kpn:.4f} > {max_kpn:.4f}")
    else:
        print(f"  Sanity check passed: {avg_kpn:.4f} >= {min_kpn:.4f}")

    # 5A3: Best single (PLS20) - for comparison
    print("\n[5A3] Best single (PLS20)")
    result_5a3 = {
        'name': '5A3_Best_Single_PLS20',
        'val_mean_auc': 0.7935,  # From Phase 2 results
        'val_K_pneumoniae_auc': 0.6905,
    }
    results['5A3'] = result_5a3

    # =============================================================================
    # COMPARISON TABLE
    # =============================================================================

    print("\n" + "=" * 80)
    print("PHASE 5.5B: ENSEMBLE COMPARISON")
    print("=" * 80)

    print(f"\nBaseline (from Phase 5.5):")
    print(f"  Val Mean AUC:     {BASELINE_VAL_MEAN:.4f}")
    print(f"  Val K.pn AUC:     {BASELINE_VAL_KPN:.4f}")

    print(f"\nEnsemble Methods:")
    print(f"\n{'Method':<30} {'Val Mean':>10} {'Val K.pn':>10} {'Δ Baseline':>12}")
    print("-" * 80)

    for exp_id, result in results.items():
        val_mean = result['val_mean_auc']
        val_kpn = result['val_K_pneumoniae_auc']
        delta = val_mean - BASELINE_VAL_MEAN

        marker = " ★" if val_mean == max(r['val_mean_auc'] for r in results.values()) else ""

        print(f"{result['name']:<30} {val_mean:>10.4f} {val_kpn:>10.4f} {delta:>+12.4f}{marker}")

    # Find best by Val Mean AUC (primary metric - matches leaderboard scoring)
    best_exp_id = max(results.items(), key=lambda x: x[1]['val_mean_auc'])[0]
    best_result = results[best_exp_id]

    print("\n" + "=" * 80)
    print("BEST ENSEMBLE (BASED ON VAL MEAN AUC)")
    print("=" * 80)
    print(f"\nBest: {best_result['name']}")
    print(f"  Val Mean AUC:  {best_result['val_mean_auc']:.4f}")
    print(f"  Val K.pn AUC:  {best_result['val_K_pneumoniae_auc']:.4f} (secondary)")
    print(f"  Δ from baseline: {best_result['val_mean_auc'] - BASELINE_VAL_MEAN:+.4f}")

    # Save combined results
    combined_results = {
        'timestamp': datetime.now().isoformat(),
        'phase': '5.5b_averaging',
        'baseline': {
            'val_mean_auc': BASELINE_VAL_MEAN,
            'val_K_pneumoniae_auc': BASELINE_VAL_KPN
        },
        'results': results,
        'best_method': best_exp_id
    }

    output_path = OUTPUT_DIR / "phase5.5b_averaging_combined.json"
    with open(output_path, 'w') as f:
        json.dump(combined_results, f, indent=2, default=str)

    print(f"\n✓ Combined results saved to: {output_path}")

    # Save comparison table to CSV
    comparison_df = pd.DataFrame({
        'Method': [r['name'] for r in results.values()],
        'Val_Mean_AUC': [r['val_mean_auc'] for r in results.values()],
        'Val_K_pn_AUC': [r['val_K_pneumoniae_auc'] for r in results.values()],
        'Delta_Baseline': [r['val_mean_auc'] - BASELINE_VAL_MEAN for r in results.values()]
    })
    comparison_path = OUTPUT_DIR / "phase5.5b_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"✓ Comparison table saved to: {comparison_path}")

    # Print timing
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 80)
    print("PHASE 5.5B: ENSEMBLE AVERAGING COMPLETE")
    print("=" * 80)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration:   {duration}")
    print()


if __name__ == "__main__":
    main()
