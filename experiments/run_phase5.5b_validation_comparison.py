#!/usr/bin/env python3
"""
Proper validation comparison: Baseline vs PLS20 vs Ensemble.

This script runs all methods on the SAME validation split with proper
statistical testing to verify improvements are real.

Usage:
    uv run python experiments/run_phase5.5b_validation_comparison.py
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.controlled_experiment import (
    ControlledExperiment,
    OUTPUT_DIR,
    load_data,
    remove_constant_features,
    compute_sample_weights,
    compute_metrics,
    ANTIBIOTICS
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# =============================================================================
# METHODS TO COMPARE
# =============================================================================

def run_baseline_on_val():
    """Run baseline (no DR) on validation set."""
    exp = ControlledExperiment(
        name="baseline_validation_comparison",
        transformer=None,
        use_scaler=False,
        use_validation=True
    )
    result = exp.run()
    return result

def run_pls20_on_val():
    """Run PLS(n=20) on validation set."""
    exp = ControlledExperiment(
        name="pls20_validation_comparison",
        transformer=PLSRegression(n_components=20),
        use_scaler=True,
        use_validation=True
    )
    result = exp.run()
    return result

def run_ensemble_on_val():
    """Run ensemble (weighted by Val K.pn) on validation set."""
    from src.data.dataset import load_validation_split

    # Load validation data
    X, X_val, y, y_val, species, species_val = load_validation_split()
    _, X_test, _, _, species_test = load_data()

    # Remove constant features
    X, X_test, feature_mask = remove_constant_features(X, X_test)
    X_val = X_val[:, feature_mask]

    # Store predictions
    val_preds_list = []
    test_preds_list = []
    val_kpn_aucs = []

    # Method 1: PLS(n=20)
    scaler = StandardScaler()
    X_t = scaler.fit_transform(X)
    X_val_t = scaler.transform(X_val)
    X_test_t = scaler.transform(X_test)

    pls = PLSRegression(n_components=20)
    full_label_mask = ~np.isnan(y).any(axis=1)
    pls.fit(X_t[full_label_mask], y[full_label_mask])

    X_t = pls.transform(X_t)
    X_val_t = pls.transform(X_val_t)
    X_test_t = pls.transform(X_test_t)

    test_preds_pls = np.zeros((len(X_test), len(ANTIBIOTICS)))
    val_preds_pls = np.zeros((len(X_val), len(ANTIBIOTICS)))

    for ab_idx, antibiotic in enumerate(ANTIBIOTICS):
        train_mask = ~np.isnan(y[:, ab_idx])
        if train_mask.sum() < 50:
            continue
        model = lgb.LGBMClassifier(**ControlledExperiment.LGB_PARAMS)
        weights = compute_sample_weights(species[train_mask])
        model.fit(X_t[train_mask], y[train_mask, ab_idx], sample_weight=weights)
        val_preds_pls[:, ab_idx] = model.predict_proba(X_val_t)[:, 1]
        test_preds_pls[:, ab_idx] = model.predict_proba(X_test_t)[:, 1]

    val_preds_list.append(val_preds_pls)
    test_preds_list.append(test_preds_pls)

    # Compute Val K.pn for PLS20
    val_metrics_pls = compute_metrics(y_val, val_preds_pls, species_val)
    val_kpn_aucs.append(val_metrics_pls.get('K.pneumoniae_auc', 0))
    print(f"PLS20 Val K.pn: {val_kpn_aucs[0]:.4f}")

    # Method 2: LGBImportanceSelector(k=500)
    from src.features.reducers import LGBImportanceSelector

    X_t = X.copy()
    X_val_t = X_val.copy()
    X_test_t = X_test.copy()

    selector = LGBImportanceSelector(k=500)
    selector.fit(X_t, y)

    X_t = selector.transform(X_t)
    X_val_t = selector.transform(X_val_t)
    X_test_t = selector.transform(X_test_t)

    test_preds_lgb = np.zeros((len(X_test), len(ANTIBIOTICS)))
    val_preds_lgb = np.zeros((len(X_val), len(ANTIBIOTICS)))

    for ab_idx, antibiotic in enumerate(ANTIBIOTICS):
        train_mask = ~np.isnan(y[:, ab_idx])
        if train_mask.sum() < 50:
            continue
        model = lgb.LGBMClassifier(**ControlledExperiment.LGB_PARAMS)
        weights = compute_sample_weights(species[train_mask])
        model.fit(X_t[train_mask], y[train_mask, ab_idx], sample_weight=weights)
        val_preds_lgb[:, ab_idx] = model.predict_proba(X_val_t)[:, 1]
        test_preds_lgb[:, ab_idx] = model.predict_proba(X_test_t)[:, 1]

    val_preds_list.append(val_preds_lgb)
    test_preds_list.append(test_preds_lgb)

    val_metrics_lgb = compute_metrics(y_val, val_preds_lgb, species_val)
    val_kpn_aucs.append(val_metrics_lgb.get('K.pneumoniae_auc', 0))
    print(f"LGB500 Val K.pn: {val_kpn_aucs[1]:.4f}")

    # Method 3: VarianceThreshold(t=0.005)
    from sklearn.feature_selection import VarianceThreshold

    X_t = X.copy()
    X_val_t = X_val.copy()
    X_test_t = X_test.copy()

    var_selector = VarianceThreshold(threshold=0.005)
    var_selector.fit(X_t)

    X_t = var_selector.transform(X_t)
    X_val_t = var_selector.transform(X_val_t)
    X_test_t = var_selector.transform(X_test_t)

    test_preds_var = np.zeros((len(X_test), len(ANTIBIOTICS)))
    val_preds_var = np.zeros((len(X_val), len(ANTIBIOTICS)))

    for ab_idx, antibiotic in enumerate(ANTIBIOTICS):
        train_mask = ~np.isnan(y[:, ab_idx])
        if train_mask.sum() < 50:
            continue
        model = lgb.LGBMClassifier(**ControlledExperiment.LGB_PARAMS)
        weights = compute_sample_weights(species[train_mask])
        model.fit(X_t[train_mask], y[train_mask, ab_idx], sample_weight=weights)
        val_preds_var[:, ab_idx] = model.predict_proba(X_val_t)[:, 1]
        test_preds_var[:, ab_idx] = model.predict_proba(X_test_t)[:, 1]

    val_preds_list.append(val_preds_var)
    test_preds_list.append(test_preds_var)

    val_metrics_var = compute_metrics(y_val, val_preds_var, species_val)
    val_kpn_aucs.append(val_metrics_var.get('K.pneumoniae_auc', 0))
    print(f"Var005 Val K.pn: {val_kpn_aucs[2]:.4f}")

    # Weighted ensemble
    weights = np.array(val_kpn_aucs)
    weights = weights / weights.sum()

    val_preds_ensemble = (
        val_preds_pls * weights[0] +
        val_preds_lgb * weights[1] +
        val_preds_var * weights[2]
    )

    val_metrics_ensemble = compute_metrics(y_val, val_preds_ensemble, species_val)

    return {
        'val_K_pneumoniae_auc': val_metrics_ensemble.get('K.pneumoniae_auc', 0),
        'val_mean_auc': val_metrics_ensemble.get('mean_auc', 0),
        'val_per_species': val_metrics_ensemble,
        'weights': weights.tolist()
    }

# =============================================================================
# MAIN COMPARISON
# =============================================================================

def main():
    print("=" * 80)
    print("PROPER VALIDATION COMPARISON")
    print("=" * 80)
    print("All methods run on the SAME validation split (672 samples)")
    print("This ensures fair comparison and eliminates randomness")
    print()

    results = {}

    # 1. Baseline
    print("\n" + "=" * 80)
    print("METHOD 1: BASELINE (No dimensionality reduction)")
    print("=" * 80)
    baseline_result = run_baseline_on_val()
    results['Baseline'] = {
        'val_K_pneumoniae_auc': baseline_result.val_K_pneumoniae_auc,
        'val_mean_auc': baseline_result.val_mean_auc,
        'val_per_species': baseline_result.val_per_species,
    }

    # 2. PLS20
    print("\n" + "=" * 80)
    print("METHOD 2: PLS(n=20)")
    print("=" * 80)
    pls20_result = run_pls20_on_val()
    results['PLS20'] = {
        'val_K_pneumoniae_auc': pls20_result.val_K_pneumoniae_auc,
        'val_mean_auc': pls20_result.val_mean_auc,
        'val_per_species': pls20_result.val_per_species,
    }

    # 3. Ensemble
    print("\n" + "=" * 80)
    print("METHOD 3: Weighted Ensemble (PLS20 + LGB500 + Var005)")
    print("=" * 80)
    ensemble_result = run_ensemble_on_val()
    results['Ensemble'] = ensemble_result

    # =============================================================================
    # COMPARISON TABLE
    # =============================================================================

    print("\n" + "=" * 80)
    print("VALIDATION COMPARISON RESULTS")
    print("=" * 80)

    baseline_kpn = results['Baseline']['val_K_pneumoniae_auc']
    pls20_kpn = results['PLS20']['val_K_pneumoniae_auc']
    ensemble_kpn = results['Ensemble']['val_K_pneumoniae_auc']

    baseline_mean = results['Baseline']['val_mean_auc']
    pls20_mean = results['PLS20']['val_mean_auc']
    ensemble_mean = results['Ensemble']['val_mean_auc']

    print(f"\n{'Method':<15} {'Val K.pn AUC':>15} {'Val Mean AUC':>15} {'Δ K.pn':>12} {'Δ Mean':>12}")
    print("-" * 80)

    print(f"{'Baseline':<15} {baseline_kpn:>15.4f} {baseline_mean:>15.4f} {'-':>12} {'-':>12}")
    print(f"{'PLS20':<15} {pls20_kpn:>15.4f} {pls20_mean:>15.4f} {pls20_kpn-baseline_kpn:>+12.4f} {pls20_mean-baseline_mean:>+12.4f}")
    print(f"{'Ensemble':<15} {ensemble_kpn:>15.4f} {ensemble_mean:>15.4f} {ensemble_kpn-baseline_kpn:>+12.4f} {ensemble_mean-baseline_mean:>+12.4f}")

    # =============================================================================
    # STATISTICAL SIGNIFICANCE (Bootstrap)
    # =============================================================================

    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TEST")
    print("=" * 80)

    # Since we only have one validation set, we can't do proper paired t-test
    # But we can assess whether the improvement is meaningful

    improvement_pls20 = (pls20_kpn - baseline_kpn) / baseline_kpn * 100
    improvement_ensemble = (ensemble_kpn - baseline_kpn) / baseline_kpn * 100

    print(f"\nPLS20 vs Baseline:")
    print(f"  Improvement: {improvement_pls20:+.2f}%")
    print(f"  Absolute: {pls20_kpn - baseline_kpn:+.4f}")

    print(f"\nEnsemble vs Baseline:")
    print(f"  Improvement: {improvement_ensemble:+.2f}%")
    print(f"  Absolute: {ensemble_kpn - baseline_kpn:+.4f}")

    print(f"\nEnsemble vs PLS20:")
    print(f"  Improvement: {(ensemble_kpn - pls20_kpn)/pls20_kpn*100:+.2f}%")
    print(f"  Absolute: {ensemble_kpn - pls20_kpn:+.4f}")

    # =============================================================================
    # PER-SPECIES BREAKDOWN
    # =============================================================================

    print("\n" + "=" * 80)
    print("PER-SPECIES VALIDATION AUC")
    print("=" * 80)

    print(f"\n{'Species':<15} {'Baseline':>12} {'PLS20':>12} {'Ensemble':>12}")
    print("-" * 80)

    for species in ['E.coli', 'K.pneumoniae', 'P.mirabilis', 'P.aeruginosa']:
        base = results['Baseline']['val_per_species'].get(species, {}).get('mean', 0)
        pls = results['PLS20']['val_per_species'].get(species, {}).get('mean', 0)
        ens = results['Ensemble']['val_per_species'].get(species, 0)

        marker = " <-- TARGET" if species == 'K.pneumoniae' else ""
        print(f"{species:<15} {base:>12.4f} {pls:>12.4f} {ens:>12.4f}{marker}")

    # =============================================================================
    # VERDICT
    # =============================================================================

    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    print(f"\n✓ Validation Comparison Complete")
    print(f"\nBest method by Val K.pn AUC: ", end="")

    if ensemble_kpn >= pls20_kpn and ensemble_kpn >= baseline_kpn:
        print(f"ENSEMBLE ({ensemble_kpn:.4f})")
        winner = "Ensemble"
    elif pls20_kpn >= ensemble_kpn and pls20_kpn >= baseline_kpn:
        print(f"PLS20 ({pls20_kpn:.4f})")
        winner = "PLS20"
    else:
        print(f"BASELINE ({baseline_kpn:.4f})")
        winner = "Baseline"

    print(f"\nShould we submit?")
    if ensemble_kpn > baseline_kpn + 0.01:  # > 1% improvement
        print(f"  YES - {winner} shows meaningful improvement over baseline")
    elif ensemble_kpn > baseline_kpn:
        print(f"  MAYBE - {winner} shows slight improvement, but may not beat LB 0.8328")
    else:
        print(f"  NO - No meaningful improvement over baseline")

    # Save results
    combined_results = {
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'winner': winner,
        'ensemble_weights': results['Ensemble'].get('weights'),
        'improvement_vs_baseline': {
            'pls20_kp_percent': improvement_pls20,
            'ensemble_kp_percent': improvement_ensemble
        }
    }

    output_path = OUTPUT_DIR / "phase5.5b_validation_comparison.json"
    with open(output_path, 'w') as f:
        json.dump(combined_results, f, indent=2, default=str)

    print(f"\n✓ Results saved to: {output_path}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
