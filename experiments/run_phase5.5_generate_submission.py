#!/usr/bin/env python3
"""
Phase 5.5b: Generate Kaggle Submission.

This script generates a Kaggle submission for the best method selected
by validation K.pneumoniae AUC.

Best method: 5A2 Weighted by K.pn AUC (Val K.pn AUC = 0.6967)

Usage:
    uv run python experiments/run_phase5.5_generate_submission.py
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
    ANTIBIOTICS,
    compute_sample_weights,
    remove_constant_features,
    apply_intrinsic_rules
)
from src.data.dataset import load_validation_split

# =============================================================================
# CONFIGURATION
# =============================================================================

# Best method: 5A2 Weighted by K.pn AUC
BEST_METHOD = "5A2_Weighted_By_KpN"
VAL_KPN_AUC = 0.6967
VAL_MEAN_AUC = 0.8142

# Ensemble weights (from Phase 5.5b results)
WEIGHTS = [0.341, 0.336, 0.323]  # PLS20, LGB500, Var005

# =============================================================================
# GENERATE ENSEMBLE PREDICTIONS
# =============================================================================

def generate_ensemble_test_predictions() -> np.ndarray:
    """Generate test predictions for the weighted ensemble."""

    from sklearn.cross_decomposition import PLSRegression
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import StandardScaler
    from src.features.reducers import LGBImportanceSelector
    import lightgbm as lgb
    from sklearn.base import clone

    # Load data
    X, X_test, y, species, species_test = load_data()

    # For validation, we need to use the training subset (without val)
    X_train, X_val, y_train, y_val, species_train, species_val = load_validation_split()

    print(f"Data shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  X_test: {X_test.shape}")

    # Remove constant features
    X_train, X_test, feature_mask = remove_constant_features(X_train, X_test)
    X_val = X_val[:, feature_mask]

    print(f"After constant removal: {X_train.shape[1]} features")

    # =============================================================================
    # METHOD 1: PLS(n=20)
    # =============================================================================

    print("\n[1/3] PLS(n=20)")
    scaler = StandardScaler()
    X_t = scaler.fit_transform(X_train)
    X_val_t = scaler.transform(X_val)
    X_test_t = scaler.transform(X_test)

    pls = PLSRegression(n_components=20)
    # Fit on fully-labeled samples
    full_label_mask = ~np.isnan(y_train).any(axis=1)
    pls.fit(X_t[full_label_mask], y_train[full_label_mask])

    X_t = pls.transform(X_t)
    X_val_t = pls.transform(X_val_t)
    X_test_t = pls.transform(X_test_t)

    test_preds_pls = np.zeros((len(X_test), len(ANTIBIOTICS)))
    for ab_idx, antibiotic in enumerate(ANTIBIOTICS):
        train_mask = ~np.isnan(y_train[:, ab_idx])
        if train_mask.sum() < 50:
            continue
        model = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        )
        weights = compute_sample_weights(species_train[train_mask])
        model.fit(X_t[train_mask], y_train[train_mask, ab_idx], sample_weight=weights)
        test_preds_pls[:, ab_idx] = model.predict_proba(X_test_t)[:, 1]

    print(f"  Test predictions shape: {test_preds_pls.shape}")

    # =============================================================================
    # METHOD 2: LGBImportanceSelector(k=500)
    # =============================================================================

    print("\n[2/3] LGBImportanceSelector(k=500)")
    X_t = X_train.copy()
    X_val_t = X_val.copy()
    X_test_t = X_test.copy()

    selector = LGBImportanceSelector(k=500)
    selector.fit(X_t, y_train)

    X_t = selector.transform(X_t)
    X_val_t = selector.transform(X_val_t)
    X_test_t = selector.transform(X_test_t)

    test_preds_lgb = np.zeros((len(X_test), len(ANTIBIOTICS)))
    for ab_idx, antibiotic in enumerate(ANTIBIOTICS):
        train_mask = ~np.isnan(y_train[:, ab_idx])
        if train_mask.sum() < 50:
            continue
        model = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        )
        weights = compute_sample_weights(species_train[train_mask])
        model.fit(X_t[train_mask], y_train[train_mask, ab_idx], sample_weight=weights)
        test_preds_lgb[:, ab_idx] = model.predict_proba(X_test_t)[:, 1]

    print(f"  Test predictions shape: {test_preds_lgb.shape}")

    # =============================================================================
    # METHOD 3: VarianceThreshold(t=0.005)
    # =============================================================================

    print("\n[3/3] VarianceThreshold(t=0.005)")
    X_t = X_train.copy()
    X_val_t = X_val.copy()
    X_test_t = X_test.copy()

    # VarianceThreshold was already applied (constant removal at threshold 1e-5)
    # The threshold 0.005 is looser, so we need to re-apply
    var_selector = VarianceThreshold(threshold=0.005)
    var_selector.fit(X_t)

    X_t = var_selector.transform(X_t)
    X_val_t = var_selector.transform(X_val_t)
    X_test_t = var_selector.transform(X_test_t)

    print(f"  After VarianceThreshold(0.005): {X_t.shape[1]} features")

    test_preds_var = np.zeros((len(X_test), len(ANTIBIOTICS)))
    for ab_idx, antibiotic in enumerate(ANTIBIOTICS):
        train_mask = ~np.isnan(y_train[:, ab_idx])
        if train_mask.sum() < 50:
            continue
        model = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        )
        weights = compute_sample_weights(species_train[train_mask])
        model.fit(X_t[train_mask], y_train[train_mask, ab_idx], sample_weight=weights)
        test_preds_var[:, ab_idx] = model.predict_proba(X_test_t)[:, 1]

    print(f"  Test predictions shape: {test_preds_var.shape}")

    # =============================================================================
    # ENSEMBLE: WEIGHTED AVERAGING
    # =============================================================================

    print("\n[ENSEMBLE] Weighted averaging")
    print(f"  Weights: {WEIGHTS}")

    # Normalize weights
    weights = np.array(WEIGHTS)
    weights = weights / weights.sum()

    # Average predictions
    test_preds = (test_preds_pls * weights[0] +
                  test_preds_lgb * weights[1] +
                  test_preds_var * weights[2])

    print(f"  Ensemble test predictions shape: {test_preds.shape}")

    return test_preds


# =============================================================================
# CREATE SUBMISSION
# =============================================================================

def create_submission(test_predictions: np.ndarray, output_path: Path):
    """Create Kaggle submission CSV file."""

    # Load test data for sample IDs
    _, X_test, _, _, _ = load_data()
    test_df = pd.read_csv(Path("/home/centcom/data/ml_kaggle/raw/test.csv"))

    # Create submission DataFrame
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        **{ab: test_predictions[:, idx] for idx, ab in enumerate(ANTIBIOTICS)}
    })

    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    print(f"\n✓ Submission saved: {output_path}")

    # Validate format
    sample_sub_path = Path("/home/centcom/data/ml_kaggle/raw/sample_submission.csv")
    if sample_sub_path.exists():
        sample_sub = pd.read_csv(sample_sub_path)

        # Shape check
        if submission.shape != sample_sub.shape:
            print(f"  WARNING: Shape mismatch! Expected {sample_sub.shape}, got {submission.shape}")
        else:
            print(f"  Shape validated: {submission.shape}")

        # Column check
        expected_cols = list(sample_sub.columns)
        actual_cols = list(submission.columns)
        if expected_cols != actual_cols:
            print(f"  WARNING: Column mismatch!")
            print(f"    Expected: {expected_cols}")
            print(f"    Actual: {actual_cols}")
        else:
            print(f"  Columns validated")

        # Value range check
        for ab in ANTIBIOTICS:
            if not submission[ab].between(0, 1).all():
                print(f"  WARNING: {ab} has values outside [0, 1]!")
            else:
                if ab == ANTIBIOTICS[0]:
                    print(f"  Value range validated: [0, 1]")

    # Print basic stats
    print(f"\n  Prediction stats:")
    for ab in ANTIBIOTICS:
        mean_val = submission[ab].mean()
        print(f"    {ab:35} {mean_val:.4f}")

    return submission


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Generate Kaggle submission for best method."""
    start_time = datetime.now()

    print("=" * 80)
    print("PHASE 5.5B: GENERATE KAGGLE SUBMISSION")
    print("=" * 80)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\nBest method: {BEST_METHOD}")
    print(f"  Val K.pn AUC:  {VAL_KPN_AUC:.4f}")
    print(f"  Val Mean AUC:  {VAL_MEAN_AUC:.4f}")
    print(f"  Δ Baseline:    {VAL_KPN_AUC - 0.6595:+.4f}")

    # Generate test predictions
    print("\n" + "=" * 80)
    print("GENERATING TEST PREDICTIONS")
    print("=" * 80)

    test_predictions = generate_ensemble_test_predictions()

    # Apply intrinsic resistance rules
    _, _, _, _, species_test = load_data()
    test_predictions = apply_intrinsic_rules(test_predictions, species_test)

    # Create submission
    print("\n" + "=" * 80)
    print("CREATING SUBMISSION")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    submission_path = OUTPUT_PATH = Path("/home/centcom/data/ml_kaggle/outputs/submissions")
    submission_path = submission_path / f"sub_phase5.5b_{BEST_METHOD}_{timestamp}.csv"

    submission = create_submission(test_predictions, submission_path)

    # Print Kaggle submit command
    print("\n" + "=" * 80)
    print("KAGGLE SUBMIT COMMAND")
    print("=" * 80)

    submit_cmd = (
        f"kaggle competitions submit -c antimicrobial-resistance-prediction-from-maldi-tof \\\n"
        f"  -f {submission_path} \\\n"
        f"  -m 'Phase 5.5b: {BEST_METHOD} (Val K.pn AUC={VAL_KPN_AUC:.4f}, Δ={VAL_KPN_AUC - 0.6595:+.4f})'"
    )

    print(f"\n{submit_cmd}")

    # Print timing
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 80)
    print("SUBMISSION GENERATION COMPLETE")
    print("=" * 80)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration:   {duration}")
    print()


if __name__ == "__main__":
    main()
