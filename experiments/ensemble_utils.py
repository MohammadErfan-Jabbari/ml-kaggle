#!/usr/bin/env python3
"""
Ensemble utilities for Phase 5.

Provides functions for:
- Prediction averaging (equal and weighted)
- Stacking with meta-learners (LogisticRegression, LightGBM)
- Ensemble evaluation using same metrics as ControlledExperiment

Usage:
    from ensemble_utils import average_predictions, train_stacking_lr, evaluate_ensemble
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data.dataset import ANTIBIOTICS

# Import from controlled_experiment
from controlled_experiment import (
    compute_metrics,
    apply_intrinsic_rules,
    compute_sample_weights,
    SPECIES_NAMES
)

# Constants
OUTPUT_DIR = Path("/sata_disk/users/erfan/ml_kaggle/outputs/experiments")
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# PREDICTION AVERAGING
# =============================================================================

def average_predictions(
    prediction_dicts: List[Dict[str, Any]],
    weights: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Average predictions from multiple methods.

    Args:
        prediction_dicts: List of dicts with 'oof_predictions' and 'test_predictions'
        weights: Optional weights (default: equal weights)

    Returns:
        Dict with averaged 'oof_predictions', 'test_predictions', and metadata

    Example:
        predictions = [
            {'oof_predictions': oof1, 'test_predictions': test1},
            {'oof_predictions': oof2, 'test_predictions': test2}
        ]
        avg = average_predictions(predictions, weights=[0.6, 0.4])
    """
    n_models = len(prediction_dicts)

    if weights is None:
        weights = [1.0 / n_models] * n_models

    if len(weights) != n_models:
        raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({n_models})")

    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()

    # Initialize with first model
    oof_avg = prediction_dicts[0]['oof_predictions'].copy() * weights[0]
    test_avg = prediction_dicts[0]['test_predictions'].copy() * weights[0]

    # Add remaining models
    for pred_dict, weight in zip(prediction_dicts[1:], weights[1:]):
        oof_avg += pred_dict['oof_predictions'] * weight
        test_avg += pred_dict['test_predictions'] * weight

    return {
        'oof_predictions': oof_avg,
        'test_predictions': test_avg,
        'weights': weights.tolist(),
        'n_models': n_models,
        'base_models': [p.get('name', 'unknown') for p in prediction_dicts]
    }


def compute_weights_by_kpn_auc(kpn_aucs: List[float]) -> List[float]:
    """
    Compute weights proportional to K.pneumoniae AUC.

    Args:
        kpn_aucs: List of K.pneumoniae AUC values

    Returns:
        Normalized weights summing to 1.0
    """
    total = sum(kpn_aucs)
    return [auc / total for auc in kpn_aucs]


# =============================================================================
# STACKING - META-LEARNER TRAINING
# =============================================================================

def prepare_meta_features(
    oof_predictions_list: List[np.ndarray],
    test_predictions_list: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare meta-features for stacking.

    Stacks predictions from base models into (n_samples, n_base_models, n_targets).

    Args:
        oof_predictions_list: List of OOF predictions from base models
        test_predictions_list: List of test predictions from base models

    Returns:
        X_meta: (n_train_samples, n_base_models, n_targets)
        X_test_meta: (n_test_samples, n_base_models, n_targets)
    """
    X_meta = np.stack(oof_predictions_list, axis=1)
    X_test_meta = np.stack(test_predictions_list, axis=1)

    return X_meta, X_test_meta


def train_stacking_lr(
    oof_predictions_list: List[np.ndarray],
    y_train: np.ndarray,
    species_train: np.ndarray,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train LogisticRegression meta-learner on OOF predictions.

    For each antibiotic, trains a separate LogisticRegression on the
    OOF predictions from base models. Handles missing labels with masking.

    Args:
        oof_predictions_list: List of OOF predictions from base models
        y_train: True labels (may contain NaN)
        species_train: Species IDs (not used in LR but kept for consistency)
        verbose: Print training progress

    Returns:
        Dict with fitted meta-learners (one per antibiotic)
    """
    meta_learners = {}

    X_meta = np.stack(oof_predictions_list, axis=1)  # (n_samples, n_base_models, n_targets)

    for ab_idx, antibiotic in enumerate(ANTIBIOTICS):
        # Get valid samples for this antibiotic
        mask = ~np.isnan(y_train[:, ab_idx])
        n_valid = mask.sum()

        if n_valid < 50:
            if verbose:
                print(f"  WARNING: {antibiotic} has only {n_valid} valid samples, using averaging")
            meta_learners[antibiotic] = None  # Use averaging fallback
            continue

        # Extract features for this antibiotic
        X_ab = X_meta[mask, :, ab_idx]  # (n_valid, n_base_models)
        y_ab = y_train[mask, ab_idx].astype(int)

        # Train meta-learner
        meta_lr = LogisticRegression(
            penalty='l2',
            C=1.0,
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        try:
            meta_lr.fit(X_ab, y_ab)
            meta_learners[antibiotic] = meta_lr
            if verbose:
                print(f"  {antibiotic:35} coef={meta_lr.coef_[0].tolist()}")
        except Exception as e:
            if verbose:
                print(f"  WARNING: {antibiotic} fit failed: {e}, using averaging")
            meta_learners[antibiotic] = None

    return meta_learners


def train_stacking_lgb(
    oof_predictions_list: List[np.ndarray],
    y_train: np.ndarray,
    species_train: np.ndarray,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train LightGBM meta-learner on OOF predictions.

    For each antibiotic, trains a separate LightGBM on the
    OOF predictions from base models. Uses sample weights to
    counteract species distribution shift.

    Args:
        oof_predictions_list: List of OOF predictions from base models
        y_train: True labels (may contain NaN)
        species_train: Species IDs for sample weighting
        verbose: Print training progress

    Returns:
        Dict with fitted meta-learners (one per antibiotic)
    """
    meta_learners = {}

    X_meta = np.stack(oof_predictions_list, axis=1)  # (n_samples, n_base_models, n_targets)

    for ab_idx, antibiotic in enumerate(ANTIBIOTICS):
        # Get valid samples for this antibiotic
        mask = ~np.isnan(y_train[:, ab_idx])
        n_valid = mask.sum()

        if n_valid < 50:
            if verbose:
                print(f"  WARNING: {antibiotic} has only {n_valid} valid samples, using averaging")
            meta_learners[antibiotic] = None  # Use averaging fallback
            continue

        # Extract features and targets for this antibiotic
        X_ab = X_meta[mask, :, ab_idx]  # (n_valid, n_base_models)
        y_ab = y_train[mask, ab_idx].astype(int)
        species_ab = species_train[mask]

        # Compute sample weights (same as ControlledExperiment)
        weights = compute_sample_weights(species_ab, pa_weight=0.3)

        # Train meta-learner
        meta_lgb = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            n_jobs=1
        )

        try:
            meta_lgb.fit(X_ab, y_ab, sample_weight=weights)
            meta_learners[antibiotic] = meta_lgb
            if verbose:
                print(f"  {antibiotic:35} trained on {n_valid} samples")
        except Exception as e:
            if verbose:
                print(f"  WARNING: {antibiotic} fit failed: {e}, using averaging")
            meta_learners[antibiotic] = None

    return meta_learners


def predict_stacking(
    meta_learners: Dict[str, Any],
    test_predictions_list: List[np.ndarray],
    oof_predictions_list: Optional[List[np.ndarray]] = None,
    fallback_to_averaging: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Generate predictions using fitted meta-learners.

    Args:
        meta_learners: Dict of fitted meta-learners (one per antibiotic)
        test_predictions_list: List of test predictions from base models
        oof_predictions_list: Optional list of OOF predictions for meta OOF
        fallback_to_averaging: If True, use averaging when meta-learner is None

    Returns:
        test_predictions: (n_test_samples, n_targets) array
        oof_predictions: (n_train_samples, n_targets) array or None
    """
    X_test_meta = np.stack(test_predictions_list, axis=1)  # (n_test, n_base_models, n_targets)

    test_preds = np.zeros((X_test_meta.shape[0], len(ANTIBIOTICS)))

    # Compute test predictions
    for ab_idx, antibiotic in enumerate(ANTIBIOTICS):
        meta_learner = meta_learners.get(antibiotic)

        if meta_learner is None and fallback_to_averaging:
            # Fallback to averaging
            test_preds[:, ab_idx] = X_test_meta[:, :, ab_idx].mean(axis=1)
        elif meta_learner is None:
            test_preds[:, ab_idx] = 0.5  # Default prediction
        else:
            # Use meta-learner
            X_ab = X_test_meta[:, :, ab_idx]
            test_preds[:, ab_idx] = meta_learner.predict_proba(X_ab)[:, 1]

    # Compute OOF predictions if requested
    oof_preds = None
    if oof_predictions_list is not None:
        X_oof_meta = np.stack(oof_predictions_list, axis=1)
        oof_preds = np.full((X_oof_meta.shape[0], len(ANTIBIOTICS)), np.nan)

        for ab_idx, antibiotic in enumerate(ANTIBIOTICS):
            meta_learner = meta_learners.get(antibiotic)

            if meta_learner is None and fallback_to_averaging:
                oof_preds[:, ab_idx] = X_oof_meta[:, :, ab_idx].mean(axis=1)
            elif meta_learner is None:
                oof_preds[:, ab_idx] = 0.5
            else:
                X_ab = X_oof_meta[:, :, ab_idx]
                # Check for NaN values (missing labels) - only predict on non-NaN rows
                nan_mask = np.isnan(X_ab).any(axis=1)
                if nan_mask.any():
                    # Use averaging for NaN rows
                    oof_preds[nan_mask, ab_idx] = X_ab[nan_mask].mean(axis=1)
                    # Use meta-learner for non-NaN rows
                    if (~nan_mask).any():
                        X_ab_valid = X_ab[~nan_mask]
                        oof_preds[~nan_mask, ab_idx] = meta_learner.predict_proba(X_ab_valid)[:, 1]
                else:
                    oof_preds[:, ab_idx] = meta_learner.predict_proba(X_ab)[:, 1]

    return test_preds, oof_preds


# =============================================================================
# ENSEMBLE EVALUATION
# =============================================================================

def evaluate_ensemble(
    oof_predictions: np.ndarray,
    y_train: np.ndarray,
    species_train: np.ndarray,
    test_predictions: np.ndarray,
    species_test: np.ndarray,
    ensemble_name: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Evaluate ensemble predictions using same metrics as ControlledExperiment.

    Args:
        oof_predictions: OOF predictions from ensemble
        y_train: True labels (may contain NaN)
        species_train: Species IDs for training samples
        test_predictions: Test predictions from ensemble
        species_test: Species IDs for test samples
        ensemble_name: Name of the ensemble experiment
        config: Optional config dict to include in results

    Returns:
        Dict with ensemble results (compatible with ExperimentResult)
    """
    # Compute OOF metrics
    oof_metrics = compute_metrics(y_train, oof_predictions, species_train)

    # Apply intrinsic resistance rules to test predictions
    test_preds_final = apply_intrinsic_rules(test_predictions.copy(), species_test)

    # Build result dict
    result = {
        'name': ensemble_name,
        'timestamp': datetime.now().isoformat(),
        'mean_auc': oof_metrics['mean_auc'],
        'mean_auc_std': 0.0,  # Not applicable for ensemble
        'per_antibiotic': {},
        'per_species': {},
        'config': config or {},
        'oof_predictions': oof_predictions,
        'test_predictions': test_preds_final
    }

    # Parse per-antibiotic metrics
    for ab in ANTIBIOTICS:
        if ab in oof_metrics:
            result['per_antibiotic'][ab] = {
                'mean': oof_metrics[ab],
                'std': None,
                'folds': []
            }

    # Parse per-species metrics
    for species_name in SPECIES_NAMES.values():
        key = f'{species_name}_auc'
        if key in oof_metrics:
            result['per_species'][species_name] = {
                'mean': oof_metrics[key],
                'std': None,
                'folds': []
            }

    # Add convenience keys
    if 'K.pneumoniae' in result['per_species']:
        result['K.pneumoniae_auc'] = result['per_species']['K.pneumoniae']['mean']
        result['K.pneumoniae_auc_std'] = result['per_species']['K.pneumoniae']['std']

    return result


def print_ensemble_summary(results: Dict[str, Dict[str, Any]], baseline_mean: float = 0.7993):
    """
    Print summary of ensemble results.

    Args:
        results: Dict of experiment_id -> result_dict
        baseline_mean: Baseline Mean AUC for comparison (average across 8 antibiotics)
    """
    print("\n" + "=" * 80)
    print("PHASE 5 SUMMARY: ENSEMBLE EXPERIMENTS")
    print("=" * 80)

    print(f"\nBaseline Mean AUC: {baseline_mean:.4f}")
    print(f"Results from {len(results)} ensemble experiments:")

    # Sort by Mean AUC (primary metric - matches leaderboard scoring)
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('mean_auc', 0), reverse=True)

    print(f"\n{'Experiment':<20} {'Mean AUC':>10} {'K.pn AUC':>10} {'Δ Baseline':>12}")
    print("-" * 80)

    for exp_id, result in sorted_results:
        name = result['name']
        mean = result.get('mean_auc', 0)
        kpn = result.get('K.pneumoniae_auc', 0)
        delta = mean - baseline_mean

        marker = " ★" if exp_id == sorted_results[0][0] else ""
        print(f"{exp_id:<20} {mean:>10.4f} {kpn:>10.4f} {delta:>+11.4f}{marker}")

    # Find best
    best_exp_id, best_result = sorted_results[0]
    best_mean = best_result.get('mean_auc', 0)
    best_kpn = best_result.get('K.pneumoniae_auc', 0)

    print("\n" + "=" * 80)
    print("BEST ENSEMBLE")
    print("=" * 80)
    print(f"\nBest: {best_exp_id} ({best_result['name']})")
    print(f"Mean AUC: {best_mean:.4f} (Δ = {best_mean - baseline_mean:+.4f})")
    print(f"K.pn AUC: {best_kpn:.4f} (secondary metric)")

    if best_mean > baseline_mean:
        print(f"✓ Improvement over baseline: +{(best_mean - baseline_mean)*100:.2f}%")
    else:
        print(f"✗ Worse than baseline: {(best_mean - baseline_mean)*100:.2f}%")

    print("=" * 80)


# =============================================================================
# SUBMISSION GENERATION
# =============================================================================

def create_submission(
    test_predictions: np.ndarray,
    sample_ids: np.ndarray,
    output_path: Path,
    ensemble_name: str
) -> None:
    """
    Create Kaggle submission CSV file.

    Args:
        test_predictions: Test predictions array (n_samples, 8)
        sample_ids: Sample IDs for test set
        output_path: Path to save the submission CSV
        ensemble_name: Name of the ensemble (for logging)
    """
    # Create submission DataFrame
    submission = pd.DataFrame({
        'sample_id': sample_ids,
        **{ab: test_predictions[:, idx] for idx, ab in enumerate(ANTIBIOTICS)}
    })

    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    print(f"Submission saved: {output_path}")

    # Validate format
    sample_sub_path = Path("/home/erfan/data/ml_kaggle/raw/sample_submission.csv")
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
    print(f"  Prediction stats:")
    for ab in ANTIBIOTICS:
        mean_val = submission[ab].mean()
        print(f"    {ab:35} {mean_val:.4f}")


def save_combined_results(
    results: Dict[str, Dict[str, Any]],
    output_path: Optional[Path] = None
) -> Path:
    """
    Save combined Phase 5 results to JSON.

    Args:
        results: Dict of experiment_id -> result_dict
        output_path: Path to save the combined results

    Returns:
        Path where results were saved
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "phase5_combined_results.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove prediction arrays from results (too large for JSON)
    results_clean = {}
    for exp_id, result in results.items():
        result_copy = result.copy()
        result_copy.pop('oof_predictions', None)
        result_copy.pop('test_predictions', None)
        results_clean[exp_id] = result_copy

    combined = {
        'phase': 5,
        'timestamp': datetime.now().isoformat(),
        'n_experiments': len(results),
        'results': results_clean
    }

    with open(output_path, 'w') as f:
        import json
        json.dump(combined, f, indent=2, default=str)

    print(f"Combined results saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    # Quick test
    print("Ensemble utilities module loaded successfully!")
    print(f"ANTIBIOTICS: {ANTIBIOTICS}")
    print(f"SPECIES_NAMES: {SPECIES_NAMES}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
