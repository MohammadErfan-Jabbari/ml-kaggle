#!/usr/bin/env python3
"""
Comprehensive experimentation framework for AMR prediction.

Tests 10+ different modeling approaches and compares them systematically.
Each experiment outputs OOF predictions and metrics for comparison.

Run with: uv run python experiments/run_all_experiments.py
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import KMeans
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data.dataset import ANTIBIOTICS

# Constants
DATA_DIR = Path("/home/erfan/data/ml_kaggle/raw")
OUTPUT_DIR = Path("/sata_disk/users/erfan/ml_kaggle/outputs/experiments")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPECIES_NAMES = {0: "E.coli", 1: "K.pneumoniae", 2: "P.mirabilis", 3: "P.aeruginosa"}
N_FOLDS = 5
RANDOM_STATE = 42

# Global results storage
EXPERIMENT_RESULTS = {}


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess data."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    feature_cols = [f"maldi_feature_{i}" for i in range(6000)]
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    y_train = train_df[ANTIBIOTICS].values
    species_train = train_df["species_id"].values
    species_test = test_df["species_id"].values

    return X_train, X_test, y_train, species_train, species_test


def remove_constant_features(X_train: np.ndarray, X_test: np.ndarray,
                             threshold: float = 1e-5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove features with variance below threshold."""
    variances = X_train.var(axis=0)
    mask = variances > threshold
    return X_train[:, mask], X_test[:, mask], mask


def compute_sample_weights(species: np.ndarray, pa_weight: float = 0.3) -> np.ndarray:
    """Compute sample weights to counteract species distribution shift."""
    return np.where(species == 3, pa_weight, 1.0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    species: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive metrics including per-species AUCs."""
    metrics = {}

    # Per-antibiotic AUC
    antibiotic_aucs = []
    for idx, antibiotic in enumerate(ANTIBIOTICS):
        mask = ~np.isnan(y_true[:, idx])
        if mask.sum() > 10 and len(np.unique(y_true[mask, idx])) > 1:
            auc = roc_auc_score(y_true[mask, idx], y_pred[mask, idx])
            metrics[antibiotic] = auc
            antibiotic_aucs.append(auc)

    metrics['mean_auc'] = np.mean(antibiotic_aucs) if antibiotic_aucs else 0.0

    # Per-species AUC (critical for K.pneumoniae)
    for species_id, species_name in SPECIES_NAMES.items():
        species_mask = (species == species_id)
        if species_mask.sum() > 0:
            species_aucs = []
            for idx, antibiotic in enumerate(ANTIBIOTICS):
                label_mask = ~np.isnan(y_true[:, idx])
                combined_mask = species_mask & label_mask
                if combined_mask.sum() > 10:
                    y_s = y_true[combined_mask, idx]
                    p_s = y_pred[combined_mask, idx]
                    if len(np.unique(y_s)) > 1:
                        species_aucs.append(roc_auc_score(y_s, p_s))

            if species_aucs:
                metrics[f'{species_name}_auc'] = np.mean(species_aucs)

    return metrics


def print_metrics(name: str, metrics: Dict[str, float]):
    """Print metrics in formatted table."""
    print(f"\n{'='*60}")
    print(f"Results: {name}")
    print(f"{'='*60}")
    print(f"Mean AUC: {metrics.get('mean_auc', 0):.4f}")
    print(f"\nPer-Species (critical - K.pneumoniae is 51% of test!):")
    for species_name in SPECIES_NAMES.values():
        key = f'{species_name}_auc'
        if key in metrics:
            marker = " <-- PRIMARY TARGET" if species_name == "K.pneumoniae" else ""
            print(f"  {species_name:15} {metrics[key]:.4f}{marker}")
    print(f"\nPer-Antibiotic:")
    for antibiotic in ANTIBIOTICS:
        if antibiotic in metrics:
            print(f"  {antibiotic:30} {metrics[antibiotic]:.4f}")


# =============================================================================
# EXPERIMENT 1: Species-Specific Models
# =============================================================================

def experiment_species_specific(X_train: np.ndarray, y_train: np.ndarray,
                                 species_train: np.ndarray,
                                 X_test: np.ndarray = None,
                                 species_test: np.ndarray = None) -> Tuple[np.ndarray, Dict]:
    """
    Train separate LightGBM models for each species.

    Hypothesis: Species have different resistance mechanisms.
    Training species-specific models will capture these patterns better.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Species-Specific Models")
    print("="*60)

    oof_preds = np.full_like(y_train, np.nan, dtype=float)
    test_preds = np.zeros((len(X_test), len(ANTIBIOTICS))) if X_test is not None else None

    for species_id, species_name in SPECIES_NAMES.items():
        print(f"\n[Species: {species_name}]")
        species_mask = (species_train == species_id)
        n_samples = species_mask.sum()
        print(f"  Training samples: {n_samples}")

        if n_samples < 50:
            print(f"  WARNING: Very few samples, using global model fallback")
            continue

        X_species = X_train[species_mask]
        y_species = y_train[species_mask]

        for idx, antibiotic in enumerate(ANTIBIOTICS):
            label_mask = ~np.isnan(y_species[:, idx])
            n_labeled = label_mask.sum()

            if n_labeled < 20:
                print(f"    {antibiotic}: Too few labeled samples ({n_labeled}), skipping")
                continue

            X_ab = X_species[label_mask]
            y_ab = y_species[label_mask, idx]

            # Check if we have both classes
            if len(np.unique(y_ab)) < 2:
                # Constant label - predict that value
                oof_preds[species_mask, idx] = y_ab[0]
                if test_preds is not None:
                    test_mask = (species_test == species_id)
                    test_preds[test_mask, idx] = y_ab[0]
                continue

            # 5-fold CV within species
            skf = StratifiedKFold(n_splits=min(5, n_labeled // 10), shuffle=True, random_state=RANDOM_STATE)

            fold_preds = np.zeros(n_labeled)
            fold_models = []

            for train_idx, val_idx in skf.split(X_ab, y_ab):
                model = lgb.LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    num_leaves=15,
                    min_child_samples=max(5, n_labeled // 50),
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=RANDOM_STATE,
                    verbose=-1,
                )
                model.fit(X_ab[train_idx], y_ab[train_idx])
                fold_preds[val_idx] = model.predict_proba(X_ab[val_idx])[:, 1]
                fold_models.append(model)

            # Map back to original indices
            species_indices = np.where(species_mask)[0]
            labeled_indices = species_indices[label_mask]
            oof_preds[labeled_indices, idx] = fold_preds

            # Test predictions (ensemble of fold models)
            if test_preds is not None:
                test_mask = (species_test == species_id)
                if test_mask.sum() > 0:
                    test_species = X_test[test_mask]
                    test_preds_species = np.mean([m.predict_proba(test_species)[:, 1] for m in fold_models], axis=0)
                    test_preds[test_mask, idx] = test_preds_species

    # Apply intrinsic resistance rules
    oof_preds = apply_intrinsic_rules(oof_preds, species_train)
    if test_preds is not None:
        test_preds = apply_intrinsic_rules(test_preds, species_test)

    metrics = compute_metrics(y_train, oof_preds, species_train)
    print_metrics("Species-Specific Models", metrics)

    return oof_preds, test_preds, metrics


def apply_intrinsic_rules(predictions: np.ndarray, species_ids: np.ndarray) -> np.ndarray:
    """Apply biological intrinsic resistance rules."""
    predictions = predictions.copy()

    ANTIBIOTIC_INDICES = {ab: idx for idx, ab in enumerate(ANTIBIOTICS)}

    # P. aeruginosa intrinsic resistance
    pa_mask = (species_ids == 3)
    for ab in ["Ampicillin", "Amoxicillin_Clavulanic_acid", "Ertapenem", "Cefotaxime", "Cefuroxime"]:
        predictions[pa_mask, ANTIBIOTIC_INDICES[ab]] = 1.0

    # P. mirabilis intrinsic resistance to Imipenem
    pm_mask = (species_ids == 2)
    predictions[pm_mask, ANTIBIOTIC_INDICES["Imipenem"]] = 1.0

    return predictions


# =============================================================================
# EXPERIMENT 2: PLS Feature Extraction + LightGBM
# =============================================================================

def experiment_pls_features(X_train: np.ndarray, y_train: np.ndarray,
                            species_train: np.ndarray,
                            X_test: np.ndarray = None,
                            species_test: np.ndarray = None) -> Tuple[np.ndarray, Dict]:
    """
    Use PLS to extract supervised features, then train LightGBM.

    From course: PLS finds features that maximize covariance with target.
    Better than PCA for prediction tasks.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: PLS Features + LightGBM")
    print("="*60)

    oof_preds = np.full_like(y_train, np.nan, dtype=float)
    test_preds = np.zeros((len(X_test), len(ANTIBIOTICS))) if X_test is not None else None

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None

    n_components = 100  # PLS components

    for idx, antibiotic in enumerate(ANTIBIOTICS):
        print(f"\n[{idx+1}/8] {antibiotic}")

        label_mask = ~np.isnan(y_train[:, idx])
        n_labeled = label_mask.sum()

        if n_labeled < 50:
            print(f"  Too few labeled samples ({n_labeled})")
            continue

        X_ab = X_train_scaled[label_mask]
        y_ab = y_train[label_mask, idx]
        species_ab = species_train[label_mask]

        if len(np.unique(y_ab)) < 2:
            oof_preds[label_mask, idx] = y_ab[0]
            if test_preds is not None:
                test_preds[:, idx] = y_ab[0]
            continue

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_preds = np.zeros(n_labeled)
        fold_test_preds = []

        for train_idx, val_idx in skf.split(X_ab, species_ab):
            # Fit PLS on training fold
            pls = PLSRegression(n_components=min(n_components, len(train_idx) - 1))
            pls.fit(X_ab[train_idx], y_ab[train_idx])

            # Transform
            X_train_pls = pls.transform(X_ab[train_idx])
            X_val_pls = pls.transform(X_ab[val_idx])

            # Train LightGBM
            model = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                random_state=RANDOM_STATE,
                verbose=-1,
            )

            weights = compute_sample_weights(species_ab[train_idx])
            model.fit(X_train_pls, y_ab[train_idx], sample_weight=weights)

            fold_preds[val_idx] = model.predict_proba(X_val_pls)[:, 1]

            if X_test_scaled is not None:
                X_test_pls = pls.transform(X_test_scaled)
                fold_test_preds.append(model.predict_proba(X_test_pls)[:, 1])

        oof_preds[label_mask, idx] = fold_preds

        if test_preds is not None and fold_test_preds:
            test_preds[:, idx] = np.mean(fold_test_preds, axis=0)

    # Apply intrinsic rules
    oof_preds = apply_intrinsic_rules(oof_preds, species_train)
    if test_preds is not None:
        test_preds = apply_intrinsic_rules(test_preds, species_test)

    metrics = compute_metrics(y_train, oof_preds, species_train)
    print_metrics("PLS Features + LightGBM", metrics)

    return oof_preds, test_preds, metrics


# =============================================================================
# EXPERIMENT 3: Stacked Ensemble (LightGBM + LogReg meta-learner)
# =============================================================================

def experiment_stacked_ensemble(X_train: np.ndarray, y_train: np.ndarray,
                                 species_train: np.ndarray,
                                 X_test: np.ndarray = None,
                                 species_test: np.ndarray = None) -> Tuple[np.ndarray, Dict]:
    """
    Two-level stacking:
    Level 1: Multiple base learners (LightGBM with different configs)
    Level 2: Logistic Regression meta-learner

    From course: Ensemble methods reduce variance.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: Stacked Ensemble")
    print("="*60)

    from sklearn.linear_model import LogisticRegression

    oof_preds = np.full_like(y_train, np.nan, dtype=float)
    test_preds = np.zeros((len(X_test), len(ANTIBIOTICS))) if X_test is not None else None

    # Base learner configurations
    base_configs = [
        {"n_estimators": 100, "num_leaves": 15, "learning_rate": 0.1},
        {"n_estimators": 200, "num_leaves": 31, "learning_rate": 0.05},
        {"n_estimators": 300, "num_leaves": 63, "learning_rate": 0.03},
        {"n_estimators": 150, "num_leaves": 7, "learning_rate": 0.1, "max_depth": 3},
    ]

    for idx, antibiotic in enumerate(ANTIBIOTICS):
        print(f"\n[{idx+1}/8] {antibiotic}")

        label_mask = ~np.isnan(y_train[:, idx])
        n_labeled = label_mask.sum()

        if n_labeled < 50:
            continue

        X_ab = X_train[label_mask]
        y_ab = y_train[label_mask, idx]
        species_ab = species_train[label_mask]

        if len(np.unique(y_ab)) < 2:
            oof_preds[label_mask, idx] = y_ab[0]
            if test_preds is not None:
                test_preds[:, idx] = y_ab[0]
            continue

        # Generate level-1 OOF predictions
        level1_oof = np.zeros((n_labeled, len(base_configs)))
        level1_test = np.zeros((len(X_test), len(base_configs))) if X_test is not None else None

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        for config_idx, config in enumerate(base_configs):
            fold_test_preds = []

            for train_idx, val_idx in skf.split(X_ab, species_ab):
                model = lgb.LGBMClassifier(
                    **config,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=RANDOM_STATE,
                    verbose=-1,
                )

                weights = compute_sample_weights(species_ab[train_idx])
                model.fit(X_ab[train_idx], y_ab[train_idx], sample_weight=weights)

                level1_oof[val_idx, config_idx] = model.predict_proba(X_ab[val_idx])[:, 1]

                if X_test is not None:
                    fold_test_preds.append(model.predict_proba(X_test)[:, 1])

            if level1_test is not None and fold_test_preds:
                level1_test[:, config_idx] = np.mean(fold_test_preds, axis=0)

        # Level-2: Meta-learner
        meta = LogisticRegression(C=1.0, max_iter=1000)
        meta.fit(level1_oof, y_ab)

        oof_preds[label_mask, idx] = meta.predict_proba(level1_oof)[:, 1]

        if test_preds is not None:
            test_preds[:, idx] = meta.predict_proba(level1_test)[:, 1]

    # Apply intrinsic rules
    oof_preds = apply_intrinsic_rules(oof_preds, species_train)
    if test_preds is not None:
        test_preds = apply_intrinsic_rules(test_preds, species_test)

    metrics = compute_metrics(y_train, oof_preds, species_train)
    print_metrics("Stacked Ensemble", metrics)

    return oof_preds, test_preds, metrics


# =============================================================================
# EXPERIMENT 4: Pseudo-labeling for Missing Labels
# =============================================================================

def experiment_pseudo_labeling(X_train: np.ndarray, y_train: np.ndarray,
                                species_train: np.ndarray,
                                X_test: np.ndarray = None,
                                species_test: np.ndarray = None) -> Tuple[np.ndarray, Dict]:
    """
    Semi-supervised learning using pseudo-labeling.

    1. Train on labeled data
    2. Predict on unlabeled data (within train)
    3. Add high-confidence predictions as pseudo-labels
    4. Retrain on augmented data

    From course: Cluster-based labeling for semi-supervised learning.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 4: Pseudo-labeling")
    print("="*60)

    CONFIDENCE_THRESHOLD = 0.9  # Only use predictions > 0.9 or < 0.1

    oof_preds = np.full_like(y_train, np.nan, dtype=float)
    test_preds = np.zeros((len(X_test), len(ANTIBIOTICS))) if X_test is not None else None

    for idx, antibiotic in enumerate(ANTIBIOTICS):
        print(f"\n[{idx+1}/8] {antibiotic}")

        label_mask = ~np.isnan(y_train[:, idx])
        unlabeled_mask = np.isnan(y_train[:, idx])
        n_labeled = label_mask.sum()
        n_unlabeled = unlabeled_mask.sum()

        print(f"  Labeled: {n_labeled}, Unlabeled: {n_unlabeled}")

        if n_labeled < 50 or n_unlabeled < 10:
            # Fall back to standard training
            if n_labeled >= 50:
                X_ab = X_train[label_mask]
                y_ab = y_train[label_mask, idx]
                species_ab = species_train[label_mask]

                if len(np.unique(y_ab)) >= 2:
                    model = lgb.LGBMClassifier(n_estimators=100, verbose=-1)
                    model.fit(X_ab, y_ab)
                    oof_preds[label_mask, idx] = model.predict_proba(X_ab)[:, 1]
                    if test_preds is not None:
                        test_preds[:, idx] = model.predict_proba(X_test)[:, 1]
            continue

        X_labeled = X_train[label_mask]
        y_labeled = y_train[label_mask, idx]
        X_unlabeled = X_train[unlabeled_mask]
        species_labeled = species_train[label_mask]

        if len(np.unique(y_labeled)) < 2:
            oof_preds[label_mask, idx] = y_labeled[0]
            if test_preds is not None:
                test_preds[:, idx] = y_labeled[0]
            continue

        # Phase 1: Train initial model
        model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
        weights = compute_sample_weights(species_labeled)
        model.fit(X_labeled, y_labeled, sample_weight=weights)

        # Predict on unlabeled
        unlabeled_probs = model.predict_proba(X_unlabeled)[:, 1]

        # Select high-confidence pseudo-labels
        high_conf_mask = (unlabeled_probs > CONFIDENCE_THRESHOLD) | (unlabeled_probs < (1 - CONFIDENCE_THRESHOLD))
        n_pseudo = high_conf_mask.sum()
        print(f"  High-confidence pseudo-labels: {n_pseudo}")

        if n_pseudo > 0:
            # Create pseudo-labels
            pseudo_labels = (unlabeled_probs[high_conf_mask] > 0.5).astype(float)

            # Augment training data
            X_augmented = np.vstack([X_labeled, X_unlabeled[high_conf_mask]])
            y_augmented = np.concatenate([y_labeled, pseudo_labels])
            species_augmented = np.concatenate([species_labeled, species_train[unlabeled_mask][high_conf_mask]])

            # Retrain on augmented data
            model = lgb.LGBMClassifier(
                n_estimators=150,
                learning_rate=0.1,
                num_leaves=31,
                random_state=RANDOM_STATE,
                verbose=-1,
            )
            weights_aug = compute_sample_weights(species_augmented)
            # Down-weight pseudo-labels
            weights_aug[len(y_labeled):] *= 0.5
            model.fit(X_augmented, y_augmented, sample_weight=weights_aug)

        # Generate OOF predictions using CV
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_preds = np.zeros(n_labeled)
        fold_test_preds = []

        for train_idx, val_idx in skf.split(X_labeled, species_labeled):
            model_cv = lgb.LGBMClassifier(
                n_estimators=150,
                learning_rate=0.1,
                num_leaves=31,
                random_state=RANDOM_STATE,
                verbose=-1,
            )
            weights_cv = compute_sample_weights(species_labeled[train_idx])
            model_cv.fit(X_labeled[train_idx], y_labeled[train_idx], sample_weight=weights_cv)

            fold_preds[val_idx] = model_cv.predict_proba(X_labeled[val_idx])[:, 1]

            if X_test is not None:
                fold_test_preds.append(model_cv.predict_proba(X_test)[:, 1])

        oof_preds[label_mask, idx] = fold_preds

        if test_preds is not None and fold_test_preds:
            test_preds[:, idx] = np.mean(fold_test_preds, axis=0)

    # Apply intrinsic rules
    oof_preds = apply_intrinsic_rules(oof_preds, species_train)
    if test_preds is not None:
        test_preds = apply_intrinsic_rules(test_preds, species_test)

    metrics = compute_metrics(y_train, oof_preds, species_train)
    print_metrics("Pseudo-labeling", metrics)

    return oof_preds, test_preds, metrics


# =============================================================================
# EXPERIMENT 5: XGBoost Comparison
# =============================================================================

def experiment_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                       species_train: np.ndarray,
                       X_test: np.ndarray = None,
                       species_test: np.ndarray = None) -> Tuple[np.ndarray, Dict]:
    """
    XGBoost as alternative to LightGBM.
    Different splitting algorithm may capture different patterns.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 5: XGBoost")
    print("="*60)

    try:
        import xgboost as xgb
    except ImportError:
        print("XGBoost not installed. Skipping...")
        return None, None, {}

    oof_preds = np.full_like(y_train, np.nan, dtype=float)
    test_preds = np.zeros((len(X_test), len(ANTIBIOTICS))) if X_test is not None else None

    for idx, antibiotic in enumerate(ANTIBIOTICS):
        print(f"\n[{idx+1}/8] {antibiotic}")

        label_mask = ~np.isnan(y_train[:, idx])
        n_labeled = label_mask.sum()

        if n_labeled < 50:
            continue

        X_ab = X_train[label_mask]
        y_ab = y_train[label_mask, idx]
        species_ab = species_train[label_mask]

        if len(np.unique(y_ab)) < 2:
            oof_preds[label_mask, idx] = y_ab[0]
            if test_preds is not None:
                test_preds[:, idx] = y_ab[0]
            continue

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_preds = np.zeros(n_labeled)
        fold_test_preds = []

        for train_idx, val_idx in skf.split(X_ab, species_ab):
            model = xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0,
            )

            weights = compute_sample_weights(species_ab[train_idx])
            model.fit(X_ab[train_idx], y_ab[train_idx], sample_weight=weights)

            fold_preds[val_idx] = model.predict_proba(X_ab[val_idx])[:, 1]

            if X_test is not None:
                fold_test_preds.append(model.predict_proba(X_test)[:, 1])

        oof_preds[label_mask, idx] = fold_preds

        if test_preds is not None and fold_test_preds:
            test_preds[:, idx] = np.mean(fold_test_preds, axis=0)

    # Apply intrinsic rules
    oof_preds = apply_intrinsic_rules(oof_preds, species_train)
    if test_preds is not None:
        test_preds = apply_intrinsic_rules(test_preds, species_test)

    metrics = compute_metrics(y_train, oof_preds, species_train)
    print_metrics("XGBoost", metrics)

    return oof_preds, test_preds, metrics


# =============================================================================
# EXPERIMENT 6: Target Encoding
# =============================================================================

def experiment_target_encoding(X_train: np.ndarray, y_train: np.ndarray,
                               species_train: np.ndarray,
                               X_test: np.ndarray = None,
                               species_test: np.ndarray = None) -> Tuple[np.ndarray, Dict]:
    """
    Add target-encoded species features.

    For each antibiotic, encode species as mean resistance rate.
    Use leave-one-out to prevent leakage.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 6: Target Encoding + LightGBM")
    print("="*60)

    oof_preds = np.full_like(y_train, np.nan, dtype=float)
    test_preds = np.zeros((len(X_test), len(ANTIBIOTICS))) if X_test is not None else None

    for idx, antibiotic in enumerate(ANTIBIOTICS):
        print(f"\n[{idx+1}/8] {antibiotic}")

        label_mask = ~np.isnan(y_train[:, idx])
        n_labeled = label_mask.sum()

        if n_labeled < 50:
            continue

        X_ab = X_train[label_mask]
        y_ab = y_train[label_mask, idx]
        species_ab = species_train[label_mask]

        if len(np.unique(y_ab)) < 2:
            oof_preds[label_mask, idx] = y_ab[0]
            if test_preds is not None:
                test_preds[:, idx] = y_ab[0]
            continue

        # Compute global target encoding for test set
        global_encoding = {}
        for sp in range(4):
            sp_mask = (species_ab == sp)
            if sp_mask.sum() > 0:
                global_encoding[sp] = y_ab[sp_mask].mean()
            else:
                global_encoding[sp] = y_ab.mean()

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_preds = np.zeros(n_labeled)
        fold_test_preds = []

        for train_idx, val_idx in skf.split(X_ab, species_ab):
            # Leave-one-out target encoding for training
            X_train_fold = X_ab[train_idx]
            y_train_fold = y_ab[train_idx]
            species_train_fold = species_ab[train_idx]

            X_val_fold = X_ab[val_idx]
            species_val_fold = species_ab[val_idx]

            # Compute encoding from training fold
            fold_encoding = {}
            for sp in range(4):
                sp_mask = (species_train_fold == sp)
                if sp_mask.sum() > 0:
                    fold_encoding[sp] = y_train_fold[sp_mask].mean()
                else:
                    fold_encoding[sp] = y_train_fold.mean()

            # Add target-encoded feature
            te_train = np.array([fold_encoding[s] for s in species_train_fold]).reshape(-1, 1)
            te_val = np.array([fold_encoding[s] for s in species_val_fold]).reshape(-1, 1)

            X_train_aug = np.hstack([X_train_fold, te_train])
            X_val_aug = np.hstack([X_val_fold, te_val])

            model = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                verbose=-1,
            )

            weights = compute_sample_weights(species_train_fold)
            model.fit(X_train_aug, y_train_fold, sample_weight=weights)

            fold_preds[val_idx] = model.predict_proba(X_val_aug)[:, 1]

            if X_test is not None:
                te_test = np.array([global_encoding.get(s, y_ab.mean()) for s in species_test]).reshape(-1, 1)
                X_test_aug = np.hstack([X_test, te_test])
                fold_test_preds.append(model.predict_proba(X_test_aug)[:, 1])

        oof_preds[label_mask, idx] = fold_preds

        if test_preds is not None and fold_test_preds:
            test_preds[:, idx] = np.mean(fold_test_preds, axis=0)

    # Apply intrinsic rules
    oof_preds = apply_intrinsic_rules(oof_preds, species_train)
    if test_preds is not None:
        test_preds = apply_intrinsic_rules(test_preds, species_test)

    metrics = compute_metrics(y_train, oof_preds, species_train)
    print_metrics("Target Encoding + LightGBM", metrics)

    return oof_preds, test_preds, metrics


# =============================================================================
# EXPERIMENT 7: Multi-Output Exploiting Correlations
# =============================================================================

def experiment_multioutput(X_train: np.ndarray, y_train: np.ndarray,
                           species_train: np.ndarray,
                           X_test: np.ndarray = None,
                           species_test: np.ndarray = None) -> Tuple[np.ndarray, Dict]:
    """
    Multi-output model that predicts all antibiotics together.

    From EDA: Levo/Cipro correlation 0.925, Erta/Cefo 0.813
    Shared representation should help.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 7: Multi-Output Neural Network")
    print("="*60)

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("PyTorch not available. Skipping...")
        return None, None, {}

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None

    # PCA to reduce dimensions
    pca = PCA(n_components=200)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled) if X_test_scaled is not None else None

    # Add species as one-hot
    species_onehot_train = np.zeros((len(species_train), 4))
    species_onehot_train[np.arange(len(species_train)), species_train] = 1
    X_train_aug = np.hstack([X_train_pca, species_onehot_train])

    if X_test is not None:
        species_onehot_test = np.zeros((len(species_test), 4))
        species_onehot_test[np.arange(len(species_test)), species_test] = 1
        X_test_aug = np.hstack([X_test_pca, species_onehot_test])

    # Define model
    class MultiOutputMLP(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, output_dim),
            )

        def forward(self, x):
            return self.net(x)

    oof_preds = np.full_like(y_train, np.nan, dtype=float)
    test_preds_list = []

    # CV training
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_aug, species_train)):
        print(f"\n  Fold {fold_idx + 1}/{N_FOLDS}")

        X_fold_train = torch.FloatTensor(X_train_aug[train_idx])
        X_fold_val = torch.FloatTensor(X_train_aug[val_idx])
        y_fold_train = torch.FloatTensor(y_train[train_idx])
        y_fold_val = torch.FloatTensor(y_train[val_idx])

        # Compute sample weights
        weights = compute_sample_weights(species_train[train_idx])
        weights_tensor = torch.FloatTensor(weights)

        model = MultiOutputMLP(X_train_aug.shape[1], len(ANTIBIOTICS))
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        train_dataset = TensorDataset(X_fold_train, y_fold_train, weights_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(100):
            model.train()
            for X_batch, y_batch, w_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)

                # Masked BCE loss
                mask = ~torch.isnan(y_batch)
                if mask.sum() > 0:
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        outputs[mask], y_batch[mask],
                        weight=w_batch.unsqueeze(1).expand_as(y_batch)[mask]
                    )
                    loss.backward()
                    optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_fold_val)
                val_mask = ~torch.isnan(y_fold_val)
                if val_mask.sum() > 0:
                    val_loss = nn.functional.binary_cross_entropy_with_logits(
                        val_outputs[val_mask], y_fold_val[val_mask]
                    )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_model_state = model.state_dict().copy()
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            break

        # Load best model
        model.load_state_dict(best_model_state)
        model.eval()

        with torch.no_grad():
            val_preds = torch.sigmoid(model(X_fold_val)).numpy()
            oof_preds[val_idx] = val_preds

            if X_test_aug is not None:
                test_preds_fold = torch.sigmoid(model(torch.FloatTensor(X_test_aug))).numpy()
                test_preds_list.append(test_preds_fold)

    test_preds = np.mean(test_preds_list, axis=0) if test_preds_list else None

    # Apply intrinsic rules
    oof_preds = apply_intrinsic_rules(oof_preds, species_train)
    if test_preds is not None:
        test_preds = apply_intrinsic_rules(test_preds, species_test)

    metrics = compute_metrics(y_train, oof_preds, species_train)
    print_metrics("Multi-Output MLP", metrics)

    return oof_preds, test_preds, metrics


# =============================================================================
# EXPERIMENT 8: Calibrated Classifiers
# =============================================================================

def experiment_calibration(X_train: np.ndarray, y_train: np.ndarray,
                           species_train: np.ndarray,
                           X_test: np.ndarray = None,
                           species_test: np.ndarray = None) -> Tuple[np.ndarray, Dict]:
    """
    Calibrate LightGBM predictions using isotonic regression.

    Better calibrated probabilities may improve AUC.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 8: Calibrated LightGBM")
    print("="*60)

    from sklearn.calibration import CalibratedClassifierCV

    oof_preds = np.full_like(y_train, np.nan, dtype=float)
    test_preds = np.zeros((len(X_test), len(ANTIBIOTICS))) if X_test is not None else None

    for idx, antibiotic in enumerate(ANTIBIOTICS):
        print(f"\n[{idx+1}/8] {antibiotic}")

        label_mask = ~np.isnan(y_train[:, idx])
        n_labeled = label_mask.sum()

        if n_labeled < 50:
            continue

        X_ab = X_train[label_mask]
        y_ab = y_train[label_mask, idx]
        species_ab = species_train[label_mask]

        if len(np.unique(y_ab)) < 2:
            oof_preds[label_mask, idx] = y_ab[0]
            if test_preds is not None:
                test_preds[:, idx] = y_ab[0]
            continue

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_preds = np.zeros(n_labeled)
        fold_test_preds = []

        for train_idx, val_idx in skf.split(X_ab, species_ab):
            base_model = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                verbose=-1,
            )

            # Calibrate with isotonic regression
            calibrated = CalibratedClassifierCV(base_model, method='isotonic', cv=3)

            weights = compute_sample_weights(species_ab[train_idx])
            calibrated.fit(X_ab[train_idx], y_ab[train_idx], sample_weight=weights)

            fold_preds[val_idx] = calibrated.predict_proba(X_ab[val_idx])[:, 1]

            if X_test is not None:
                fold_test_preds.append(calibrated.predict_proba(X_test)[:, 1])

        oof_preds[label_mask, idx] = fold_preds

        if test_preds is not None and fold_test_preds:
            test_preds[:, idx] = np.mean(fold_test_preds, axis=0)

    # Apply intrinsic rules
    oof_preds = apply_intrinsic_rules(oof_preds, species_train)
    if test_preds is not None:
        test_preds = apply_intrinsic_rules(test_preds, species_test)

    metrics = compute_metrics(y_train, oof_preds, species_train)
    print_metrics("Calibrated LightGBM", metrics)

    return oof_preds, test_preds, metrics


# =============================================================================
# EXPERIMENT 9: Adversarial Validation Feature Selection
# =============================================================================

def experiment_adversarial_validation(X_train: np.ndarray, y_train: np.ndarray,
                                       species_train: np.ndarray,
                                       X_test: np.ndarray = None,
                                       species_test: np.ndarray = None) -> Tuple[np.ndarray, Dict]:
    """
    Use adversarial validation to find features that differ between train/test.
    Remove or down-weight these features.

    From course: Domain classifier for distribution shift detection.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 9: Adversarial Validation Feature Selection")
    print("="*60)

    # Create domain classification dataset
    X_combined = np.vstack([X_train, X_test])
    y_domain = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])

    # Train domain classifier
    domain_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    domain_model.fit(X_combined, y_domain)

    # Get feature importance for domain prediction
    domain_importance = domain_model.feature_importances_

    # Remove top 10% most domain-discriminating features
    threshold = np.percentile(domain_importance, 90)
    keep_mask = domain_importance < threshold
    n_removed = (~keep_mask).sum()

    print(f"  Removing {n_removed} domain-discriminating features")
    print(f"  Keeping {keep_mask.sum()} features")

    X_train_filtered = X_train[:, keep_mask]
    X_test_filtered = X_test[:, keep_mask] if X_test is not None else None

    # Now train with filtered features
    oof_preds = np.full_like(y_train, np.nan, dtype=float)
    test_preds = np.zeros((len(X_test), len(ANTIBIOTICS))) if X_test is not None else None

    for idx, antibiotic in enumerate(ANTIBIOTICS):
        print(f"\n[{idx+1}/8] {antibiotic}")

        label_mask = ~np.isnan(y_train[:, idx])
        n_labeled = label_mask.sum()

        if n_labeled < 50:
            continue

        X_ab = X_train_filtered[label_mask]
        y_ab = y_train[label_mask, idx]
        species_ab = species_train[label_mask]

        if len(np.unique(y_ab)) < 2:
            oof_preds[label_mask, idx] = y_ab[0]
            if test_preds is not None:
                test_preds[:, idx] = y_ab[0]
            continue

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_preds = np.zeros(n_labeled)
        fold_test_preds = []

        for train_idx, val_idx in skf.split(X_ab, species_ab):
            model = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                verbose=-1,
            )

            weights = compute_sample_weights(species_ab[train_idx])
            model.fit(X_ab[train_idx], y_ab[train_idx], sample_weight=weights)

            fold_preds[val_idx] = model.predict_proba(X_ab[val_idx])[:, 1]

            if X_test_filtered is not None:
                fold_test_preds.append(model.predict_proba(X_test_filtered)[:, 1])

        oof_preds[label_mask, idx] = fold_preds

        if test_preds is not None and fold_test_preds:
            test_preds[:, idx] = np.mean(fold_test_preds, axis=0)

    # Apply intrinsic rules
    oof_preds = apply_intrinsic_rules(oof_preds, species_train)
    if test_preds is not None:
        test_preds = apply_intrinsic_rules(test_preds, species_test)

    metrics = compute_metrics(y_train, oof_preds, species_train)
    print_metrics("Adversarial Feature Selection", metrics)

    return oof_preds, test_preds, metrics


# =============================================================================
# EXPERIMENT 10: Feature Engineering (Clusters, Ratios)
# =============================================================================

def experiment_feature_engineering(X_train: np.ndarray, y_train: np.ndarray,
                                   species_train: np.ndarray,
                                   X_test: np.ndarray = None,
                                   species_test: np.ndarray = None) -> Tuple[np.ndarray, Dict]:
    """
    Add engineered features:
    1. K-Means cluster assignments
    2. PCA components
    3. Statistics (row sum, max, std, non-zero count)
    4. Species one-hot

    From course: Clustering for semi-supervised, PCA for dimensionality reduction.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 10: Feature Engineering")
    print("="*60)

    # Scale features first
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None

    # 1. PCA components (top 50)
    print("  Adding PCA features...")
    pca = PCA(n_components=50)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled) if X_test_scaled is not None else None

    # 2. K-Means cluster assignments (soft and hard)
    print("  Adding cluster features...")
    kmeans = KMeans(n_clusters=8, random_state=RANDOM_STATE, n_init=10)
    cluster_train = kmeans.fit_predict(X_train_pca)
    cluster_test = kmeans.predict(X_test_pca) if X_test_pca is not None else None

    # One-hot encode clusters
    cluster_onehot_train = np.zeros((len(X_train), 8))
    cluster_onehot_train[np.arange(len(X_train)), cluster_train] = 1

    if X_test is not None:
        cluster_onehot_test = np.zeros((len(X_test), 8))
        cluster_onehot_test[np.arange(len(X_test)), cluster_test] = 1

    # 3. Statistics features
    print("  Adding statistical features...")
    stats_train = np.column_stack([
        X_train.sum(axis=1),           # row sum
        X_train.max(axis=1),           # row max
        X_train.std(axis=1),           # row std
        (X_train > 0).sum(axis=1),     # non-zero count
        np.percentile(X_train, 75, axis=1),  # 75th percentile
    ])

    if X_test is not None:
        stats_test = np.column_stack([
            X_test.sum(axis=1),
            X_test.max(axis=1),
            X_test.std(axis=1),
            (X_test > 0).sum(axis=1),
            np.percentile(X_test, 75, axis=1),
        ])

    # 4. Species one-hot
    print("  Adding species features...")
    species_onehot_train = np.zeros((len(species_train), 4))
    species_onehot_train[np.arange(len(species_train)), species_train] = 1

    if X_test is not None:
        species_onehot_test = np.zeros((len(species_test), 4))
        species_onehot_test[np.arange(len(species_test)), species_test] = 1

    # Combine all features
    X_train_eng = np.hstack([
        X_train,           # Original features
        X_train_pca,       # PCA
        cluster_onehot_train,  # Clusters
        stats_train,       # Statistics
        species_onehot_train,  # Species
    ])

    if X_test is not None:
        X_test_eng = np.hstack([
            X_test,
            X_test_pca,
            cluster_onehot_test,
            stats_test,
            species_onehot_test,
        ])
    else:
        X_test_eng = None

    print(f"  Original features: {X_train.shape[1]}")
    print(f"  Engineered features: {X_train_eng.shape[1]}")

    # Now train with engineered features
    oof_preds = np.full_like(y_train, np.nan, dtype=float)
    test_preds = np.zeros((len(X_test), len(ANTIBIOTICS))) if X_test is not None else None

    for idx, antibiotic in enumerate(ANTIBIOTICS):
        print(f"\n[{idx+1}/8] {antibiotic}")

        label_mask = ~np.isnan(y_train[:, idx])
        n_labeled = label_mask.sum()

        if n_labeled < 50:
            continue

        X_ab = X_train_eng[label_mask]
        y_ab = y_train[label_mask, idx]
        species_ab = species_train[label_mask]

        if len(np.unique(y_ab)) < 2:
            oof_preds[label_mask, idx] = y_ab[0]
            if test_preds is not None:
                test_preds[:, idx] = y_ab[0]
            continue

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_preds = np.zeros(n_labeled)
        fold_test_preds = []

        for train_idx, val_idx in skf.split(X_ab, species_ab):
            model = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                verbose=-1,
            )

            weights = compute_sample_weights(species_ab[train_idx])
            model.fit(X_ab[train_idx], y_ab[train_idx], sample_weight=weights)

            fold_preds[val_idx] = model.predict_proba(X_ab[val_idx])[:, 1]

            if X_test_eng is not None:
                fold_test_preds.append(model.predict_proba(X_test_eng)[:, 1])

        oof_preds[label_mask, idx] = fold_preds

        if test_preds is not None and fold_test_preds:
            test_preds[:, idx] = np.mean(fold_test_preds, axis=0)

    # Apply intrinsic rules
    oof_preds = apply_intrinsic_rules(oof_preds, species_train)
    if test_preds is not None:
        test_preds = apply_intrinsic_rules(test_preds, species_test)

    metrics = compute_metrics(y_train, oof_preds, species_train)
    print_metrics("Feature Engineering", metrics)

    return oof_preds, test_preds, metrics


# =============================================================================
# EXPERIMENT 11: Blending Top Models
# =============================================================================

def experiment_blending(results: Dict[str, Tuple]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Blend predictions from multiple models.
    Use weighted average based on K.pneumoniae AUC (since it's 51% of test).
    """
    print("\n" + "="*60)
    print("EXPERIMENT 11: Blending Top Models")
    print("="*60)

    # Filter to valid results with K.pneumoniae metrics
    valid_results = {}
    for name, (oof, test_pred, metrics) in results.items():
        if oof is not None and 'K.pneumoniae_auc' in metrics:
            valid_results[name] = (oof, test_pred, metrics)

    if len(valid_results) < 2:
        print("  Not enough valid models to blend")
        return None, None, {}

    # Rank by K.pneumoniae AUC (most important for test)
    ranked = sorted(valid_results.items(),
                    key=lambda x: x[1][2].get('K.pneumoniae_auc', 0),
                    reverse=True)

    print("\nModel ranking by K.pneumoniae AUC:")
    for name, (_, _, metrics) in ranked:
        print(f"  {name:40} K.pn: {metrics.get('K.pneumoniae_auc', 0):.4f}  Mean: {metrics.get('mean_auc', 0):.4f}")

    # Take top 5 models for blending
    top_models = ranked[:5]

    # Weight by K.pneumoniae AUC
    weights = []
    for name, (_, _, metrics) in top_models:
        w = metrics.get('K.pneumoniae_auc', 0)
        weights.append(w)

    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize

    print(f"\nBlending weights:")
    for (name, _), w in zip(top_models, weights):
        print(f"  {name:40} {w:.3f}")

    # Blend OOF predictions
    oof_blend = np.zeros_like(top_models[0][1][0])
    test_blend = np.zeros_like(top_models[0][1][1]) if top_models[0][1][1] is not None else None

    for (name, (oof, test_pred, _)), w in zip(top_models, weights):
        # Handle NaN in oof
        valid_mask = ~np.isnan(oof)
        oof_contrib = np.where(valid_mask, oof * w, 0)
        oof_blend += oof_contrib

        if test_blend is not None and test_pred is not None:
            test_blend += test_pred * w

    return oof_blend, test_blend, None  # Will compute metrics later


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_all_experiments():
    """Run all experiments and compare results."""

    print("\n" + "="*80)
    print("AMR Prediction: Comprehensive Experimentation")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print("\n[Loading Data]")
    X_train, X_test, y_train, species_train, species_test = load_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Labels: {y_train.shape}")

    # Remove constant features
    X_train_filtered, X_test_filtered, feature_mask = remove_constant_features(X_train, X_test)
    print(f"After filtering: Train {X_train_filtered.shape}, Test {X_test_filtered.shape}")

    # Store results
    results = {}

    # Run experiments
    experiments = [
        ("Species-Specific", experiment_species_specific),
        ("PLS-Features", experiment_pls_features),
        ("Stacked-Ensemble", experiment_stacked_ensemble),
        ("Pseudo-Labeling", experiment_pseudo_labeling),
        ("XGBoost", experiment_xgboost),
        ("Target-Encoding", experiment_target_encoding),
        ("Multi-Output-MLP", experiment_multioutput),
        ("Calibrated", experiment_calibration),
        ("Adversarial-FS", experiment_adversarial_validation),
        ("Feature-Engineering", experiment_feature_engineering),
    ]

    for name, experiment_fn in experiments:
        try:
            oof, test_pred, metrics = experiment_fn(
                X_train_filtered, y_train, species_train,
                X_test_filtered, species_test
            )
            results[name] = (oof, test_pred, metrics)
            EXPERIMENT_RESULTS[name] = metrics
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            import traceback
            traceback.print_exc()

    # Blend top models
    if len(results) >= 2:
        oof_blend, test_blend, _ = experiment_blending(results)
        if oof_blend is not None:
            blend_metrics = compute_metrics(y_train, oof_blend, species_train)
            results["Blended"] = (oof_blend, test_blend, blend_metrics)
            EXPERIMENT_RESULTS["Blended"] = blend_metrics
            print_metrics("Blended", blend_metrics)

    # Summary
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)

    # Create summary table
    summary_data = []
    for name, metrics in EXPERIMENT_RESULTS.items():
        summary_data.append({
            'Method': name,
            'Mean AUC': metrics.get('mean_auc', 0),
            'K.pneumoniae': metrics.get('K.pneumoniae_auc', 0),
            'E.coli': metrics.get('E.coli_auc', 0),
            'P.mirabilis': metrics.get('P.mirabilis_auc', 0),
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('K.pneumoniae', ascending=False)

    print("\nRanked by K.pneumoniae AUC (51% of test!):")
    print("-" * 80)
    print(f"{'Method':<30} {'Mean AUC':>10} {'K.pneumoniae':>12} {'E.coli':>10} {'P.mirabilis':>12}")
    print("-" * 80)

    for _, row in summary_df.iterrows():
        print(f"{row['Method']:<30} {row['Mean AUC']:>10.4f} {row['K.pneumoniae']:>12.4f} {row['E.coli']:>10.4f} {row['P.mirabilis']:>12.4f}")

    # Save results
    results_path = OUTPUT_DIR / f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(results_path, 'w') as f:
        json.dump(EXPERIMENT_RESULTS, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Identify best method
    best_method = summary_df.iloc[0]['Method']
    best_kpn_auc = summary_df.iloc[0]['K.pneumoniae']

    print(f"\n{'='*80}")
    print(f"BEST METHOD: {best_method}")
    print(f"K.pneumoniae AUC: {best_kpn_auc:.4f}")
    print(f"{'='*80}")

    # Save test predictions for best method
    if best_method in results:
        _, test_pred_best, _ = results[best_method]
        if test_pred_best is not None:
            # Create submission
            test_df = pd.read_csv(DATA_DIR / "test.csv")
            submission = pd.DataFrame({
                "sample_id": test_df["sample_id"].values,
                **{ab: test_pred_best[:, idx] for idx, ab in enumerate(ANTIBIOTICS)}
            })

            sub_path = OUTPUT_DIR / f"best_submission_{best_method.lower().replace('-', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            submission.to_csv(sub_path, index=False)
            print(f"\nBest submission saved to: {sub_path}")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results, EXPERIMENT_RESULTS


if __name__ == "__main__":
    results, metrics = run_all_experiments()
