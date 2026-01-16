#!/usr/bin/env python3
"""
Fast experimentation framework for AMR prediction.

Tests 10+ modeling approaches efficiently without slow neural network training.
Each experiment outputs OOF predictions and metrics for comparison.
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
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
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

# Flush print immediately
import functools
print = functools.partial(print, flush=True)


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess data."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    feature_cols = [f"maldi_feature_{i}" for i in range(6000)]
    X_train = train_df[feature_cols].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_train = train_df[ANTIBIOTICS].values.astype(np.float32)
    species_train = train_df["species_id"].values.astype(np.int32)
    species_test = test_df["species_id"].values.astype(np.int32)

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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, species: np.ndarray) -> Dict[str, float]:
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

    # Per-species AUC
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
    print(f"\nPer-Species:")
    for species_name in SPECIES_NAMES.values():
        key = f'{species_name}_auc'
        if key in metrics:
            marker = " <-- PRIMARY TARGET" if species_name == "K.pneumoniae" else ""
            print(f"  {species_name:15} {metrics[key]:.4f}{marker}")


def train_lgb_cv(X: np.ndarray, y: np.ndarray, species: np.ndarray,
                 X_test: np.ndarray = None, **lgb_params) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Train LightGBM with CV and return OOF/test predictions."""

    default_params = {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE,
        'verbose': -1,
    }
    default_params.update(lgb_params)

    oof_preds = np.zeros(len(X))
    test_preds_list = []

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for train_idx, val_idx in skf.split(X, species):
        model = lgb.LGBMClassifier(**default_params)
        weights = compute_sample_weights(species[train_idx])
        model.fit(X[train_idx], y[train_idx], sample_weight=weights)
        oof_preds[val_idx] = model.predict_proba(X[val_idx])[:, 1]

        if X_test is not None:
            test_preds_list.append(model.predict_proba(X_test)[:, 1])

    test_preds = np.mean(test_preds_list, axis=0) if test_preds_list else None
    return oof_preds, test_preds


# =============================================================================
# EXPERIMENT 1: Species-Specific Models
# =============================================================================

def experiment_species_specific(X_train, y_train, species_train, X_test, species_test):
    """Train separate LightGBM models for each species."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: Species-Specific Models")
    print("="*60)

    oof_preds = np.full_like(y_train, np.nan, dtype=float)
    test_preds = np.zeros((len(X_test), len(ANTIBIOTICS)))

    for species_id, species_name in SPECIES_NAMES.items():
        print(f"\n  [Species: {species_name}]")
        species_mask = (species_train == species_id)
        n_samples = species_mask.sum()

        if n_samples < 50:
            print(f"    Too few samples ({n_samples}), skipping")
            continue

        X_species = X_train[species_mask]
        y_species = y_train[species_mask]

        for idx, antibiotic in enumerate(ANTIBIOTICS):
            label_mask = ~np.isnan(y_species[:, idx])
            n_labeled = label_mask.sum()

            if n_labeled < 20:
                continue

            X_ab = X_species[label_mask]
            y_ab = y_species[label_mask, idx]

            if len(np.unique(y_ab)) < 2:
                oof_preds[species_mask, idx] = y_ab[0]
                test_preds[species_test == species_id, idx] = y_ab[0]
                continue

            # CV within species
            n_splits = min(5, n_labeled // 10)
            if n_splits < 2:
                n_splits = 2

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
            fold_preds = np.zeros(n_labeled)
            fold_models = []

            for train_idx, val_idx in skf.split(X_ab, y_ab):
                model = lgb.LGBMClassifier(
                    n_estimators=200, learning_rate=0.05, num_leaves=15,
                    min_child_samples=max(5, n_labeled // 50),
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=0.1,
                    random_state=RANDOM_STATE, verbose=-1,
                )
                model.fit(X_ab[train_idx], y_ab[train_idx])
                fold_preds[val_idx] = model.predict_proba(X_ab[val_idx])[:, 1]
                fold_models.append(model)

            species_indices = np.where(species_mask)[0]
            labeled_indices = species_indices[label_mask]
            oof_preds[labeled_indices, idx] = fold_preds

            # Test predictions
            test_mask = (species_test == species_id)
            if test_mask.sum() > 0:
                test_species = X_test[test_mask]
                test_preds[test_mask, idx] = np.mean([m.predict_proba(test_species)[:, 1] for m in fold_models], axis=0)

    oof_preds = apply_intrinsic_rules(oof_preds, species_train)
    test_preds = apply_intrinsic_rules(test_preds, species_test)

    metrics = compute_metrics(y_train, oof_preds, species_train)
    print_metrics("Species-Specific Models", metrics)
    return oof_preds, test_preds, metrics


# =============================================================================
# EXPERIMENT 2: PLS Feature Extraction
# =============================================================================

def experiment_pls_features(X_train, y_train, species_train, X_test, species_test):
    """Use PLS to extract supervised features, then train LightGBM."""
    print("\n" + "="*60)
    print("EXPERIMENT 2: PLS Features + LightGBM")
    print("="*60)

    oof_preds = np.full_like(y_train, np.nan, dtype=float)
    test_preds = np.zeros((len(X_test), len(ANTIBIOTICS)))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    n_components = 100

    for idx, antibiotic in enumerate(ANTIBIOTICS):
        print(f"  [{idx+1}/8] {antibiotic}")

        label_mask = ~np.isnan(y_train[:, idx])
        n_labeled = label_mask.sum()

        if n_labeled < 50:
            continue

        X_ab = X_train_scaled[label_mask]
        y_ab = y_train[label_mask, idx]
        species_ab = species_train[label_mask]

        if len(np.unique(y_ab)) < 2:
            oof_preds[label_mask, idx] = y_ab[0]
            test_preds[:, idx] = y_ab[0]
            continue

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_preds = np.zeros(n_labeled)
        fold_test_preds = []

        for train_idx, val_idx in skf.split(X_ab, species_ab):
            pls = PLSRegression(n_components=min(n_components, len(train_idx) - 1))
            pls.fit(X_ab[train_idx], y_ab[train_idx])

            X_train_pls = pls.transform(X_ab[train_idx])
            X_val_pls = pls.transform(X_ab[val_idx])

            model = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.05, num_leaves=31,
                min_child_samples=20, random_state=RANDOM_STATE, verbose=-1,
            )
            weights = compute_sample_weights(species_ab[train_idx])
            model.fit(X_train_pls, y_ab[train_idx], sample_weight=weights)

            fold_preds[val_idx] = model.predict_proba(X_val_pls)[:, 1]
            fold_test_preds.append(model.predict_proba(pls.transform(X_test_scaled))[:, 1])

        oof_preds[label_mask, idx] = fold_preds
        test_preds[:, idx] = np.mean(fold_test_preds, axis=0)

    oof_preds = apply_intrinsic_rules(oof_preds, species_train)
    test_preds = apply_intrinsic_rules(test_preds, species_test)

    metrics = compute_metrics(y_train, oof_preds, species_train)
    print_metrics("PLS Features + LightGBM", metrics)
    return oof_preds, test_preds, metrics


# =============================================================================
# EXPERIMENT 3: Stacked Ensemble
# =============================================================================

def experiment_stacked_ensemble(X_train, y_train, species_train, X_test, species_test):
    """Two-level stacking with multiple base learners."""
    print("\n" + "="*60)
    print("EXPERIMENT 3: Stacked Ensemble")
    print("="*60)

    oof_preds = np.full_like(y_train, np.nan, dtype=float)
    test_preds = np.zeros((len(X_test), len(ANTIBIOTICS)))

    base_configs = [
        {"n_estimators": 100, "num_leaves": 15, "learning_rate": 0.1},
        {"n_estimators": 200, "num_leaves": 31, "learning_rate": 0.05},
        {"n_estimators": 300, "num_leaves": 63, "learning_rate": 0.03},
        {"n_estimators": 150, "num_leaves": 7, "learning_rate": 0.1, "max_depth": 3},
    ]

    for idx, antibiotic in enumerate(ANTIBIOTICS):
        print(f"  [{idx+1}/8] {antibiotic}")

        label_mask = ~np.isnan(y_train[:, idx])
        n_labeled = label_mask.sum()

        if n_labeled < 50:
            continue

        X_ab = X_train[label_mask]
        y_ab = y_train[label_mask, idx]
        species_ab = species_train[label_mask]

        if len(np.unique(y_ab)) < 2:
            oof_preds[label_mask, idx] = y_ab[0]
            test_preds[:, idx] = y_ab[0]
            continue

        level1_oof = np.zeros((n_labeled, len(base_configs)))
        level1_test = np.zeros((len(X_test), len(base_configs)))

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        for config_idx, config in enumerate(base_configs):
            fold_test_preds = []

            for train_idx, val_idx in skf.split(X_ab, species_ab):
                model = lgb.LGBMClassifier(
                    **config, min_child_samples=20, subsample=0.8,
                    colsample_bytree=0.8, random_state=RANDOM_STATE, verbose=-1,
                )
                weights = compute_sample_weights(species_ab[train_idx])
                model.fit(X_ab[train_idx], y_ab[train_idx], sample_weight=weights)
                level1_oof[val_idx, config_idx] = model.predict_proba(X_ab[val_idx])[:, 1]
                fold_test_preds.append(model.predict_proba(X_test)[:, 1])

            level1_test[:, config_idx] = np.mean(fold_test_preds, axis=0)

        # Meta-learner
        meta = LogisticRegression(C=1.0, max_iter=1000)
        meta.fit(level1_oof, y_ab)

        oof_preds[label_mask, idx] = meta.predict_proba(level1_oof)[:, 1]
        test_preds[:, idx] = meta.predict_proba(level1_test)[:, 1]

    oof_preds = apply_intrinsic_rules(oof_preds, species_train)
    test_preds = apply_intrinsic_rules(test_preds, species_test)

    metrics = compute_metrics(y_train, oof_preds, species_train)
    print_metrics("Stacked Ensemble", metrics)
    return oof_preds, test_preds, metrics


# =============================================================================
# EXPERIMENT 4: Pseudo-labeling
# =============================================================================

def experiment_pseudo_labeling(X_train, y_train, species_train, X_test, species_test):
    """Semi-supervised learning using pseudo-labeling."""
    print("\n" + "="*60)
    print("EXPERIMENT 4: Pseudo-labeling")
    print("="*60)

    CONFIDENCE_THRESHOLD = 0.9
    oof_preds = np.full_like(y_train, np.nan, dtype=float)
    test_preds = np.zeros((len(X_test), len(ANTIBIOTICS)))

    for idx, antibiotic in enumerate(ANTIBIOTICS):
        label_mask = ~np.isnan(y_train[:, idx])
        unlabeled_mask = np.isnan(y_train[:, idx])
        n_labeled = label_mask.sum()
        n_unlabeled = unlabeled_mask.sum()

        print(f"  [{idx+1}/8] {antibiotic}: {n_labeled} labeled, {n_unlabeled} unlabeled")

        if n_labeled < 50:
            continue

        X_labeled = X_train[label_mask]
        y_labeled = y_train[label_mask, idx]
        species_labeled = species_train[label_mask]

        if len(np.unique(y_labeled)) < 2:
            oof_preds[label_mask, idx] = y_labeled[0]
            test_preds[:, idx] = y_labeled[0]
            continue

        # Phase 1: Train initial model
        model = lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.1, num_leaves=31,
            random_state=RANDOM_STATE, verbose=-1,
        )
        weights = compute_sample_weights(species_labeled)
        model.fit(X_labeled, y_labeled, sample_weight=weights)

        # Predict on unlabeled
        if n_unlabeled > 0:
            X_unlabeled = X_train[unlabeled_mask]
            unlabeled_probs = model.predict_proba(X_unlabeled)[:, 1]
            high_conf_mask = (unlabeled_probs > CONFIDENCE_THRESHOLD) | (unlabeled_probs < (1 - CONFIDENCE_THRESHOLD))
            n_pseudo = high_conf_mask.sum()

            if n_pseudo > 0:
                pseudo_labels = (unlabeled_probs[high_conf_mask] > 0.5).astype(float)
                X_augmented = np.vstack([X_labeled, X_unlabeled[high_conf_mask]])
                y_augmented = np.concatenate([y_labeled, pseudo_labels])
                species_augmented = np.concatenate([species_labeled, species_train[unlabeled_mask][high_conf_mask]])
                weights_aug = compute_sample_weights(species_augmented)
                weights_aug[len(y_labeled):] *= 0.5  # Down-weight pseudo-labels

                # Retrain
                model = lgb.LGBMClassifier(
                    n_estimators=150, learning_rate=0.1, num_leaves=31,
                    random_state=RANDOM_STATE, verbose=-1,
                )
                model.fit(X_augmented, y_augmented, sample_weight=weights_aug)

        # CV for OOF predictions
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_preds = np.zeros(n_labeled)
        fold_test_preds = []

        for train_idx, val_idx in skf.split(X_labeled, species_labeled):
            model_cv = lgb.LGBMClassifier(
                n_estimators=150, learning_rate=0.1, num_leaves=31,
                random_state=RANDOM_STATE, verbose=-1,
            )
            weights_cv = compute_sample_weights(species_labeled[train_idx])
            model_cv.fit(X_labeled[train_idx], y_labeled[train_idx], sample_weight=weights_cv)
            fold_preds[val_idx] = model_cv.predict_proba(X_labeled[val_idx])[:, 1]
            fold_test_preds.append(model_cv.predict_proba(X_test)[:, 1])

        oof_preds[label_mask, idx] = fold_preds
        test_preds[:, idx] = np.mean(fold_test_preds, axis=0)

    oof_preds = apply_intrinsic_rules(oof_preds, species_train)
    test_preds = apply_intrinsic_rules(test_preds, species_test)

    metrics = compute_metrics(y_train, oof_preds, species_train)
    print_metrics("Pseudo-labeling", metrics)
    return oof_preds, test_preds, metrics


# =============================================================================
# EXPERIMENT 5: XGBoost
# =============================================================================

def experiment_xgboost(X_train, y_train, species_train, X_test, species_test):
    """XGBoost as alternative to LightGBM."""
    print("\n" + "="*60)
    print("EXPERIMENT 5: XGBoost")
    print("="*60)

    try:
        import xgboost as xgb
    except ImportError:
        print("  XGBoost not installed. Skipping...")
        return None, None, {}

    oof_preds = np.full_like(y_train, np.nan, dtype=float)
    test_preds = np.zeros((len(X_test), len(ANTIBIOTICS)))

    for idx, antibiotic in enumerate(ANTIBIOTICS):
        print(f"  [{idx+1}/8] {antibiotic}")

        label_mask = ~np.isnan(y_train[:, idx])
        n_labeled = label_mask.sum()

        if n_labeled < 50:
            continue

        X_ab = X_train[label_mask]
        y_ab = y_train[label_mask, idx]
        species_ab = species_train[label_mask]

        if len(np.unique(y_ab)) < 2:
            oof_preds[label_mask, idx] = y_ab[0]
            test_preds[:, idx] = y_ab[0]
            continue

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_preds = np.zeros(n_labeled)
        fold_test_preds = []

        for train_idx, val_idx in skf.split(X_ab, species_ab):
            model = xgb.XGBClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=6,
                min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_STATE,
                use_label_encoder=False, eval_metric='logloss', verbosity=0,
            )
            weights = compute_sample_weights(species_ab[train_idx])
            model.fit(X_ab[train_idx], y_ab[train_idx], sample_weight=weights)
            fold_preds[val_idx] = model.predict_proba(X_ab[val_idx])[:, 1]
            fold_test_preds.append(model.predict_proba(X_test)[:, 1])

        oof_preds[label_mask, idx] = fold_preds
        test_preds[:, idx] = np.mean(fold_test_preds, axis=0)

    oof_preds = apply_intrinsic_rules(oof_preds, species_train)
    test_preds = apply_intrinsic_rules(test_preds, species_test)

    metrics = compute_metrics(y_train, oof_preds, species_train)
    print_metrics("XGBoost", metrics)
    return oof_preds, test_preds, metrics


# =============================================================================
# EXPERIMENT 6: CatBoost
# =============================================================================

def experiment_catboost(X_train, y_train, species_train, X_test, species_test):
    """CatBoost with automatic handling of categorical features."""
    print("\n" + "="*60)
    print("EXPERIMENT 6: CatBoost")
    print("="*60)

    try:
        from catboost import CatBoostClassifier
    except ImportError:
        print("  CatBoost not installed. Skipping...")
        return None, None, {}

    # Add species as categorical feature
    X_train_aug = np.hstack([X_train, species_train.reshape(-1, 1)])
    X_test_aug = np.hstack([X_test, species_test.reshape(-1, 1)])
    cat_features = [X_train_aug.shape[1] - 1]  # Last column is species

    oof_preds = np.full_like(y_train, np.nan, dtype=float)
    test_preds = np.zeros((len(X_test), len(ANTIBIOTICS)))

    for idx, antibiotic in enumerate(ANTIBIOTICS):
        print(f"  [{idx+1}/8] {antibiotic}")

        label_mask = ~np.isnan(y_train[:, idx])
        n_labeled = label_mask.sum()

        if n_labeled < 50:
            continue

        X_ab = X_train_aug[label_mask]
        y_ab = y_train[label_mask, idx]
        species_ab = species_train[label_mask]

        if len(np.unique(y_ab)) < 2:
            oof_preds[label_mask, idx] = y_ab[0]
            test_preds[:, idx] = y_ab[0]
            continue

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_preds = np.zeros(n_labeled)
        fold_test_preds = []

        for train_idx, val_idx in skf.split(X_ab, species_ab):
            model = CatBoostClassifier(
                iterations=200, learning_rate=0.05, depth=6,
                l2_leaf_reg=3, random_seed=RANDOM_STATE, verbose=False,
            )
            weights = compute_sample_weights(species_ab[train_idx])
            model.fit(X_ab[train_idx], y_ab[train_idx],
                      sample_weight=weights, cat_features=cat_features)
            fold_preds[val_idx] = model.predict_proba(X_ab[val_idx])[:, 1]
            fold_test_preds.append(model.predict_proba(X_test_aug)[:, 1])

        oof_preds[label_mask, idx] = fold_preds
        test_preds[:, idx] = np.mean(fold_test_preds, axis=0)

    oof_preds = apply_intrinsic_rules(oof_preds, species_train)
    test_preds = apply_intrinsic_rules(test_preds, species_test)

    metrics = compute_metrics(y_train, oof_preds, species_train)
    print_metrics("CatBoost", metrics)
    return oof_preds, test_preds, metrics


# =============================================================================
# EXPERIMENT 7: Target Encoding
# =============================================================================

def experiment_target_encoding(X_train, y_train, species_train, X_test, species_test):
    """Add target-encoded species features."""
    print("\n" + "="*60)
    print("EXPERIMENT 7: Target Encoding + LightGBM")
    print("="*60)

    oof_preds = np.full_like(y_train, np.nan, dtype=float)
    test_preds = np.zeros((len(X_test), len(ANTIBIOTICS)))

    for idx, antibiotic in enumerate(ANTIBIOTICS):
        print(f"  [{idx+1}/8] {antibiotic}")

        label_mask = ~np.isnan(y_train[:, idx])
        n_labeled = label_mask.sum()

        if n_labeled < 50:
            continue

        X_ab = X_train[label_mask]
        y_ab = y_train[label_mask, idx]
        species_ab = species_train[label_mask]

        if len(np.unique(y_ab)) < 2:
            oof_preds[label_mask, idx] = y_ab[0]
            test_preds[:, idx] = y_ab[0]
            continue

        # Global encoding for test
        global_encoding = {}
        for sp in range(4):
            sp_mask = (species_ab == sp)
            global_encoding[sp] = y_ab[sp_mask].mean() if sp_mask.sum() > 0 else y_ab.mean()

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_preds = np.zeros(n_labeled)
        fold_test_preds = []

        for train_idx, val_idx in skf.split(X_ab, species_ab):
            # Leave-one-out encoding
            fold_encoding = {}
            for sp in range(4):
                sp_mask = (species_ab[train_idx] == sp)
                fold_encoding[sp] = y_ab[train_idx][sp_mask].mean() if sp_mask.sum() > 0 else y_ab[train_idx].mean()

            te_train = np.array([fold_encoding[s] for s in species_ab[train_idx]]).reshape(-1, 1)
            te_val = np.array([fold_encoding[s] for s in species_ab[val_idx]]).reshape(-1, 1)
            te_test = np.array([global_encoding.get(s, y_ab.mean()) for s in species_test]).reshape(-1, 1)

            X_train_aug = np.hstack([X_ab[train_idx], te_train])
            X_val_aug = np.hstack([X_ab[val_idx], te_val])
            X_test_aug = np.hstack([X_test, te_test])

            model = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.05, num_leaves=31,
                min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                random_state=RANDOM_STATE, verbose=-1,
            )
            weights = compute_sample_weights(species_ab[train_idx])
            model.fit(X_train_aug, y_ab[train_idx], sample_weight=weights)
            fold_preds[val_idx] = model.predict_proba(X_val_aug)[:, 1]
            fold_test_preds.append(model.predict_proba(X_test_aug)[:, 1])

        oof_preds[label_mask, idx] = fold_preds
        test_preds[:, idx] = np.mean(fold_test_preds, axis=0)

    oof_preds = apply_intrinsic_rules(oof_preds, species_train)
    test_preds = apply_intrinsic_rules(test_preds, species_test)

    metrics = compute_metrics(y_train, oof_preds, species_train)
    print_metrics("Target Encoding", metrics)
    return oof_preds, test_preds, metrics


# =============================================================================
# EXPERIMENT 8: Calibrated LightGBM
# =============================================================================

def experiment_calibration(X_train, y_train, species_train, X_test, species_test):
    """Calibrate LightGBM predictions using isotonic regression."""
    print("\n" + "="*60)
    print("EXPERIMENT 8: Calibrated LightGBM")
    print("="*60)

    oof_preds = np.full_like(y_train, np.nan, dtype=float)
    test_preds = np.zeros((len(X_test), len(ANTIBIOTICS)))

    for idx, antibiotic in enumerate(ANTIBIOTICS):
        print(f"  [{idx+1}/8] {antibiotic}")

        label_mask = ~np.isnan(y_train[:, idx])
        n_labeled = label_mask.sum()

        if n_labeled < 50:
            continue

        X_ab = X_train[label_mask]
        y_ab = y_train[label_mask, idx]
        species_ab = species_train[label_mask]

        if len(np.unique(y_ab)) < 2:
            oof_preds[label_mask, idx] = y_ab[0]
            test_preds[:, idx] = y_ab[0]
            continue

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_preds = np.zeros(n_labeled)
        fold_test_preds = []

        for train_idx, val_idx in skf.split(X_ab, species_ab):
            base_model = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.05, num_leaves=31,
                min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                random_state=RANDOM_STATE, verbose=-1,
            )
            calibrated = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
            weights = compute_sample_weights(species_ab[train_idx])
            calibrated.fit(X_ab[train_idx], y_ab[train_idx], sample_weight=weights)
            fold_preds[val_idx] = calibrated.predict_proba(X_ab[val_idx])[:, 1]
            fold_test_preds.append(calibrated.predict_proba(X_test)[:, 1])

        oof_preds[label_mask, idx] = fold_preds
        test_preds[:, idx] = np.mean(fold_test_preds, axis=0)

    oof_preds = apply_intrinsic_rules(oof_preds, species_train)
    test_preds = apply_intrinsic_rules(test_preds, species_test)

    metrics = compute_metrics(y_train, oof_preds, species_train)
    print_metrics("Calibrated LightGBM", metrics)
    return oof_preds, test_preds, metrics


# =============================================================================
# EXPERIMENT 9: Adversarial Validation Feature Selection
# =============================================================================

def experiment_adversarial_validation(X_train, y_train, species_train, X_test, species_test):
    """Remove features that distinguish train from test."""
    print("\n" + "="*60)
    print("EXPERIMENT 9: Adversarial Validation Feature Selection")
    print("="*60)

    # Train domain classifier
    X_combined = np.vstack([X_train, X_test])
    y_domain = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])

    domain_model = lgb.LGBMClassifier(
        n_estimators=100, learning_rate=0.1, num_leaves=31,
        random_state=RANDOM_STATE, verbose=-1,
    )
    domain_model.fit(X_combined, y_domain)

    # Remove top 10% domain-discriminating features
    importance = domain_model.feature_importances_
    threshold = np.percentile(importance, 90)
    keep_mask = importance < threshold
    print(f"  Removing {(~keep_mask).sum()} domain-discriminating features")

    X_train_filtered = X_train[:, keep_mask]
    X_test_filtered = X_test[:, keep_mask]

    oof_preds = np.full_like(y_train, np.nan, dtype=float)
    test_preds = np.zeros((len(X_test), len(ANTIBIOTICS)))

    for idx, antibiotic in enumerate(ANTIBIOTICS):
        print(f"  [{idx+1}/8] {antibiotic}")

        label_mask = ~np.isnan(y_train[:, idx])
        n_labeled = label_mask.sum()

        if n_labeled < 50:
            continue

        X_ab = X_train_filtered[label_mask]
        y_ab = y_train[label_mask, idx]
        species_ab = species_train[label_mask]

        if len(np.unique(y_ab)) < 2:
            oof_preds[label_mask, idx] = y_ab[0]
            test_preds[:, idx] = y_ab[0]
            continue

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_preds = np.zeros(n_labeled)
        fold_test_preds = []

        for train_idx, val_idx in skf.split(X_ab, species_ab):
            model = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.05, num_leaves=31,
                min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                random_state=RANDOM_STATE, verbose=-1,
            )
            weights = compute_sample_weights(species_ab[train_idx])
            model.fit(X_ab[train_idx], y_ab[train_idx], sample_weight=weights)
            fold_preds[val_idx] = model.predict_proba(X_ab[val_idx])[:, 1]
            fold_test_preds.append(model.predict_proba(X_test_filtered)[:, 1])

        oof_preds[label_mask, idx] = fold_preds
        test_preds[:, idx] = np.mean(fold_test_preds, axis=0)

    oof_preds = apply_intrinsic_rules(oof_preds, species_train)
    test_preds = apply_intrinsic_rules(test_preds, species_test)

    metrics = compute_metrics(y_train, oof_preds, species_train)
    print_metrics("Adversarial Feature Selection", metrics)
    return oof_preds, test_preds, metrics


# =============================================================================
# EXPERIMENT 10: Feature Engineering
# =============================================================================

def experiment_feature_engineering(X_train, y_train, species_train, X_test, species_test):
    """Add engineered features: PCA, clusters, statistics."""
    print("\n" + "="*60)
    print("EXPERIMENT 10: Feature Engineering")
    print("="*60)

    # Scale and PCA
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=50)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # K-Means clusters
    kmeans = KMeans(n_clusters=8, random_state=RANDOM_STATE, n_init=10)
    cluster_train = kmeans.fit_predict(X_train_pca)
    cluster_test = kmeans.predict(X_test_pca)

    cluster_onehot_train = np.zeros((len(X_train), 8))
    cluster_onehot_train[np.arange(len(X_train)), cluster_train] = 1
    cluster_onehot_test = np.zeros((len(X_test), 8))
    cluster_onehot_test[np.arange(len(X_test)), cluster_test] = 1

    # Statistics
    stats_train = np.column_stack([
        X_train.sum(axis=1), X_train.max(axis=1), X_train.std(axis=1),
        (X_train > 0).sum(axis=1), np.percentile(X_train, 75, axis=1),
    ])
    stats_test = np.column_stack([
        X_test.sum(axis=1), X_test.max(axis=1), X_test.std(axis=1),
        (X_test > 0).sum(axis=1), np.percentile(X_test, 75, axis=1),
    ])

    # Species one-hot
    species_onehot_train = np.zeros((len(species_train), 4))
    species_onehot_train[np.arange(len(species_train)), species_train] = 1
    species_onehot_test = np.zeros((len(species_test), 4))
    species_onehot_test[np.arange(len(species_test)), species_test] = 1

    # Combine
    X_train_eng = np.hstack([X_train, X_train_pca, cluster_onehot_train, stats_train, species_onehot_train])
    X_test_eng = np.hstack([X_test, X_test_pca, cluster_onehot_test, stats_test, species_onehot_test])

    print(f"  Original: {X_train.shape[1]}, Engineered: {X_train_eng.shape[1]}")

    oof_preds = np.full_like(y_train, np.nan, dtype=float)
    test_preds = np.zeros((len(X_test), len(ANTIBIOTICS)))

    for idx, antibiotic in enumerate(ANTIBIOTICS):
        print(f"  [{idx+1}/8] {antibiotic}")

        label_mask = ~np.isnan(y_train[:, idx])
        n_labeled = label_mask.sum()

        if n_labeled < 50:
            continue

        X_ab = X_train_eng[label_mask]
        y_ab = y_train[label_mask, idx]
        species_ab = species_train[label_mask]

        if len(np.unique(y_ab)) < 2:
            oof_preds[label_mask, idx] = y_ab[0]
            test_preds[:, idx] = y_ab[0]
            continue

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_preds = np.zeros(n_labeled)
        fold_test_preds = []

        for train_idx, val_idx in skf.split(X_ab, species_ab):
            model = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.05, num_leaves=31,
                min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                random_state=RANDOM_STATE, verbose=-1,
            )
            weights = compute_sample_weights(species_ab[train_idx])
            model.fit(X_ab[train_idx], y_ab[train_idx], sample_weight=weights)
            fold_preds[val_idx] = model.predict_proba(X_ab[val_idx])[:, 1]
            fold_test_preds.append(model.predict_proba(X_test_eng)[:, 1])

        oof_preds[label_mask, idx] = fold_preds
        test_preds[:, idx] = np.mean(fold_test_preds, axis=0)

    oof_preds = apply_intrinsic_rules(oof_preds, species_train)
    test_preds = apply_intrinsic_rules(test_preds, species_test)

    metrics = compute_metrics(y_train, oof_preds, species_train)
    print_metrics("Feature Engineering", metrics)
    return oof_preds, test_preds, metrics


# =============================================================================
# EXPERIMENT 11: Blending
# =============================================================================

def experiment_blending(results, y_train, species_train, species_test):
    """Blend predictions from multiple models."""
    print("\n" + "="*60)
    print("EXPERIMENT 11: Blending Top Models")
    print("="*60)

    valid_results = {k: v for k, v in results.items() if v[0] is not None and 'K.pneumoniae_auc' in v[2]}

    if len(valid_results) < 2:
        print("  Not enough valid models")
        return None, None, {}

    # Rank by K.pneumoniae AUC
    ranked = sorted(valid_results.items(), key=lambda x: x[1][2].get('K.pneumoniae_auc', 0), reverse=True)

    print("\n  Ranking by K.pneumoniae AUC:")
    for name, (_, _, m) in ranked[:5]:
        print(f"    {name:30} {m.get('K.pneumoniae_auc', 0):.4f}")

    # Take top 5
    top_models = ranked[:5]
    weights = np.array([m[2].get('K.pneumoniae_auc', 0) for _, m in top_models])
    weights = weights / weights.sum()

    print(f"\n  Blend weights:")
    for (name, _), w in zip(top_models, weights):
        print(f"    {name:30} {w:.3f}")

    oof_blend = np.zeros_like(top_models[0][1][0])
    test_blend = np.zeros_like(top_models[0][1][1])

    for (name, (oof, test_pred, _)), w in zip(top_models, weights):
        valid_mask = ~np.isnan(oof)
        oof_blend = np.where(valid_mask, oof_blend + oof * w, oof_blend)
        test_blend += test_pred * w

    metrics = compute_metrics(y_train, oof_blend, species_train)
    print_metrics("Blended", metrics)
    return oof_blend, test_blend, metrics


# =============================================================================
# MAIN
# =============================================================================

def run_all_experiments():
    """Run all experiments and compare."""
    print("\n" + "="*80)
    print("AMR Prediction: Fast Experimentation")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print("\n[Loading Data]")
    X_train, X_test, y_train, species_train, species_test = load_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Remove constant features
    X_train, X_test, _ = remove_constant_features(X_train, X_test)
    print(f"After filtering: Train {X_train.shape}")

    results = {}

    experiments = [
        ("Species-Specific", experiment_species_specific),
        ("PLS-Features", experiment_pls_features),
        ("Stacked-Ensemble", experiment_stacked_ensemble),
        ("Pseudo-Labeling", experiment_pseudo_labeling),
        ("XGBoost", experiment_xgboost),
        ("CatBoost", experiment_catboost),
        ("Target-Encoding", experiment_target_encoding),
        ("Calibrated", experiment_calibration),
        ("Adversarial-FS", experiment_adversarial_validation),
        ("Feature-Eng", experiment_feature_engineering),
    ]

    for name, fn in experiments:
        try:
            oof, test_pred, metrics = fn(X_train, y_train, species_train, X_test, species_test)
            results[name] = (oof, test_pred, metrics)
            EXPERIMENT_RESULTS[name] = metrics
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            import traceback
            traceback.print_exc()

    # Blend
    if len(results) >= 2:
        oof_blend, test_blend, blend_metrics = experiment_blending(results, y_train, species_train, species_test)
        if oof_blend is not None:
            results["Blended"] = (oof_blend, test_blend, blend_metrics)
            EXPERIMENT_RESULTS["Blended"] = blend_metrics

    # Summary
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)

    summary = []
    for name, metrics in EXPERIMENT_RESULTS.items():
        summary.append({
            'Method': name,
            'Mean': metrics.get('mean_auc', 0),
            'K.pn': metrics.get('K.pneumoniae_auc', 0),
            'E.coli': metrics.get('E.coli_auc', 0),
            'P.mir': metrics.get('P.mirabilis_auc', 0),
        })

    summary_df = pd.DataFrame(summary).sort_values('K.pn', ascending=False)

    print("\nRanked by K.pneumoniae AUC (51% of test!):")
    print("-" * 80)
    print(f"{'Method':<25} {'Mean':>8} {'K.pn':>8} {'E.coli':>8} {'P.mir':>8}")
    print("-" * 80)
    for _, r in summary_df.iterrows():
        print(f"{r['Method']:<25} {r['Mean']:>8.4f} {r['K.pn']:>8.4f} {r['E.coli']:>8.4f} {r['P.mir']:>8.4f}")

    # Save results
    results_path = OUTPUT_DIR / f"fast_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(results_path, 'w') as f:
        json.dump(EXPERIMENT_RESULTS, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Save best submission
    best_method = summary_df.iloc[0]['Method']
    best_kpn = summary_df.iloc[0]['K.pn']

    print(f"\n{'='*80}")
    print(f"BEST: {best_method} (K.pn AUC: {best_kpn:.4f})")
    print(f"{'='*80}")

    if best_method in results:
        _, test_pred, _ = results[best_method]
        test_df = pd.read_csv(DATA_DIR / "test.csv")
        submission = pd.DataFrame({
            "sample_id": test_df["sample_id"].values,
            **{ab: test_pred[:, idx] for idx, ab in enumerate(ANTIBIOTICS)}
        })
        sub_path = OUTPUT_DIR / f"best_{best_method.lower().replace('-', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        submission.to_csv(sub_path, index=False)
        print(f"Best submission: {sub_path}")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return results, EXPERIMENT_RESULTS


if __name__ == "__main__":
    results, metrics = run_all_experiments()
