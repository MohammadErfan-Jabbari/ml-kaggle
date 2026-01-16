#!/usr/bin/env python3
"""
Round 2: Advanced experimentation for AMR prediction.

Focuses on:
1. Hyperparameter optimization
2. Better blending strategies
3. Optimized species weighting
4. Rank averaging
5. More aggressive feature engineering
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple, Any
from datetime import datetime
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import rankdata
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

EXPERIMENT_RESULTS = {}

import functools
print = functools.partial(print, flush=True)


def load_data():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    feature_cols = [f"maldi_feature_{i}" for i in range(6000)]
    X_train = train_df[feature_cols].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_train = train_df[ANTIBIOTICS].values.astype(np.float32)
    species_train = train_df["species_id"].values.astype(np.int32)
    species_test = test_df["species_id"].values.astype(np.int32)

    return X_train, X_test, y_train, species_train, species_test


def remove_constant_features(X_train, X_test, threshold=1e-5):
    variances = X_train.var(axis=0)
    mask = variances > threshold
    return X_train[:, mask], X_test[:, mask], mask


def apply_intrinsic_rules(predictions, species_ids):
    predictions = predictions.copy()
    ANTIBIOTIC_INDICES = {ab: idx for idx, ab in enumerate(ANTIBIOTICS)}

    pa_mask = (species_ids == 3)
    for ab in ["Ampicillin", "Amoxicillin_Clavulanic_acid", "Ertapenem", "Cefotaxime", "Cefuroxime"]:
        predictions[pa_mask, ANTIBIOTIC_INDICES[ab]] = 1.0

    pm_mask = (species_ids == 2)
    predictions[pm_mask, ANTIBIOTIC_INDICES["Imipenem"]] = 1.0

    return predictions


def compute_metrics(y_true, y_pred, species):
    metrics = {}

    antibiotic_aucs = []
    for idx, antibiotic in enumerate(ANTIBIOTICS):
        mask = ~np.isnan(y_true[:, idx])
        if mask.sum() > 10 and len(np.unique(y_true[mask, idx])) > 1:
            auc = roc_auc_score(y_true[mask, idx], y_pred[mask, idx])
            metrics[antibiotic] = auc
            antibiotic_aucs.append(auc)

    metrics['mean_auc'] = np.mean(antibiotic_aucs) if antibiotic_aucs else 0.0

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


def print_metrics(name, metrics):
    print(f"\n{'='*60}")
    print(f"Results: {name}")
    print(f"{'='*60}")
    print(f"Mean AUC: {metrics.get('mean_auc', 0):.4f}")
    print(f"K.pneumoniae: {metrics.get('K.pneumoniae_auc', 0):.4f}")


# =============================================================================
# EXPERIMENT 1: Optimized Species Weighting
# =============================================================================

def experiment_optimized_weights(X_train, y_train, species_train, X_test, species_test):
    """
    Try different species weight combinations to find optimal.
    Based on test distribution: K.pn 51%, E.coli 27%, P.mir 19%, P.aer 3%
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Optimized Species Weighting")
    print("="*60)

    # Test distribution weights (approximate inverse of train/test ratio)
    # Train: P.aer 43%, K.pn 28%, E.coli 17%, P.mir 12%
    # Test:  P.aer 3%, K.pn 51%, E.coli 27%, P.mir 19%
    weight_configs = [
        # (P.aer, K.pn, E.coli, P.mir)
        {"name": "Baseline", "weights": {3: 0.3, 1: 1.0, 0: 1.0, 2: 1.0}},
        {"name": "Test-Proportional", "weights": {3: 0.07, 1: 1.8, 0: 1.6, 2: 1.5}},
        {"name": "K.pn-Focus", "weights": {3: 0.1, 1: 2.0, 0: 1.0, 2: 1.0}},
        {"name": "Extreme-K.pn", "weights": {3: 0.05, 1: 3.0, 0: 1.5, 2: 1.5}},
        {"name": "Balanced-NonPA", "weights": {3: 0.1, 1: 1.5, 0: 1.5, 2: 1.5}},
    ]

    best_config = None
    best_kpn_auc = 0
    best_oof = None
    best_test = None

    for config in weight_configs:
        print(f"\n  Testing: {config['name']}")
        weights_map = config['weights']

        oof_preds = np.full_like(y_train, np.nan, dtype=float)
        test_preds = np.zeros((len(X_test), len(ANTIBIOTICS)))

        for idx, antibiotic in enumerate(ANTIBIOTICS):
            label_mask = ~np.isnan(y_train[:, idx])
            if label_mask.sum() < 50:
                continue

            X_ab = X_train[label_mask]
            y_ab = y_train[label_mask, idx]
            species_ab = species_train[label_mask]

            if len(np.unique(y_ab)) < 2:
                oof_preds[label_mask, idx] = y_ab[0]
                test_preds[:, idx] = y_ab[0]
                continue

            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            fold_preds = np.zeros(label_mask.sum())
            fold_test_preds = []

            for train_idx, val_idx in skf.split(X_ab, species_ab):
                sample_weights = np.array([weights_map.get(s, 1.0) for s in species_ab[train_idx]])

                model = lgb.LGBMClassifier(
                    n_estimators=200, learning_rate=0.05, num_leaves=31,
                    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                    random_state=RANDOM_STATE, verbose=-1,
                )
                model.fit(X_ab[train_idx], y_ab[train_idx], sample_weight=sample_weights)
                fold_preds[val_idx] = model.predict_proba(X_ab[val_idx])[:, 1]
                fold_test_preds.append(model.predict_proba(X_test)[:, 1])

            oof_preds[label_mask, idx] = fold_preds
            test_preds[:, idx] = np.mean(fold_test_preds, axis=0)

        oof_preds = apply_intrinsic_rules(oof_preds, species_train)
        test_preds = apply_intrinsic_rules(test_preds, species_test)

        metrics = compute_metrics(y_train, oof_preds, species_train)
        kpn_auc = metrics.get('K.pneumoniae_auc', 0)
        print(f"    K.pn AUC: {kpn_auc:.4f}")

        if kpn_auc > best_kpn_auc:
            best_kpn_auc = kpn_auc
            best_config = config['name']
            best_oof = oof_preds.copy()
            best_test = test_preds.copy()

    print(f"\n  Best config: {best_config} (K.pn: {best_kpn_auc:.4f})")

    metrics = compute_metrics(y_train, best_oof, species_train)
    print_metrics("Optimized Weights", metrics)
    return best_oof, best_test, metrics


# =============================================================================
# EXPERIMENT 2: Hyperparameter Tuning
# =============================================================================

def experiment_hyperparam_tuning(X_train, y_train, species_train, X_test, species_test):
    """
    Grid search over key LightGBM hyperparameters.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: Hyperparameter Tuning")
    print("="*60)

    param_grid = [
        {"n_estimators": 300, "num_leaves": 15, "learning_rate": 0.03, "min_child_samples": 10},
        {"n_estimators": 400, "num_leaves": 31, "learning_rate": 0.02, "min_child_samples": 15},
        {"n_estimators": 500, "num_leaves": 63, "learning_rate": 0.01, "min_child_samples": 20},
        {"n_estimators": 250, "num_leaves": 7, "learning_rate": 0.05, "min_child_samples": 5, "max_depth": 4},
        {"n_estimators": 350, "num_leaves": 127, "learning_rate": 0.02, "min_child_samples": 25},
    ]

    best_params = None
    best_kpn_auc = 0
    best_oof = None
    best_test = None

    for i, params in enumerate(param_grid):
        print(f"\n  Config {i+1}/{len(param_grid)}: {params}")

        oof_preds = np.full_like(y_train, np.nan, dtype=float)
        test_preds = np.zeros((len(X_test), len(ANTIBIOTICS)))

        for idx, antibiotic in enumerate(ANTIBIOTICS):
            label_mask = ~np.isnan(y_train[:, idx])
            if label_mask.sum() < 50:
                continue

            X_ab = X_train[label_mask]
            y_ab = y_train[label_mask, idx]
            species_ab = species_train[label_mask]

            if len(np.unique(y_ab)) < 2:
                oof_preds[label_mask, idx] = y_ab[0]
                test_preds[:, idx] = y_ab[0]
                continue

            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            fold_preds = np.zeros(label_mask.sum())
            fold_test_preds = []

            for train_idx, val_idx in skf.split(X_ab, species_ab):
                weights = np.where(species_ab[train_idx] == 3, 0.3, 1.0)

                model = lgb.LGBMClassifier(
                    **params,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=RANDOM_STATE, verbose=-1,
                )
                model.fit(X_ab[train_idx], y_ab[train_idx], sample_weight=weights)
                fold_preds[val_idx] = model.predict_proba(X_ab[val_idx])[:, 1]
                fold_test_preds.append(model.predict_proba(X_test)[:, 1])

            oof_preds[label_mask, idx] = fold_preds
            test_preds[:, idx] = np.mean(fold_test_preds, axis=0)

        oof_preds = apply_intrinsic_rules(oof_preds, species_train)
        test_preds = apply_intrinsic_rules(test_preds, species_test)

        metrics = compute_metrics(y_train, oof_preds, species_train)
        kpn_auc = metrics.get('K.pneumoniae_auc', 0)
        print(f"    K.pn AUC: {kpn_auc:.4f}")

        if kpn_auc > best_kpn_auc:
            best_kpn_auc = kpn_auc
            best_params = params
            best_oof = oof_preds.copy()
            best_test = test_preds.copy()

    print(f"\n  Best params: {best_params}")

    metrics = compute_metrics(y_train, best_oof, species_train)
    print_metrics("Hyperparameter Tuning", metrics)
    return best_oof, best_test, metrics


# =============================================================================
# EXPERIMENT 3: PLS with Optimized Components
# =============================================================================

def experiment_pls_optimized(X_train, y_train, species_train, X_test, species_test):
    """
    PLS with tuned number of components per antibiotic.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: PLS Optimized")
    print("="*60)

    oof_preds = np.full_like(y_train, np.nan, dtype=float)
    test_preds = np.zeros((len(X_test), len(ANTIBIOTICS)))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use test-proportional weights
    weights_map = {3: 0.1, 1: 2.0, 0: 1.0, 2: 1.0}

    for idx, antibiotic in enumerate(ANTIBIOTICS):
        print(f"  [{idx+1}/8] {antibiotic}")

        label_mask = ~np.isnan(y_train[:, idx])
        if label_mask.sum() < 50:
            continue

        X_ab = X_train_scaled[label_mask]
        y_ab = y_train[label_mask, idx]
        species_ab = species_train[label_mask]

        if len(np.unique(y_ab)) < 2:
            oof_preds[label_mask, idx] = y_ab[0]
            test_preds[:, idx] = y_ab[0]
            continue

        # Find best n_components via inner CV
        best_n_comp = 100
        best_inner_auc = 0

        for n_comp in [50, 100, 150, 200]:
            inner_aucs = []
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

            for train_idx, val_idx in skf.split(X_ab, species_ab):
                pls = PLSRegression(n_components=min(n_comp, len(train_idx) - 1))
                pls.fit(X_ab[train_idx], y_ab[train_idx])

                X_train_pls = pls.transform(X_ab[train_idx])
                X_val_pls = pls.transform(X_ab[val_idx])

                model = lgb.LGBMClassifier(
                    n_estimators=200, learning_rate=0.05, num_leaves=31,
                    random_state=RANDOM_STATE, verbose=-1,
                )
                weights = np.array([weights_map.get(s, 1.0) for s in species_ab[train_idx]])
                model.fit(X_train_pls, y_ab[train_idx], sample_weight=weights)

                pred = model.predict_proba(X_val_pls)[:, 1]
                inner_aucs.append(roc_auc_score(y_ab[val_idx], pred))

            mean_auc = np.mean(inner_aucs)
            if mean_auc > best_inner_auc:
                best_inner_auc = mean_auc
                best_n_comp = n_comp

        # Train with best n_components
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_preds = np.zeros(label_mask.sum())
        fold_test_preds = []

        for train_idx, val_idx in skf.split(X_ab, species_ab):
            pls = PLSRegression(n_components=min(best_n_comp, len(train_idx) - 1))
            pls.fit(X_ab[train_idx], y_ab[train_idx])

            X_train_pls = pls.transform(X_ab[train_idx])
            X_val_pls = pls.transform(X_ab[val_idx])

            model = lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.03, num_leaves=31,
                min_child_samples=15, random_state=RANDOM_STATE, verbose=-1,
            )
            weights = np.array([weights_map.get(s, 1.0) for s in species_ab[train_idx]])
            model.fit(X_train_pls, y_ab[train_idx], sample_weight=weights)

            fold_preds[val_idx] = model.predict_proba(X_val_pls)[:, 1]
            fold_test_preds.append(model.predict_proba(pls.transform(X_test_scaled))[:, 1])

        oof_preds[label_mask, idx] = fold_preds
        test_preds[:, idx] = np.mean(fold_test_preds, axis=0)

    oof_preds = apply_intrinsic_rules(oof_preds, species_train)
    test_preds = apply_intrinsic_rules(test_preds, species_test)

    metrics = compute_metrics(y_train, oof_preds, species_train)
    print_metrics("PLS Optimized", metrics)
    return oof_preds, test_preds, metrics


# =============================================================================
# EXPERIMENT 4: Species-Specific + Global Blend
# =============================================================================

def experiment_species_global_blend(X_train, y_train, species_train, X_test, species_test):
    """
    Blend species-specific models with global model.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 4: Species-Specific + Global Blend")
    print("="*60)

    weights_map = {3: 0.1, 1: 2.0, 0: 1.0, 2: 1.0}

    # Train global model first
    print("  Training global model...")
    global_oof = np.full_like(y_train, np.nan, dtype=float)
    global_test = np.zeros((len(X_test), len(ANTIBIOTICS)))

    for idx, antibiotic in enumerate(ANTIBIOTICS):
        label_mask = ~np.isnan(y_train[:, idx])
        if label_mask.sum() < 50:
            continue

        X_ab = X_train[label_mask]
        y_ab = y_train[label_mask, idx]
        species_ab = species_train[label_mask]

        if len(np.unique(y_ab)) < 2:
            global_oof[label_mask, idx] = y_ab[0]
            global_test[:, idx] = y_ab[0]
            continue

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_preds = np.zeros(label_mask.sum())
        fold_test_preds = []

        for train_idx, val_idx in skf.split(X_ab, species_ab):
            weights = np.array([weights_map.get(s, 1.0) for s in species_ab[train_idx]])

            model = lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.03, num_leaves=31,
                min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
                random_state=RANDOM_STATE, verbose=-1,
            )
            model.fit(X_ab[train_idx], y_ab[train_idx], sample_weight=weights)
            fold_preds[val_idx] = model.predict_proba(X_ab[val_idx])[:, 1]
            fold_test_preds.append(model.predict_proba(X_test)[:, 1])

        global_oof[label_mask, idx] = fold_preds
        global_test[:, idx] = np.mean(fold_test_preds, axis=0)

    # Train species-specific models
    print("  Training species-specific models...")
    species_oof = np.full_like(y_train, np.nan, dtype=float)
    species_test_preds = np.zeros((len(X_test), len(ANTIBIOTICS)))

    for species_id, species_name in SPECIES_NAMES.items():
        species_mask = (species_train == species_id)
        n_samples = species_mask.sum()

        if n_samples < 50:
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
                species_oof[species_mask, idx] = np.where(
                    np.isnan(y_species[:, idx]), np.nan, y_ab[0]
                )
                species_test_preds[species_test == species_id, idx] = y_ab[0]
                continue

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
            species_oof[labeled_indices, idx] = fold_preds

            test_mask = (species_test == species_id)
            if test_mask.sum() > 0:
                test_species = X_test[test_mask]
                species_test_preds[test_mask, idx] = np.mean(
                    [m.predict_proba(test_species)[:, 1] for m in fold_models], axis=0
                )

    # Blend: 60% global + 40% species-specific (where available)
    print("  Blending...")
    blend_oof = np.where(np.isnan(species_oof), global_oof, 0.6 * global_oof + 0.4 * species_oof)
    blend_test = 0.6 * global_test + 0.4 * species_test_preds

    blend_oof = apply_intrinsic_rules(blend_oof, species_train)
    blend_test = apply_intrinsic_rules(blend_test, species_test)

    metrics = compute_metrics(y_train, blend_oof, species_train)
    print_metrics("Species + Global Blend", metrics)
    return blend_oof, blend_test, metrics


# =============================================================================
# EXPERIMENT 5: Rank Averaging
# =============================================================================

def experiment_rank_averaging(results, y_train, species_train, species_test):
    """
    Use rank averaging instead of probability averaging.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 5: Rank Averaging")
    print("="*60)

    valid_results = {k: v for k, v in results.items() if v[0] is not None and 'K.pneumoniae_auc' in v[2]}

    if len(valid_results) < 2:
        print("  Not enough models")
        return None, None, {}

    # Take top 5 by Mean AUC (primary metric - matches leaderboard scoring)
    ranked = sorted(valid_results.items(), key=lambda x: x[1][2].get('mean_auc', 0), reverse=True)[:5]

    print("  Using models:")
    for name, (_, _, m) in ranked:
        print(f"    {name}: Mean {m.get('mean_auc', 0):.4f}, K.pn {m.get('K.pneumoniae_auc', 0):.4f}")

    # Rank average
    n_train = len(y_train)
    n_test = len(results[ranked[0][0]][1])

    oof_rank = np.zeros((n_train, len(ANTIBIOTICS)))
    test_rank = np.zeros((n_test, len(ANTIBIOTICS)))

    for name, (oof, test_pred, _) in ranked:
        for idx in range(len(ANTIBIOTICS)):
            # Handle NaN in OOF
            valid_mask = ~np.isnan(oof[:, idx])
            if valid_mask.sum() > 0:
                ranks = rankdata(oof[valid_mask, idx])
                ranks_full = np.zeros(n_train)
                ranks_full[valid_mask] = ranks / len(ranks)
                oof_rank[:, idx] += ranks_full

            # Test ranking
            test_rank[:, idx] += rankdata(test_pred[:, idx]) / len(test_pred)

    oof_rank /= len(ranked)
    test_rank /= len(ranked)

    oof_rank = apply_intrinsic_rules(oof_rank, species_train)
    test_rank = apply_intrinsic_rules(test_rank, species_test)

    metrics = compute_metrics(y_train, oof_rank, species_train)
    print_metrics("Rank Averaging", metrics)
    return oof_rank, test_rank, metrics


# =============================================================================
# EXPERIMENT 6: Power Blending
# =============================================================================

def experiment_power_blending(results, y_train, species_train, species_test):
    """
    Weight models by K.pneumoniae AUC raised to a power (sharper weights).
    """
    print("\n" + "="*60)
    print("EXPERIMENT 6: Power Blending")
    print("="*60)

    valid_results = {k: v for k, v in results.items() if v[0] is not None and 'K.pneumoniae_auc' in v[2]}

    if len(valid_results) < 2:
        print("  Not enough models")
        return None, None, {}

    ranked = sorted(valid_results.items(), key=lambda x: x[1][2].get('mean_auc', 0), reverse=True)[:5]

    # Raise weights to power 3 for sharper distinction
    weights = np.array([m[2].get('mean_auc', 0) ** 3 for _, m in ranked])
    weights = weights / weights.sum()

    print("  Power weights (by Mean AUC):")
    for (name, _), w in zip(ranked, weights):
        print(f"    {name}: {w:.3f}")

    n_train = len(y_train)
    n_test = len(results[ranked[0][0]][1])

    oof_blend = np.zeros((n_train, len(ANTIBIOTICS)))
    test_blend = np.zeros((n_test, len(ANTIBIOTICS)))

    for (name, (oof, test_pred, _)), w in zip(ranked, weights):
        valid_mask = ~np.isnan(oof)
        oof_blend = np.where(valid_mask, oof_blend + oof * w, oof_blend)
        test_blend += test_pred * w

    oof_blend = apply_intrinsic_rules(oof_blend, species_train)
    test_blend = apply_intrinsic_rules(test_blend, species_test)

    metrics = compute_metrics(y_train, oof_blend, species_train)
    print_metrics("Power Blending", metrics)
    return oof_blend, test_blend, metrics


# =============================================================================
# MAIN
# =============================================================================

def run_round2_experiments():
    """Run round 2 experiments."""
    print("\n" + "="*80)
    print("AMR Prediction: Round 2 Advanced Experiments")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print("\n[Loading Data]")
    X_train, X_test, y_train, species_train, species_test = load_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    X_train, X_test, _ = remove_constant_features(X_train, X_test)
    print(f"After filtering: Train {X_train.shape}")

    results = {}

    # Run experiments
    experiments = [
        ("Optimized-Weights", experiment_optimized_weights),
        ("Hyperparam-Tuning", experiment_hyperparam_tuning),
        ("PLS-Optimized", experiment_pls_optimized),
        ("Species-Global-Blend", experiment_species_global_blend),
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

    # Ensemble experiments
    if len(results) >= 2:
        oof, test_pred, metrics = experiment_rank_averaging(results, y_train, species_train, species_test)
        if oof is not None:
            results["Rank-Averaging"] = (oof, test_pred, metrics)
            EXPERIMENT_RESULTS["Rank-Averaging"] = metrics

        oof, test_pred, metrics = experiment_power_blending(results, y_train, species_train, species_test)
        if oof is not None:
            results["Power-Blending"] = (oof, test_pred, metrics)
            EXPERIMENT_RESULTS["Power-Blending"] = metrics

    # Summary
    print("\n" + "="*80)
    print("ROUND 2 COMPARISON")
    print("="*80)

    summary = []
    for name, metrics in EXPERIMENT_RESULTS.items():
        summary.append({
            'Method': name,
            'Mean': metrics.get('mean_auc', 0),
            'K.pn': metrics.get('K.pneumoniae_auc', 0),
            'E.coli': metrics.get('E.coli_auc', 0),
        })

    summary_df = pd.DataFrame(summary).sort_values('K.pn', ascending=False)

    print("\nRanked by K.pneumoniae AUC:")
    print("-" * 60)
    print(f"{'Method':<25} {'Mean':>8} {'K.pn':>8} {'E.coli':>8}")
    print("-" * 60)
    for _, r in summary_df.iterrows():
        print(f"{r['Method']:<25} {r['Mean']:>8.4f} {r['K.pn']:>8.4f} {r['E.coli']:>8.4f}")

    # Save results
    results_path = OUTPUT_DIR / f"round2_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(results_path, 'w') as f:
        json.dump(EXPERIMENT_RESULTS, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Save best submission
    best_method = summary_df.iloc[0]['Method']
    best_kpn = summary_df.iloc[0]['K.pn']

    print(f"\n{'='*80}")
    print(f"BEST ROUND 2: {best_method} (K.pn AUC: {best_kpn:.4f})")
    print(f"{'='*80}")

    if best_method in results:
        _, test_pred, _ = results[best_method]
        test_df = pd.read_csv(DATA_DIR / "test.csv")
        submission = pd.DataFrame({
            "sample_id": test_df["sample_id"].values,
            **{ab: test_pred[:, idx] for idx, ab in enumerate(ANTIBIOTICS)}
        })
        sub_path = OUTPUT_DIR / f"round2_best_{best_method.lower().replace('-', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        submission.to_csv(sub_path, index=False)
        print(f"Best submission: {sub_path}")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return results, EXPERIMENT_RESULTS


if __name__ == "__main__":
    results, metrics = run_round2_experiments()
