#!/usr/bin/env python3
"""
MIRACLE BLEND: Ultimate ensemble for competition deadline.

Strategy:
1. Maximum model diversity (LightGBM, XGBoost, CatBoost, MLP, PLS variants)
2. Multiple hyperparameter configurations per model type
3. Rank averaging (proven best ensemble method)
4. Species-aware sample weighting
5. Test-time augmentation

Target: Beat LB 0.83862
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple, List
from datetime import datetime
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPClassifier
from scipy.stats import rankdata
import lightgbm as lgb

# Try importing optional libraries
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed, will skip XGB models")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("CatBoost not installed, will skip CatBoost models")

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data.dataset import ANTIBIOTICS

DATA_DIR = Path("/home/erfan/data/ml_kaggle/raw")
OUTPUT_DIR = Path("/sata_disk/users/erfan/ml_kaggle/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPECIES_NAMES = {0: "E.coli", 1: "K.pneumoniae", 2: "P.mirabilis", 3: "P.aeruginosa"}
N_FOLDS = 5
RANDOM_STATE = 42

# Match test distribution for validation weighting
# Test: K.pn=51%, E.coli=27%, P.mirabilis=19%, P.aeruginosa=3%
SPECIES_WEIGHTS = {
    0: 1.5,   # E.coli: upweight (27% test vs 17% train)
    1: 2.0,   # K.pneumoniae: upweight strongly (51% test vs 28% train)
    2: 1.5,   # P.mirabilis: upweight (19% test vs 12% train)
    3: 0.1,   # P.aeruginosa: downweight strongly (3% test vs 43% train)
}

import functools
print = functools.partial(print, flush=True)


def load_data():
    """Load train and test data."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    feature_cols = [f"maldi_feature_{i}" for i in range(6000)]
    X_train = train_df[feature_cols].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_train = train_df[ANTIBIOTICS].values.astype(np.float32)
    species_train = train_df["species_id"].values.astype(np.int32)
    species_test = test_df["species_id"].values.astype(np.int32)
    sample_ids = test_df["sample_id"].values

    return X_train, X_test, y_train, species_train, species_test, sample_ids


def remove_constant_features(X_train, X_test, threshold=1e-5):
    """Remove constant/near-constant features."""
    variances = X_train.var(axis=0)
    mask = variances > threshold
    return X_train[:, mask], X_test[:, mask], mask


def get_sample_weights(species):
    """Get sample weights based on species to match test distribution."""
    return np.array([SPECIES_WEIGHTS.get(s, 1.0) for s in species])


def apply_intrinsic_rules(predictions, species_ids):
    """Apply biological intrinsic resistance rules."""
    predictions = predictions.copy()
    ANTIBIOTIC_INDICES = {ab: idx for idx, ab in enumerate(ANTIBIOTICS)}

    # P. aeruginosa intrinsic resistances
    pa_mask = (species_ids == 3)
    for ab in ["Ampicillin", "Amoxicillin_Clavulanic_acid", "Ertapenem", "Cefotaxime", "Cefuroxime"]:
        predictions[pa_mask, ANTIBIOTIC_INDICES[ab]] = 1.0

    # P. mirabilis intrinsic resistance to Imipenem
    pm_mask = (species_ids == 2)
    predictions[pm_mask, ANTIBIOTIC_INDICES["Imipenem"]] = 1.0

    return predictions


def compute_metrics(y_true, y_pred, species):
    """Compute mean AUC and per-species AUC."""
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
                metrics[f'{species_name}_mean_auc'] = np.mean(species_aucs)

    return metrics


# =============================================================================
# MODEL BUILDERS
# =============================================================================

def build_lgb_model(X_train, y_train, species_train, X_test, params: dict, name: str):
    """Build LightGBM model with given params."""
    print(f"  Building {name}...")

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
            weights = get_sample_weights(species_ab[train_idx])

            model = lgb.LGBMClassifier(
                random_state=RANDOM_STATE,
                verbose=-1,
                n_jobs=-1,
                **params
            )
            model.fit(X_ab[train_idx], y_ab[train_idx], sample_weight=weights)
            fold_preds[val_idx] = model.predict_proba(X_ab[val_idx])[:, 1]
            fold_test_preds.append(model.predict_proba(X_test)[:, 1])

        oof_preds[label_mask, idx] = fold_preds
        test_preds[:, idx] = np.mean(fold_test_preds, axis=0)

    return oof_preds, test_preds


def build_xgb_model(X_train, y_train, species_train, X_test, params: dict, name: str):
    """Build XGBoost model."""
    if not HAS_XGB:
        return None, None

    print(f"  Building {name}...")

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
            weights = get_sample_weights(species_ab[train_idx])

            model = xgb.XGBClassifier(
                random_state=RANDOM_STATE,
                verbosity=0,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='logloss',
                **params
            )
            model.fit(X_ab[train_idx], y_ab[train_idx], sample_weight=weights)
            fold_preds[val_idx] = model.predict_proba(X_ab[val_idx])[:, 1]
            fold_test_preds.append(model.predict_proba(X_test)[:, 1])

        oof_preds[label_mask, idx] = fold_preds
        test_preds[:, idx] = np.mean(fold_test_preds, axis=0)

    return oof_preds, test_preds


def build_catboost_model(X_train, y_train, species_train, X_test, params: dict, name: str):
    """Build CatBoost model."""
    if not HAS_CATBOOST:
        return None, None

    print(f"  Building {name}...")

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
            weights = get_sample_weights(species_ab[train_idx])

            model = CatBoostClassifier(
                random_seed=RANDOM_STATE,
                verbose=False,
                thread_count=-1,
                **params
            )
            model.fit(X_ab[train_idx], y_ab[train_idx], sample_weight=weights)
            fold_preds[val_idx] = model.predict_proba(X_ab[val_idx])[:, 1]
            fold_test_preds.append(model.predict_proba(X_test)[:, 1])

        oof_preds[label_mask, idx] = fold_preds
        test_preds[:, idx] = np.mean(fold_test_preds, axis=0)

    return oof_preds, test_preds


def build_mlp_model(X_train, y_train, species_train, X_test, params: dict, name: str):
    """Build MLP neural network model."""
    print(f"  Building {name}...")

    # Scale features for neural network
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    oof_preds = np.full_like(y_train, np.nan, dtype=float)
    test_preds = np.zeros((len(X_test), len(ANTIBIOTICS)))

    for idx, antibiotic in enumerate(ANTIBIOTICS):
        label_mask = ~np.isnan(y_train[:, idx])
        if label_mask.sum() < 50:
            continue

        X_ab = X_train_scaled[label_mask]
        y_ab = y_train[label_mask, idx].astype(int)
        species_ab = species_train[label_mask]

        if len(np.unique(y_ab)) < 2:
            oof_preds[label_mask, idx] = y_ab[0]
            test_preds[:, idx] = y_ab[0]
            continue

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_preds = np.zeros(label_mask.sum())
        fold_test_preds = []

        for train_idx, val_idx in skf.split(X_ab, species_ab):
            model = MLPClassifier(
                random_state=RANDOM_STATE,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                **params
            )
            # Note: MLPClassifier doesn't support sample_weight, so we skip it
            model.fit(X_ab[train_idx], y_ab[train_idx])
            fold_preds[val_idx] = model.predict_proba(X_ab[val_idx])[:, 1]
            fold_test_preds.append(model.predict_proba(X_test_scaled)[:, 1])

        oof_preds[label_mask, idx] = fold_preds
        test_preds[:, idx] = np.mean(fold_test_preds, axis=0)

    return oof_preds, test_preds


def build_pls_lgb_model(X_train, y_train, species_train, X_test, n_components: int, name: str):
    """Build PLS + LightGBM pipeline."""
    print(f"  Building {name}...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    oof_preds = np.full_like(y_train, np.nan, dtype=float)
    test_preds = np.zeros((len(X_test), len(ANTIBIOTICS)))

    for idx, antibiotic in enumerate(ANTIBIOTICS):
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

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_preds = np.zeros(label_mask.sum())
        fold_test_preds = []

        for train_idx, val_idx in skf.split(X_ab, species_ab):
            # Fit PLS
            n_comp = min(n_components, len(train_idx) - 1)
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X_ab[train_idx], y_ab[train_idx])

            X_train_pls = pls.transform(X_ab[train_idx])
            X_val_pls = pls.transform(X_ab[val_idx])
            X_test_pls = pls.transform(X_test_scaled)

            # Fit LightGBM on PLS features
            weights = get_sample_weights(species_ab[train_idx])
            model = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                random_state=RANDOM_STATE,
                verbose=-1,
            )
            model.fit(X_train_pls, y_ab[train_idx], sample_weight=weights)
            fold_preds[val_idx] = model.predict_proba(X_val_pls)[:, 1]
            fold_test_preds.append(model.predict_proba(X_test_pls)[:, 1])

        oof_preds[label_mask, idx] = fold_preds
        test_preds[:, idx] = np.mean(fold_test_preds, axis=0)

    return oof_preds, test_preds


# =============================================================================
# ENSEMBLE METHODS
# =============================================================================

def rank_average(predictions_list: List[Tuple[np.ndarray, np.ndarray]], species_train, species_test):
    """Rank averaging of multiple model predictions."""
    n_train = predictions_list[0][0].shape[0]
    n_test = predictions_list[0][1].shape[0]
    n_antibiotics = predictions_list[0][0].shape[1]

    rank_oof = np.zeros((n_train, n_antibiotics))
    rank_test = np.zeros((n_test, n_antibiotics))

    for oof, test in predictions_list:
        for idx in range(n_antibiotics):
            # OOF ranking
            valid_mask = ~np.isnan(oof[:, idx])
            if valid_mask.sum() > 0:
                ranks = rankdata(oof[valid_mask, idx])
                ranks_full = np.zeros(n_train)
                ranks_full[valid_mask] = ranks / len(ranks)
                rank_oof[:, idx] += ranks_full

            # Test ranking
            rank_test[:, idx] += rankdata(test[:, idx]) / len(test)

    # Average ranks
    n_models = len(predictions_list)
    rank_oof /= n_models
    rank_test /= n_models

    # Apply intrinsic rules
    rank_oof = apply_intrinsic_rules(rank_oof, species_train)
    rank_test = apply_intrinsic_rules(rank_test, species_test)

    return rank_oof, rank_test


def weighted_average(predictions_list: List[Tuple[np.ndarray, np.ndarray]], weights: List[float],
                     species_train, species_test):
    """Weighted probability averaging."""
    weights = np.array(weights)
    weights = weights / weights.sum()

    oof_avg = np.zeros_like(predictions_list[0][0])
    test_avg = np.zeros_like(predictions_list[0][1])

    for (oof, test), w in zip(predictions_list, weights):
        # Handle NaN in OOF
        oof_filled = np.nan_to_num(oof, nan=0.5)
        oof_avg += oof_filled * w
        test_avg += test * w

    # Apply intrinsic rules
    oof_avg = apply_intrinsic_rules(oof_avg, species_train)
    test_avg = apply_intrinsic_rules(test_avg, species_test)

    return oof_avg, test_avg


# =============================================================================
# MAIN
# =============================================================================

def miracle_blend():
    """Create the miracle blend for competition deadline."""
    print("\n" + "=" * 80)
    print("MIRACLE BLEND: Maximum Diversity Ensemble")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print("\n[Loading Data]")
    X_train, X_test, y_train, species_train, species_test, sample_ids = load_data()
    X_train, X_test, feature_mask = remove_constant_features(X_train, X_test)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Define model configurations
    models_to_build = []

    # LightGBM variants
    lgb_configs = [
        {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 15},
        {"n_estimators": 500, "learning_rate": 0.01, "num_leaves": 63, "min_child_samples": 20},
        {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 30},
        {"n_estimators": 400, "learning_rate": 0.02, "num_leaves": 127, "min_child_samples": 10},
        {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31, "subsample": 0.7, "colsample_bytree": 0.7},
    ]
    for i, cfg in enumerate(lgb_configs):
        models_to_build.append(("lgb", f"LGB_{i+1}", cfg))

    # XGBoost variants
    if HAS_XGB:
        xgb_configs = [
            {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 6},
            {"n_estimators": 500, "learning_rate": 0.01, "max_depth": 8},
            {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4},
        ]
        for i, cfg in enumerate(xgb_configs):
            models_to_build.append(("xgb", f"XGB_{i+1}", cfg))

    # CatBoost variants
    if HAS_CATBOOST:
        cb_configs = [
            {"iterations": 300, "learning_rate": 0.03, "depth": 6},
            {"iterations": 500, "learning_rate": 0.01, "depth": 8},
            {"iterations": 200, "learning_rate": 0.05, "depth": 4},
        ]
        for i, cfg in enumerate(cb_configs):
            models_to_build.append(("catboost", f"CatBoost_{i+1}", cfg))

    # MLP variants
    mlp_configs = [
        {"hidden_layer_sizes": (256, 128), "alpha": 0.01, "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (512, 256, 128), "alpha": 0.001, "learning_rate_init": 0.0005},
        {"hidden_layer_sizes": (128, 64), "alpha": 0.1, "learning_rate_init": 0.001},
    ]
    for i, cfg in enumerate(mlp_configs):
        models_to_build.append(("mlp", f"MLP_{i+1}", cfg))

    # PLS + LightGBM variants
    for n_comp in [20, 50, 100]:
        models_to_build.append(("pls_lgb", f"PLS{n_comp}_LGB", n_comp))

    # Build all models
    print(f"\n[Building {len(models_to_build)} Models]")
    all_predictions = []
    model_metrics = {}

    for model_type, name, config in models_to_build:
        try:
            if model_type == "lgb":
                oof, test = build_lgb_model(X_train, y_train, species_train, X_test, config, name)
            elif model_type == "xgb":
                oof, test = build_xgb_model(X_train, y_train, species_train, X_test, config, name)
            elif model_type == "catboost":
                oof, test = build_catboost_model(X_train, y_train, species_train, X_test, config, name)
            elif model_type == "mlp":
                oof, test = build_mlp_model(X_train, y_train, species_train, X_test, config, name)
            elif model_type == "pls_lgb":
                oof, test = build_pls_lgb_model(X_train, y_train, species_train, X_test, config, name)

            if oof is not None and test is not None:
                # Apply intrinsic rules
                oof = apply_intrinsic_rules(oof, species_train)
                test = apply_intrinsic_rules(test, species_test)

                all_predictions.append((oof, test))
                metrics = compute_metrics(y_train, oof, species_train)
                model_metrics[name] = metrics
                print(f"    {name}: Mean AUC = {metrics['mean_auc']:.4f}, K.pn = {metrics.get('K.pneumoniae_mean_auc', 0):.4f}")

        except Exception as e:
            print(f"    {name} FAILED: {e}")
            continue

    print(f"\n[Successfully built {len(all_predictions)} models]")

    # Create ensembles
    print("\n[Creating Ensembles]")

    # 1. Simple rank average of all
    rank_oof, rank_test = rank_average(all_predictions, species_train, species_test)
    rank_metrics = compute_metrics(y_train, rank_oof, species_train)
    print(f"  Rank-Average (all {len(all_predictions)} models): Mean AUC = {rank_metrics['mean_auc']:.4f}")

    # 2. Weighted average by mean AUC
    weights = [model_metrics[name]['mean_auc'] for name in model_metrics]
    weighted_oof, weighted_test = weighted_average(all_predictions, weights, species_train, species_test)
    weighted_metrics = compute_metrics(y_train, weighted_oof, species_train)
    print(f"  Weighted-Average (by Mean AUC): Mean AUC = {weighted_metrics['mean_auc']:.4f}")

    # 3. Top-N rank average (top 5 models)
    sorted_models = sorted(model_metrics.items(), key=lambda x: x[1]['mean_auc'], reverse=True)
    top_n = 5
    top_names = [name for name, _ in sorted_models[:top_n]]
    top_indices = [i for i, (model_type, name, cfg) in enumerate(models_to_build) if name in top_names and i < len(all_predictions)]
    if len(top_indices) >= 3:
        top_predictions = [all_predictions[i] for i in top_indices[:min(top_n, len(top_indices))]]
        top_rank_oof, top_rank_test = rank_average(top_predictions, species_train, species_test)
        top_rank_metrics = compute_metrics(y_train, top_rank_oof, species_train)
        print(f"  Top-{len(top_predictions)} Rank-Average: Mean AUC = {top_rank_metrics['mean_auc']:.4f}")
    else:
        top_rank_oof, top_rank_test = rank_oof, rank_test
        top_rank_metrics = rank_metrics

    # 4. Meta-blend: average of rank and weighted
    meta_oof = 0.5 * rank_oof + 0.5 * weighted_oof
    meta_test = 0.5 * rank_test + 0.5 * weighted_test
    meta_oof = apply_intrinsic_rules(meta_oof, species_train)
    meta_test = apply_intrinsic_rules(meta_test, species_test)
    meta_metrics = compute_metrics(y_train, meta_oof, species_train)
    print(f"  Meta-Blend (Rank + Weighted): Mean AUC = {meta_metrics['mean_auc']:.4f}")

    # Collect all blends
    all_blends = {
        'Rank-All': (rank_oof, rank_test, rank_metrics),
        'Weighted-All': (weighted_oof, weighted_test, weighted_metrics),
        'Top-N-Rank': (top_rank_oof, top_rank_test, top_rank_metrics),
        'Meta-Blend': (meta_oof, meta_test, meta_metrics),
    }

    # Find best
    best_name = max(all_blends, key=lambda x: all_blends[x][2]['mean_auc'])
    best_oof, best_test, best_metrics = all_blends[best_name]

    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print("\nIndividual Model Ranking (by Mean AUC):")
    for i, (name, m) in enumerate(sorted_models[:10]):
        print(f"  {i+1}. {name}: Mean AUC = {m['mean_auc']:.4f}")

    print("\nEnsemble Ranking:")
    for name, (_, _, m) in sorted(all_blends.items(), key=lambda x: x[1][2]['mean_auc'], reverse=True):
        marker = " ★ BEST" if name == best_name else ""
        print(f"  {name}: Mean AUC = {m['mean_auc']:.4f}{marker}")

    print(f"\n{'=' * 80}")
    print(f"BEST ENSEMBLE: {best_name}")
    print(f"Mean AUC: {best_metrics['mean_auc']:.4f}")
    print(f"K.pneumoniae Mean AUC: {best_metrics.get('K.pneumoniae_mean_auc', 0):.4f}")
    print(f"{'=' * 80}")

    # Save submissions
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    submission_dir = OUTPUT_DIR / "submissions"
    submission_dir.mkdir(exist_ok=True)

    # Save best ensemble
    best_sub = pd.DataFrame({
        "sample_id": sample_ids,
        **{ab: best_test[:, idx] for idx, ab in enumerate(ANTIBIOTICS)}
    })
    best_path = submission_dir / f"sub_miracle_{best_name.lower().replace('-', '_')}_{timestamp}.csv"
    best_sub.to_csv(best_path, index=False)
    print(f"\nBest submission saved: {best_path}")

    # Save all ensemble variants
    for name, (_, test, metrics) in all_blends.items():
        sub = pd.DataFrame({
            "sample_id": sample_ids,
            **{ab: test[:, idx] for idx, ab in enumerate(ANTIBIOTICS)}
        })
        path = submission_dir / f"sub_miracle_{name.lower().replace('-', '_')}_{timestamp}.csv"
        sub.to_csv(path, index=False)

    print(f"\nAll submissions saved to: {submission_dir}")

    # Save results JSON
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_models': len(all_predictions),
        'model_metrics': {name: {k: float(v) for k, v in m.items()} for name, m in model_metrics.items()},
        'ensemble_metrics': {name: {k: float(v) for k, v in m.items()} for name, (_, _, m) in all_blends.items()},
        'best_ensemble': best_name,
        'best_mean_auc': float(best_metrics['mean_auc']),
    }

    results_path = OUTPUT_DIR / "experiments" / f"miracle_blend_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")

    # Print kaggle submit command
    print(f"\n{'=' * 80}")
    print("KAGGLE SUBMISSION COMMAND:")
    print(f"{'=' * 80}")
    print(f'kaggle competitions submit -c antimicrobial-resistance-prediction-from-maldi-tof \\')
    print(f'  -f {best_path} \\')
    print(f'  -m "Miracle Blend: {best_name} ({len(all_predictions)} models, Mean AUC={best_metrics["mean_auc"]:.4f})"')
    print(f"{'=' * 80}")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return all_blends, model_metrics


if __name__ == "__main__":
    miracle_blend()
