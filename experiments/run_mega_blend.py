#!/usr/bin/env python3
"""
Final mega-blend: Combine best models from Round 1 and Round 2.

Takes the top performers from each round and creates an optimal ensemble.
"""

import sys
import os
import warnings
from pathlib import Path
from typing import Dict, Tuple
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
from joblib import Parallel, delayed
import multiprocessing
import argparse

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data.dataset import ANTIBIOTICS

DATA_DIR = Path("/home/erfan/data/ml_kaggle/raw")
OUTPUT_DIR = Path("/sata_disk/users/erfan/ml_kaggle/outputs/experiments")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPECIES_NAMES = {0: "E.coli", 1: "K.pneumoniae", 2: "P.mirabilis", 3: "P.aeruginosa"}
N_FOLDS = 5
RANDOM_STATE = 42

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


def load_external_submission_preds(submission_path: str | Path, test_df: pd.DataFrame) -> np.ndarray:
    """Load a Kaggle-format submission CSV and return prediction matrix aligned to test_df."""
    submission_path = Path(submission_path)
    df = pd.read_csv(submission_path)
    if "sample_id" not in df.columns:
        raise ValueError(f"External submission missing 'sample_id': {submission_path}")

    missing_cols = [ab for ab in ANTIBIOTICS if ab not in df.columns]
    if missing_cols:
        raise ValueError(f"External submission missing antibiotic columns {missing_cols}: {submission_path}")

    merged = test_df[["sample_id"]].merge(df[["sample_id", *ANTIBIOTICS]], on="sample_id", how="left")
    if merged[ANTIBIOTICS].isna().any().any():
        raise ValueError(
            f"External submission does not cover all test sample_id rows (NaNs after merge): {submission_path}"
        )

    preds = merged[ANTIBIOTICS].to_numpy(dtype=float)
    preds = np.clip(preds, 0.0, 1.0)
    return preds


def _safe_outer_jobs(requested: int | None) -> int:
    if requested is None or requested <= 0:
        return max(1, min(multiprocessing.cpu_count(), 8))
    return max(1, requested)


def _safe_inner_jobs(requested: int | None, outer_jobs: int) -> int:
    if requested is None or requested <= 0:
        # avoid massive oversubscription: split cores across outer workers
        return max(1, multiprocessing.cpu_count() // max(1, outer_jobs))
    return max(1, requested)


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


# =============================================================================
# Model 1: PLS + LightGBM with optimized weights (from Round 1)
# =============================================================================

def _train_pls_one_antibiotic(
    ab_idx: int,
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    species_train: np.ndarray,
    X_test_scaled: np.ndarray,
    n_folds: int,
    random_state: int,
    lgb_n_jobs: int,
    species_shift_weights: dict[int, float] | None,
) -> Tuple[int, np.ndarray, np.ndarray]:
    label_mask = ~np.isnan(y_train[:, ab_idx])
    if label_mask.sum() < 50:
        return ab_idx, np.full(y_train.shape[0], np.nan, dtype=float), np.full(X_test_scaled.shape[0], 0.5, dtype=float)

    X_ab = X_train_scaled[label_mask]
    y_ab = y_train[label_mask, ab_idx]
    species_ab = species_train[label_mask]

    if len(np.unique(y_ab)) < 2:
        oof_col = np.full(y_train.shape[0], np.nan, dtype=float)
        oof_col[label_mask] = y_ab[0]
        return ab_idx, oof_col, np.full(X_test_scaled.shape[0], y_ab[0], dtype=float)

    n_components = 100
    base_weights_map = {3: 0.05, 1: 3.0, 0: 1.5, 2: 1.5}  # Extreme-K.pn

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_preds = np.zeros(label_mask.sum())
    fold_test_preds = []

    for train_idx, val_idx in skf.split(X_ab, species_ab):
        pls = PLSRegression(n_components=min(n_components, len(train_idx) - 1))
        pls.fit(X_ab[train_idx], y_ab[train_idx])

        X_train_pls = pls.transform(X_ab[train_idx])
        X_val_pls = pls.transform(X_ab[val_idx])

        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=15,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbose=-1,
            n_jobs=lgb_n_jobs,
        )
        if species_shift_weights is None:
            weights = np.array([base_weights_map.get(int(s), 1.0) for s in species_ab[train_idx]])
        else:
            weights = np.array(
                [
                    base_weights_map.get(int(s), 1.0) * species_shift_weights.get(int(s), 1.0)
                    for s in species_ab[train_idx]
                ]
            )
        model.fit(X_train_pls, y_ab[train_idx], sample_weight=weights)

        fold_preds[val_idx] = model.predict_proba(X_val_pls)[:, 1]
        fold_test_preds.append(model.predict_proba(pls.transform(X_test_scaled))[:, 1])

    oof_col = np.full(y_train.shape[0], np.nan, dtype=float)
    oof_col[label_mask] = fold_preds
    test_col = np.mean(fold_test_preds, axis=0)
    return ab_idx, oof_col, test_col


def build_pls_model(
    X_train,
    y_train,
    species_train,
    X_test,
    species_test,
    n_folds: int,
    outer_jobs: int,
    lgb_n_jobs: int,
    species_shift_weights: dict[int, float] | None = None,
):
    """PLS Features + LightGBM with K.pn focus weights (parallel over antibiotics)."""
    print("  Building PLS model...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    oof_preds = np.full_like(y_train, np.nan, dtype=float)
    test_preds = np.zeros((len(X_test), len(ANTIBIOTICS)))

    results = Parallel(n_jobs=outer_jobs, prefer="threads")(
        delayed(_train_pls_one_antibiotic)(
            ab_idx,
            X_train_scaled,
            y_train,
            species_train,
            X_test_scaled,
            n_folds,
            RANDOM_STATE,
            lgb_n_jobs,
            species_shift_weights,
        )
        for ab_idx in range(len(ANTIBIOTICS))
    )

    for ab_idx, oof_col, test_col in results:
        oof_preds[:, ab_idx] = oof_col
        test_preds[:, ab_idx] = test_col

    return oof_preds, test_preds


# =============================================================================
# Model 2: Global + Species-Specific blend (from Round 2)
# =============================================================================

def _train_global_one_antibiotic(
    ab_idx: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    species_train: np.ndarray,
    X_test: np.ndarray,
    n_folds: int,
    random_state: int,
    lgb_n_jobs: int,
    species_shift_weights: dict[int, float] | None,
) -> Tuple[int, np.ndarray, np.ndarray]:
    base_weights_map = {3: 0.05, 1: 3.0, 0: 1.5, 2: 1.5}
    label_mask = ~np.isnan(y_train[:, ab_idx])
    if label_mask.sum() < 50:
        return ab_idx, np.full(y_train.shape[0], np.nan, dtype=float), np.full(X_test.shape[0], 0.5, dtype=float)

    X_ab = X_train[label_mask]
    y_ab = y_train[label_mask, ab_idx]
    species_ab = species_train[label_mask]

    if len(np.unique(y_ab)) < 2:
        oof_col = np.full(y_train.shape[0], np.nan, dtype=float)
        oof_col[label_mask] = y_ab[0]
        return ab_idx, oof_col, np.full(X_test.shape[0], y_ab[0], dtype=float)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_preds = np.zeros(label_mask.sum())
    fold_test_preds = []

    for train_idx, val_idx in skf.split(X_ab, species_ab):
        if species_shift_weights is None:
            weights = np.array([base_weights_map.get(int(s), 1.0) for s in species_ab[train_idx]])
        else:
            weights = np.array(
                [
                    base_weights_map.get(int(s), 1.0) * species_shift_weights.get(int(s), 1.0)
                    for s in species_ab[train_idx]
                ]
            )
        model = lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.02,
            num_leaves=127,
            min_child_samples=25,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbose=-1,
            n_jobs=lgb_n_jobs,
        )
        model.fit(X_ab[train_idx], y_ab[train_idx], sample_weight=weights)
        fold_preds[val_idx] = model.predict_proba(X_ab[val_idx])[:, 1]
        fold_test_preds.append(model.predict_proba(X_test)[:, 1])

    oof_col = np.full(y_train.shape[0], np.nan, dtype=float)
    oof_col[label_mask] = fold_preds
    return ab_idx, oof_col, np.mean(fold_test_preds, axis=0)


def _train_species_specific_one(
    species_id: int,
    ab_idx: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    species_train: np.ndarray,
    X_test: np.ndarray,
    species_test: np.ndarray,
    random_state: int,
    lgb_n_jobs: int,
) -> Tuple[int, int, np.ndarray, np.ndarray]:
    species_mask = species_train == species_id
    if species_mask.sum() < 50:
        return species_id, ab_idx, np.array([], dtype=int), np.array([], dtype=float)

    X_species = X_train[species_mask]
    y_species = y_train[species_mask]

    label_mask = ~np.isnan(y_species[:, ab_idx])
    n_labeled = int(label_mask.sum())
    if n_labeled < 20:
        return species_id, ab_idx, np.array([], dtype=int), np.array([], dtype=float)

    X_ab = X_species[label_mask]
    y_ab = y_species[label_mask, ab_idx]
    if len(np.unique(y_ab)) < 2:
        return species_id, ab_idx, np.array([], dtype=int), np.array([], dtype=float)

    n_splits = min(5, n_labeled // 10)
    if n_splits < 2:
        n_splits = 2

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
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
            random_state=random_state,
            verbose=-1,
            n_jobs=lgb_n_jobs,
        )
        model.fit(X_ab[train_idx], y_ab[train_idx])
        fold_preds[val_idx] = model.predict_proba(X_ab[val_idx])[:, 1]
        fold_models.append(model)

    # Map predictions back to global indices for OOF
    species_indices = np.where(species_mask)[0]
    labeled_indices = species_indices[label_mask]

    # Predict test subset of this species
    test_mask = species_test == species_id
    if test_mask.sum() > 0:
        test_species = X_test[test_mask]
        test_pred = np.mean([m.predict_proba(test_species)[:, 1] for m in fold_models], axis=0)
        # return test indices (in test array space) + predictions
        test_indices = np.where(test_mask)[0]
        # encode as interleaved (val indices, val preds) and (test indices, test preds) by concatenation
        # We'll return OOF mapping + a packed test mapping.
        packed = np.concatenate([test_indices.astype(np.float64), test_pred.astype(np.float64)])
    else:
        packed = np.array([], dtype=np.float64)

    # pack as float for easy transport; first half indices, second half preds
    oof_packed = np.concatenate([labeled_indices.astype(np.float64), fold_preds.astype(np.float64)])
    return species_id, ab_idx, oof_packed, packed


def build_species_global_blend(
    X_train,
    y_train,
    species_train,
    X_test,
    species_test,
    n_folds: int,
    outer_jobs: int,
    lgb_n_jobs: int,
    species_shift_weights: dict[int, float] | None = None,
):
    """Blend species-specific with global models (parallel over antibiotics/species+antibiotic)."""
    print("  Building Species-Global blend...")

    # Global model (parallel over antibiotics)
    global_oof = np.full_like(y_train, np.nan, dtype=float)
    global_test = np.zeros((len(X_test), len(ANTIBIOTICS)))

    global_results = Parallel(n_jobs=outer_jobs, prefer="threads")(
        delayed(_train_global_one_antibiotic)(
            ab_idx,
            X_train,
            y_train,
            species_train,
            X_test,
            n_folds,
            RANDOM_STATE,
            lgb_n_jobs,
            species_shift_weights,
        )
        for ab_idx in range(len(ANTIBIOTICS))
    )
    for ab_idx, oof_col, test_col in global_results:
        global_oof[:, ab_idx] = oof_col
        global_test[:, ab_idx] = test_col

    # Species-specific (parallel over (species, antibiotic) pairs)
    species_oof = np.full_like(y_train, np.nan, dtype=float)
    species_test_preds = np.zeros((len(X_test), len(ANTIBIOTICS)))

    pairs = [(sp_id, ab_idx) for sp_id in SPECIES_NAMES.keys() for ab_idx in range(len(ANTIBIOTICS))]
    ss_results = Parallel(n_jobs=outer_jobs, prefer="threads")(
        delayed(_train_species_specific_one)(
            sp_id,
            ab_idx,
            X_train,
            y_train,
            species_train,
            X_test,
            species_test,
            RANDOM_STATE,
            lgb_n_jobs,
        )
        for sp_id, ab_idx in pairs
    )

    for sp_id, ab_idx, oof_packed, test_packed in ss_results:
        if oof_packed.size > 0:
            mid = oof_packed.size // 2
            idxs = oof_packed[:mid].astype(int)
            preds = oof_packed[mid:]
            species_oof[idxs, ab_idx] = preds
        if test_packed.size > 0:
            mid = test_packed.size // 2
            t_idxs = test_packed[:mid].astype(int)
            t_preds = test_packed[mid:]
            species_test_preds[t_idxs, ab_idx] = t_preds

    # Blend
    blend_oof = np.where(np.isnan(species_oof), global_oof, 0.6 * global_oof + 0.4 * species_oof)
    blend_test = 0.6 * global_test + 0.4 * species_test_preds

    return blend_oof, blend_test


# =============================================================================
# Model 3: Tuned LightGBM (best hyperparams from Round 2)
# =============================================================================

def _train_tuned_one_antibiotic(
    ab_idx: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    species_train: np.ndarray,
    X_test: np.ndarray,
    n_folds: int,
    random_state: int,
    lgb_n_jobs: int,
    species_shift_weights: dict[int, float] | None,
) -> Tuple[int, np.ndarray, np.ndarray]:
    base_weights_map = {3: 0.05, 1: 3.0, 0: 1.5, 2: 1.5}
    label_mask = ~np.isnan(y_train[:, ab_idx])
    if label_mask.sum() < 50:
        return ab_idx, np.full(y_train.shape[0], np.nan, dtype=float), np.full(X_test.shape[0], 0.5, dtype=float)

    X_ab = X_train[label_mask]
    y_ab = y_train[label_mask, ab_idx]
    species_ab = species_train[label_mask]

    if len(np.unique(y_ab)) < 2:
        oof_col = np.full(y_train.shape[0], np.nan, dtype=float)
        oof_col[label_mask] = y_ab[0]
        return ab_idx, oof_col, np.full(X_test.shape[0], y_ab[0], dtype=float)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_preds = np.zeros(label_mask.sum())
    fold_test_preds = []
    for train_idx, val_idx in skf.split(X_ab, species_ab):
        if species_shift_weights is None:
            weights = np.array([base_weights_map.get(int(s), 1.0) for s in species_ab[train_idx]])
        else:
            weights = np.array(
                [
                    base_weights_map.get(int(s), 1.0) * species_shift_weights.get(int(s), 1.0)
                    for s in species_ab[train_idx]
                ]
            )
        model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.01,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbose=-1,
            n_jobs=lgb_n_jobs,
        )
        model.fit(X_ab[train_idx], y_ab[train_idx], sample_weight=weights)
        fold_preds[val_idx] = model.predict_proba(X_ab[val_idx])[:, 1]
        fold_test_preds.append(model.predict_proba(X_test)[:, 1])

    oof_col = np.full(y_train.shape[0], np.nan, dtype=float)
    oof_col[label_mask] = fold_preds
    return ab_idx, oof_col, np.mean(fold_test_preds, axis=0)


def build_tuned_lgb(
    X_train,
    y_train,
    species_train,
    X_test,
    species_test,
    n_folds: int,
    outer_jobs: int,
    lgb_n_jobs: int,
    species_shift_weights: dict[int, float] | None = None,
):
    """Tuned LightGBM with best hyperparams (parallel over antibiotics)."""
    print("  Building tuned LightGBM...")

    oof_preds = np.full_like(y_train, np.nan, dtype=float)
    test_preds = np.zeros((len(X_test), len(ANTIBIOTICS)))

    results = Parallel(n_jobs=outer_jobs, prefer="threads")(
        delayed(_train_tuned_one_antibiotic)(
            ab_idx,
            X_train,
            y_train,
            species_train,
            X_test,
            n_folds,
            RANDOM_STATE,
            lgb_n_jobs,
            species_shift_weights,
        )
        for ab_idx in range(len(ANTIBIOTICS))
    )

    for ab_idx, oof_col, test_col in results:
        oof_preds[:, ab_idx] = oof_col
        test_preds[:, ab_idx] = test_col

    return oof_preds, test_preds


# =============================================================================
# MEGA BLEND
# =============================================================================

def mega_blend(
    n_folds: int = N_FOLDS,
    outer_jobs: int = 0,
    lgb_n_jobs: int = 0,
    external_submissions: list[str] | None = None,
    reweight_to_test: bool = False,
):
    """Create final mega-blend of all best models."""
    print("\n" + "="*80)
    print("MEGA BLEND: Final Ensemble")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    outer_jobs = _safe_outer_jobs(outer_jobs)
    lgb_n_jobs = _safe_inner_jobs(lgb_n_jobs, outer_jobs)

    # Reduce OpenMP oversubscription if user parallelizes across antibiotics
    os.environ.setdefault("OMP_NUM_THREADS", str(lgb_n_jobs))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(lgb_n_jobs))
    os.environ.setdefault("MKL_NUM_THREADS", str(lgb_n_jobs))

    print(f"\n[Parallelism] outer_jobs={outer_jobs} (antibiotics/species), lgb_n_jobs={lgb_n_jobs} (per model), folds={n_folds}")

    # Load data
    print("\n[Loading Data]")
    X_train, X_test, y_train, species_train, species_test = load_data()
    X_train, X_test, _ = remove_constant_features(X_train, X_test)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    species_shift_weights: dict[int, float] | None = None
    if reweight_to_test:
        train_counts = np.bincount(species_train.astype(int), minlength=len(SPECIES_NAMES))
        test_counts = np.bincount(species_test.astype(int), minlength=len(SPECIES_NAMES))
        train_freq = train_counts / max(1, int(train_counts.sum()))
        test_freq = test_counts / max(1, int(test_counts.sum()))

        ratio = np.ones_like(train_freq, dtype=float)
        nonzero = train_freq > 0
        ratio[nonzero] = test_freq[nonzero] / train_freq[nonzero]
        ratio = np.clip(ratio, 0.2, 5.0)
        species_shift_weights = {i: float(ratio[i]) for i in range(len(SPECIES_NAMES))}
        print(f"[Shift weights] test/train ratio (clipped 0.2..5.0): {species_shift_weights}")

    # Build all models
    print("\n[Building Models]")
    models = {}

    pls_oof, pls_test = build_pls_model(
        X_train,
        y_train,
        species_train,
        X_test,
        species_test,
        n_folds,
        outer_jobs,
        lgb_n_jobs,
        species_shift_weights=species_shift_weights,
    )
    models['PLS'] = (pls_oof, pls_test)

    sgb_oof, sgb_test = build_species_global_blend(
        X_train,
        y_train,
        species_train,
        X_test,
        species_test,
        n_folds,
        outer_jobs,
        lgb_n_jobs,
        species_shift_weights=species_shift_weights,
    )
    models['Species-Global'] = (sgb_oof, sgb_test)

    lgb_oof, lgb_test = build_tuned_lgb(
        X_train,
        y_train,
        species_train,
        X_test,
        species_test,
        n_folds,
        outer_jobs,
        lgb_n_jobs,
        species_shift_weights=species_shift_weights,
    )
    models['Tuned-LGB'] = (lgb_oof, lgb_test)

    # Apply intrinsic rules
    for name in models:
        oof, test = models[name]
        models[name] = (apply_intrinsic_rules(oof, species_train), apply_intrinsic_rules(test, species_test))

    # Compute individual metrics
    print("\n[Individual Model Metrics]")
    model_kpn = {}
    for name, (oof, _) in models.items():
        metrics = compute_metrics(y_train, oof, species_train)
        kpn = metrics.get('K.pneumoniae_auc', 0)
        model_kpn[name] = kpn
        print(f"  {name:20} K.pn: {kpn:.4f}, Mean: {metrics.get('mean_auc', 0):.4f}")

    # Blend 1: Simple average
    print("\n[Blend 1: Simple Average]")
    n_models = len(models)
    avg_oof = sum(oof for oof, _ in models.values()) / n_models
    avg_test = sum(test for _, test in models.values()) / n_models

    avg_metrics = compute_metrics(y_train, avg_oof, species_train)
    print(f"  K.pn: {avg_metrics.get('K.pneumoniae_auc', 0):.4f}, Mean: {avg_metrics.get('mean_auc', 0):.4f}")

    # Blend 2: Weighted by K.pn AUC
    print("\n[Blend 2: K.pn-Weighted]")
    weights = np.array([model_kpn[name] for name in models])
    weights = weights / weights.sum()

    weighted_oof = sum(oof * w for (oof, _), w in zip(models.values(), weights))
    weighted_test = sum(test * w for (_, test), w in zip(models.values(), weights))

    weighted_metrics = compute_metrics(y_train, weighted_oof, species_train)
    print(f"  K.pn: {weighted_metrics.get('K.pneumoniae_auc', 0):.4f}, Mean: {weighted_metrics.get('mean_auc', 0):.4f}")

    # Blend 3: Power-weighted
    print("\n[Blend 3: Power-Weighted (K.pn^3)]")
    power_weights = np.array([model_kpn[name] ** 3 for name in models])
    power_weights = power_weights / power_weights.sum()

    power_oof = sum(oof * w for (oof, _), w in zip(models.values(), power_weights))
    power_test = sum(test * w for (_, test), w in zip(models.values(), power_weights))

    power_metrics = compute_metrics(y_train, power_oof, species_train)
    print(f"  K.pn: {power_metrics.get('K.pneumoniae_auc', 0):.4f}, Mean: {power_metrics.get('mean_auc', 0):.4f}")

    # Blend 4: Rank averaging
    print("\n[Blend 4: Rank Averaging]")
    n_train = len(y_train)
    n_test = len(X_test)

    rank_oof = np.zeros((n_train, len(ANTIBIOTICS)))
    rank_test = np.zeros((n_test, len(ANTIBIOTICS)))

    for name, (oof, test) in models.items():
        for idx in range(len(ANTIBIOTICS)):
            valid_mask = ~np.isnan(oof[:, idx])
            if valid_mask.sum() > 0:
                ranks = rankdata(oof[valid_mask, idx])
                ranks_full = np.zeros(n_train)
                ranks_full[valid_mask] = ranks / len(ranks)
                rank_oof[:, idx] += ranks_full

            rank_test[:, idx] += rankdata(test[:, idx]) / len(test)

    rank_oof /= len(models)
    rank_test /= len(models)

    rank_oof = apply_intrinsic_rules(rank_oof, species_train)
    rank_test = apply_intrinsic_rules(rank_test, species_test)

    rank_metrics = compute_metrics(y_train, rank_oof, species_train)
    print(f"  K.pn: {rank_metrics.get('K.pneumoniae_auc', 0):.4f}, Mean: {rank_metrics.get('mean_auc', 0):.4f}")

    # Blend 5: Blend of blends (average + rank)
    print("\n[Blend 5: Meta-Blend (Avg + Rank)]")
    meta_oof = 0.5 * weighted_oof + 0.5 * rank_oof
    meta_test = 0.5 * weighted_test + 0.5 * rank_test

    meta_metrics = compute_metrics(y_train, meta_oof, species_train)
    print(f"  K.pn: {meta_metrics.get('K.pneumoniae_auc', 0):.4f}, Mean: {meta_metrics.get('mean_auc', 0):.4f}")

    # Find best
    all_blends = {
        'Simple-Avg': (avg_oof, avg_test, avg_metrics),
        'K.pn-Weighted': (weighted_oof, weighted_test, weighted_metrics),
        'Power-Weighted': (power_oof, power_test, power_metrics),
        'Rank-Avg': (rank_oof, rank_test, rank_metrics),
        'Meta-Blend': (meta_oof, meta_test, meta_metrics),
    }

    best_name = max(all_blends, key=lambda x: all_blends[x][2].get('mean_auc', 0))
    best_oof, best_test, best_metrics = all_blends[best_name]

    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)

    print("\nRanked by Mean AUC (primary metric - matches leaderboard scoring):")
    print("-" * 60)
    for name, (_, _, m) in sorted(all_blends.items(), key=lambda x: x[1][2].get('mean_auc', 0), reverse=True):
        print(f"  {name:20} Mean: {m.get('mean_auc', 0):.4f}, K.pn: {m.get('K.pneumoniae_auc', 0):.4f}")

    print(f"\n{'='*80}")
    print(f"BEST: {best_name}")
    print(f"Mean AUC: {best_metrics.get('mean_auc', 0):.4f}")
    print(f"K.pneumoniae AUC: {best_metrics.get('K.pneumoniae_auc', 0):.4f} (secondary)")
    print(f"{'='*80}")

    # Save submission
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    submission = pd.DataFrame({
        "sample_id": test_df["sample_id"].values,
        **{ab: best_test[:, idx] for idx, ab in enumerate(ANTIBIOTICS)}
    })

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    sub_path = OUTPUT_DIR / f"mega_blend_{best_name.lower().replace('-', '_')}_{timestamp}.csv"
    submission.to_csv(sub_path, index=False)
    print(f"\nSubmission saved: {sub_path}")

    # Also save rank-avg and meta-blend for comparison
    for name in ['Rank-Avg', 'Meta-Blend']:
        _, test_pred, _ = all_blends[name]
        sub = pd.DataFrame({
            "sample_id": test_df["sample_id"].values,
            **{ab: test_pred[:, idx] for idx, ab in enumerate(ANTIBIOTICS)}
        })
        sub_path_alt = OUTPUT_DIR / f"mega_{name.lower().replace('-', '_')}_{timestamp}.csv"
        sub.to_csv(sub_path_alt, index=False)
        print(f"Alt submission saved: {sub_path_alt}")

    # Optional: blend in external submissions (e.g., self-training outputs) into Rank-Avg test preds
    if external_submissions:
        ext_preds_list = []
        for p in external_submissions:
            try:
                ext_preds_list.append(load_external_submission_preds(p, test_df))
                print(f"Loaded external submission: {p}")
            except Exception as e:
                print(f"WARNING: skipping external submission {p}: {e}")

        if ext_preds_list:
            base_rank_avg_test = all_blends['Rank-Avg'][1]
            # base_rank_avg_test is already in probability space, but was produced by rank averaging.
            # We reproduce rank-avg test mixing by adding normalized ranks from external preds.
            internal_rank_sum = np.zeros_like(base_rank_avg_test)
            for idx in range(len(ANTIBIOTICS)):
                internal_rank_sum[:, idx] = rankdata(base_rank_avg_test[:, idx]) / len(base_rank_avg_test)

            ext_rank_sum = np.zeros_like(base_rank_avg_test)
            for ext in ext_preds_list:
                for idx in range(len(ANTIBIOTICS)):
                    ext_rank_sum[:, idx] += rankdata(ext[:, idx]) / len(ext)

            combined_rank = (internal_rank_sum + ext_rank_sum) / (1.0 + len(ext_preds_list))
            combined_rank = apply_intrinsic_rules(combined_rank, species_test)

            sub_ext = pd.DataFrame({
                "sample_id": test_df["sample_id"].values,
                **{ab: combined_rank[:, idx] for idx, ab in enumerate(ANTIBIOTICS)}
            })
            sub_path_ext = OUTPUT_DIR / f"mega_rank_avg_plus_external_{timestamp}.csv"
            sub_ext.to_csv(sub_path_ext, index=False)
            print(f"External-blended submission saved: {sub_path_ext}")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return all_blends


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MEGA BLEND: parallelized across antibiotics")
    parser.add_argument("--folds", type=int, default=N_FOLDS, help="CV folds per antibiotic (default: 5)")
    parser.add_argument(
        "--outer-jobs",
        type=int,
        default=0,
        help="Parallel workers across antibiotics/species (0=auto)",
    )
    parser.add_argument(
        "--lgb-jobs",
        type=int,
        default=0,
        help="LightGBM threads per model fit (0=auto based on outer-jobs)",
    )
    parser.add_argument(
        "--external-submission",
        action="append",
        default=None,
        help="Path to a submission CSV to blend into Rank-Avg test predictions (can be repeated).",
    )
    parser.add_argument(
        "--reweight-to-test",
        action="store_true",
        help="Apply conservative species shift weights (test/train ratio) during training.",
    )
    args = parser.parse_args()
    mega_blend(
        n_folds=args.folds,
        outer_jobs=args.outer_jobs,
        lgb_n_jobs=args.lgb_jobs,
        external_submissions=args.external_submission,
        reweight_to_test=args.reweight_to_test,
    )
