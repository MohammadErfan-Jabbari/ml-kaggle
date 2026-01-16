#!/usr/bin/env python3
"""
Controlled Experiment Framework for DR Research.

Provides standardized experiment infrastructure for the Dimensionality Reduction
Research Pipeline. Guarantees consistent CV splits, fixed LightGBM hyperparameters,
and comprehensive metrics tracking for fair comparison across methods.

Usage:
    from controlled_experiment import ControlledExperiment

    # Baseline (no DR)
    exp = ControlledExperiment(name="baseline")
    results = exp.run()

    # With transformer
    from sklearn.decomposition import TruncatedSVD
    exp = ControlledExperiment(
        name="truncated_svd_50",
        transformer=TruncatedSVD(n_components=50),
        use_scaler=False
    )
    results = exp.run()
"""

import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data.dataset import ANTIBIOTICS

# Constants
DATA_DIR = Path("/home/erfan/data/ml_kaggle/raw")
OUTPUT_DIR = Path("/sata_disk/users/erfan/ml_kaggle/outputs/experiments")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

SPECIES_NAMES = {0: "E.coli", 1: "K.pneumoniae", 2: "P.mirabilis", 3: "P.aeruginosa"}
N_FOLDS = 5
RANDOM_STATE = 42

# Cached data
_DATA_CACHE = None


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess data (cached)."""
    global _DATA_CACHE
    if _DATA_CACHE is not None:
        return _DATA_CACHE

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    feature_cols = [f"maldi_feature_{i}" for i in range(6000)]
    X_train = train_df[feature_cols].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_train = train_df[ANTIBIOTICS].values.astype(np.float32)
    species_train = train_df["species_id"].values.astype(np.int32)
    species_test = test_df["species_id"].values.astype(np.int32)

    _DATA_CACHE = (X_train, X_test, y_train, species_train, species_test)
    return _DATA_CACHE


def remove_constant_features(X_train: np.ndarray, X_test: np.ndarray,
                             threshold: float = 1e-5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove features with variance below threshold."""
    variances = X_train.var(axis=0)
    mask = variances > threshold
    return X_train[:, mask], X_test[:, mask], mask


def compute_sample_weights(species: np.ndarray, pa_weight: float = 0.3) -> np.ndarray:
    """Compute sample weights to counteract species distribution shift."""
    return np.where(species == 3, pa_weight, 1.0).astype(np.float32)


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


@dataclass
class ExperimentResult:
    """Container for experiment results."""
    name: str
    timestamp: str
    mean_auc: float
    mean_auc_std: float
    per_antibiotic: Dict[str, Dict[str, Any]]
    per_species: Dict[str, Dict[str, Any]]
    config: Dict[str, Any]
    oof_predictions: Optional[np.ndarray] = field(default=None, repr=False)
    test_predictions: Optional[np.ndarray] = field(default=None, repr=False)
    # NEW: Validation metrics (Phase 5.5)
    val_predictions: Optional[np.ndarray] = field(default=None, repr=False)
    val_mean_auc: Optional[float] = field(default=None)
    val_mean_auc_ci: Optional[Tuple[float, float]] = field(default=None)  # 95% CI
    val_per_antibiotic: Optional[Dict[str, Dict[str, Any]]] = field(default=None)
    val_per_species: Optional[Dict[str, Dict[str, Any]]] = field(default=None)
    val_K_pneumoniae_auc: Optional[float] = field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        d = {
            'name': self.name,
            'timestamp': self.timestamp,
            'mean_auc': self.mean_auc,
            'mean_auc_std': self.mean_auc_std,
            'per_antibiotic': self.per_antibiotic,
            'per_species': self.per_species,
            'config': self.config,
        }
        # Add convenience top-level keys for K.pneumoniae
        if 'K.pneumoniae' in self.per_species:
            d['K.pneumoniae_auc'] = self.per_species['K.pneumoniae'].get('mean')
            d['K.pneumoniae_auc_std'] = self.per_species['K.pneumoniae'].get('std')

        # Add validation metrics if available
        if self.val_mean_auc is not None:
            d['val_mean_auc'] = self.val_mean_auc
            d['val_K_pneumoniae_auc'] = self.val_K_pneumoniae_auc
            if self.val_mean_auc_ci:
                d['val_mean_auc_ci'] = self.val_mean_auc_ci
            if self.val_per_antibiotic:
                d['val_per_antibiotic'] = self.val_per_antibiotic
            if self.val_per_species:
                d['val_per_species'] = self.val_per_species
        return d


class ControlledExperiment:
    """
    Standardized experiment framework for DR research.

    Guarantees:
    - Same 5-fold species-stratified splits for all experiments
    - Fixed LightGBM hyperparameters (no early stopping)
    - Consistent metrics computation
    - Per-fold results for statistical testing
    """

    # Fixed LightGBM params (not configurable)
    LGB_PARAMS = {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE,
        'verbose': -1,
    }

    def __init__(self, name: str, transformer=None, use_scaler: bool = False, use_validation: bool = False):
        """
        Initialize controlled experiment.

        Args:
            name: Experiment identifier (e.g., "pls_7", "truncated_svd_50")
            transformer: sklearn-compatible transformer with fit/transform interface
            use_scaler: If True, apply StandardScaler before transformer
            use_validation: If True, load validation split and compute val metrics (Phase 5.5)
        """
        self.name = name
        self.transformer = transformer
        self.use_scaler = use_scaler
        self.use_validation = use_validation
        self.result: Optional[ExperimentResult] = None
        self._n_features_after_constant_removal = None

    def _is_supervised_dr(self) -> bool:
        """Check if transformer requires y for fitting."""
        if self.transformer is None:
            return False

        # Handle Pipeline - check final step
        from sklearn.pipeline import Pipeline
        if isinstance(self.transformer, Pipeline):
            final_step = self.transformer.steps[-1][1]
            return self._check_supervised(final_step)

        return self._check_supervised(self.transformer)

    def _check_supervised(self, estimator) -> bool:
        """Check if a single estimator is a supervised DR method."""
        try:
            from sklearn.cross_decomposition import PLSRegression, PLSCanonical, PLSSVD
            # Note: CCA excluded - use PLSCanonical instead (CCA.transform requires Y)
            supervised_types = (PLSRegression, PLSCanonical, PLSSVD)
            return isinstance(estimator, supervised_types)
        except ImportError:
            return False

    def _fit_transform_fold(self, X_tr: np.ndarray, y_tr: np.ndarray,
                            X_val: np.ndarray, X_test: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply scaler and transformer to fold data."""
        # Apply scaler if requested
        if self.use_scaler:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

        # No transformer - return as-is
        if self.transformer is None:
            return X_tr, X_val, X_test

        # Clone transformer to avoid state leakage between folds
        transformer = clone(self.transformer)

        # For supervised DR (PLS/CCA): use fully-labeled rows for fitting
        if self._is_supervised_dr():
            full_label_mask = ~np.isnan(y_tr).any(axis=1)
            n_full_labels = full_label_mask.sum()
            if n_full_labels < 100:
                print(f"    WARNING: Only {n_full_labels} fully-labeled samples for supervised DR")
            transformer.fit(X_tr[full_label_mask], y_tr[full_label_mask])
        else:
            # Unsupervised: y is ignored by sklearn convention
            transformer.fit(X_tr, y_tr)

        return (
            transformer.transform(X_tr),
            transformer.transform(X_val),
            transformer.transform(X_test)
        )

    def run(self) -> ExperimentResult:
        """Execute experiment with controlled CV."""
        print(f"\n{'='*60}")
        print(f"Experiment: {self.name}")
        print(f"{'='*60}")
        print(f"Transformer: {self.transformer}")
        print(f"Use scaler: {self.use_scaler}")
        print(f"Use validation: {self.use_validation}")

        # Phase 5.5: Load validation split FIRST (prevents data leakage)
        if self.use_validation:
            print("\n[Step 1] Loading validation split...")
            from src.data.dataset import load_validation_split
            try:
                # Load the pre-saved validation split (already has raw features)
                X, X_val, y, y_val, species, species_val = load_validation_split()
                print(f"  Train: {X.shape}, Val: {X_val.shape}")

                # Load test data separately
                _, X_test, _, _, species_test = load_data()
                print(f"  Test: {X_test.shape}")

            except FileNotFoundError:
                # If split doesn't exist, load full data and create it
                print("  Validation split not found. Loading full data...")
                X_full, X_test_full, y_full, species_full, species_test_full = load_data()

                print("  Creating validation split...")
                from src.data.dataset import create_test_distribution_split
                X, X_val, y, y_val, species, species_val = create_test_distribution_split(
                    X_full, y_full, species_full
                )

                # Set test data
                X_test = X_test_full
                species_test = species_test_full

        else:
            # Original behavior: use full training data
            print("\n[Step 1] Loading data...")
            X, X_test, y, species, species_test = load_data()
            X_val, y_val, species_val = None, None, None
            print(f"  Train: {X.shape}, Test: {X_test.shape}")

        # Remove constant features from train, val (if exists), and test
        print("\n[Step 2] Removing constant features...")
        X, X_test, feature_mask = remove_constant_features(X, X_test)
        if X_val is not None:
            X_val = X_val[:, feature_mask]
        self._n_features_after_constant_removal = X.shape[1]
        print(f"  Features: 6000 -> {X.shape[1]}")

        # Setup CV and storage
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        # OOF predictions shape depends on whether we're using validation
        if self.use_validation:
            oof_preds = np.full_like(y, np.nan)  # Training subset only
        else:
            oof_preds = np.full_like(y, np.nan)  # Full training data

        test_preds_folds = []
        fold_metrics = []

        # CV loop
        print(f"\n[Step 3] Running {N_FOLDS}-fold CV...")
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, species)):
            print(f"\n  Fold {fold_idx + 1}/{N_FOLDS}")

            # Get fold data
            X_tr, X_val_fold = X[train_idx], X[val_idx]
            y_tr, y_val_fold = y[train_idx], y[val_idx]
            species_tr = species[train_idx]
            species_val_fold = species[val_idx]

            # Apply transformer (scaler + DR) - fit on train only
            X_tr_t, X_val_t, X_test_t = self._fit_transform_fold(X_tr, y_tr, X_val_fold, X_test)

            if fold_idx == 0:
                print(f"    Features after transform: {X_tr_t.shape[1]}")

            # Initialize test predictions for this fold
            fold_test_preds = np.zeros((len(X_test), len(ANTIBIOTICS)))

            # Train 8 LightGBM models
            for ab_idx, antibiotic in enumerate(ANTIBIOTICS):
                # Mask NaN labels
                train_mask = ~np.isnan(y_tr[:, ab_idx])
                val_mask = ~np.isnan(y_val_fold[:, ab_idx])

                if train_mask.sum() < 50:
                    print(f"    WARNING: {antibiotic} has only {train_mask.sum()} train samples")
                    continue

                # Train with sample weights
                model = lgb.LGBMClassifier(**self.LGB_PARAMS)
                weights = compute_sample_weights(species_tr[train_mask])
                model.fit(X_tr_t[train_mask], y_tr[train_mask, ab_idx], sample_weight=weights)

                # OOF predictions (only for samples with labels)
                if val_mask.sum() > 0:
                    val_proba = model.predict_proba(X_val_t[val_mask])[:, 1]
                    oof_preds[val_idx[val_mask], ab_idx] = val_proba

                # Test predictions
                fold_test_preds[:, ab_idx] = model.predict_proba(X_test_t)[:, 1]

            test_preds_folds.append(fold_test_preds)

            # Compute fold metrics
            fold_oof = oof_preds[val_idx]
            fold_metric = compute_metrics(y_val_fold, fold_oof, species_val_fold)
            fold_metrics.append(fold_metric)
            print(f"    Mean AUC: {fold_metric.get('mean_auc', 0):.4f}, "
                  f"K.pn: {fold_metric.get('K.pneumoniae_auc', float('nan')):.4f}")

        # Average test predictions across folds
        test_preds = np.mean(test_preds_folds, axis=0)

        # Apply intrinsic rules
        test_preds = apply_intrinsic_rules(test_preds, species_test)

        # Phase 5.5: Compute validation metrics if using validation
        val_metrics_result = None
        if self.use_validation and X_val is not None:
            print("\n[Step 4] Computing validation metrics...")

            # Need to transform val features using the same transformations
            # For this, we fit on the full training data and transform val
            if self.use_scaler:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_t = scaler.fit_transform(X)
                X_val_t = scaler.transform(X_val)
            else:
                X_t = X.copy()
                X_val_t = X_val.copy()

            # Apply transformer if present
            if self.transformer is not None:
                # Clone and fit transformer on full training data
                transformer = clone(self.transformer)
                if self._is_supervised_dr():
                    full_label_mask = ~np.isnan(y).any(axis=1)
                    transformer.fit(X_t[full_label_mask], y[full_label_mask])
                else:
                    transformer.fit(X_t, y)
                X_t = transformer.transform(X_t)
                X_val_t = transformer.transform(X_val_t)

            # Generate predictions on val set using models trained on full training data
            val_preds = np.zeros((len(X_val), len(ANTIBIOTICS)))

            # Train final models on full training data
            for ab_idx, antibiotic in enumerate(ANTIBIOTICS):
                train_mask = ~np.isnan(y[:, ab_idx])

                if train_mask.sum() < 50:
                    continue

                model = lgb.LGBMClassifier(**self.LGB_PARAMS)
                weights = compute_sample_weights(species[train_mask])
                model.fit(X_t[train_mask], y[train_mask, ab_idx], sample_weight=weights)
                val_preds[:, ab_idx] = model.predict_proba(X_val_t)[:, 1]

            # Compute val metrics
            val_metrics_result = compute_metrics(y_val, val_preds, species_val)

            print(f"  Val Mean AUC: {val_metrics_result.get('mean_auc', 0):.4f}")
            print(f"  Val K.pn AUC: {val_metrics_result.get('K.pneumoniae_auc', 0):.4f}")

        # Aggregate results
        print("\n[Step 5] Aggregating results...")
        self.result = self._aggregate_results(
            y, oof_preds, species, test_preds, fold_metrics,
            val_metrics_result=val_metrics_result,
            val_preds=val_preds if X_val is not None else None
        )

        return self.result

    def _aggregate_results(self, y: np.ndarray, oof_preds: np.ndarray,
                           species: np.ndarray, test_preds: np.ndarray,
                           fold_metrics: List[Dict],
                           val_metrics_result: Optional[Dict] = None,
                           val_preds: Optional[np.ndarray] = None) -> ExperimentResult:
        """Aggregate fold metrics into final results."""

        # Per-antibiotic metrics with fold breakdown
        per_antibiotic = {}
        for ab in ANTIBIOTICS:
            fold_aucs = [fm.get(ab, float('nan')) for fm in fold_metrics]
            valid_aucs = [a for a in fold_aucs if not np.isnan(a)]
            per_antibiotic[ab] = {
                'mean': np.mean(valid_aucs) if valid_aucs else None,
                'std': np.std(valid_aucs) if len(valid_aucs) > 1 else None,
                'folds': fold_aucs
            }

        # Per-species metrics with fold breakdown
        per_species = {}
        for species_name in SPECIES_NAMES.values():
            key = f'{species_name}_auc'
            fold_aucs = [fm.get(key, float('nan')) for fm in fold_metrics]
            valid_aucs = [a for a in fold_aucs if not np.isnan(a)]
            per_species[species_name] = {
                'mean': np.mean(valid_aucs) if valid_aucs else None,
                'std': np.std(valid_aucs) if len(valid_aucs) > 1 else None,
                'folds': fold_aucs
            }

        # Overall mean AUC with std
        mean_aucs = [fm.get('mean_auc', 0) for fm in fold_metrics]
        overall_mean = np.mean(mean_aucs)
        overall_std = np.std(mean_aucs)

        # Phase 5.5: Process validation metrics
        val_mean_auc = None
        val_mean_auc_ci = None
        val_per_antibiotic = None
        val_per_species = None
        val_K_pneumoniae_auc = None

        if val_metrics_result is not None:
            val_mean_auc = val_metrics_result.get('mean_auc')

            # Per-antibiotic val metrics
            val_per_antibiotic = {}
            for ab in ANTIBIOTICS:
                if ab in val_metrics_result:
                    val_per_antibiotic[ab] = {
                        'mean': val_metrics_result[ab],
                        'std': None,  # No fold-level val metrics
                        'folds': []
                    }

            # Per-species val metrics
            val_per_species = {}
            for species_name in SPECIES_NAMES.values():
                key = f'{species_name}_auc'
                if key in val_metrics_result:
                    val_per_species[species_name] = {
                        'mean': val_metrics_result[key],
                        'std': None,  # No fold-level val metrics
                        'folds': []
                    }

            # K.pneumoniae val AUC (for easy access)
            val_K_pneumoniae_auc = val_metrics_result.get('K.pneumoniae_auc')

        # Config for reproducibility
        config = {
            'n_folds': N_FOLDS,
            'random_state': RANDOM_STATE,
            'lgb_params': self.LGB_PARAMS,
            'n_features_input': 6000,
            'n_features_after_constant_removal': self._n_features_after_constant_removal,
            'transformer': str(self.transformer) if self.transformer else None,
            'use_scaler': self.use_scaler,
            'use_validation': self.use_validation
        }

        return ExperimentResult(
            name=self.name,
            timestamp=datetime.now().isoformat(),
            mean_auc=overall_mean,
            mean_auc_std=overall_std,
            per_antibiotic=per_antibiotic,
            per_species=per_species,
            config=config,
            oof_predictions=oof_preds,
            test_predictions=test_preds,
            val_predictions=val_preds,
            val_mean_auc=val_mean_auc,
            val_mean_auc_ci=val_mean_auc_ci,
            val_per_antibiotic=val_per_antibiotic,
            val_per_species=val_per_species,
            val_K_pneumoniae_auc=val_K_pneumoniae_auc
        )

    def print_summary(self):
        """Pretty-print experiment results."""
        if self.result is None:
            print("No results yet. Run the experiment first.")
            return

        r = self.result
        print(f"\n{'='*60}")
        print(f"RESULTS: {r.name}")
        print(f"{'='*60}")
        print(f"Timestamp: {r.timestamp}")
        print(f"\nOverall: Mean AUC = {r.mean_auc:.4f} +/- {r.mean_auc_std:.4f}")

        # Phase 5.5: Show validation metrics if available
        if r.val_mean_auc is not None:
            print(f"         Val Mean AUC = {r.val_mean_auc:.4f}")
            if r.val_mean_auc_ci:
                print(f"         Val 95% CI: [{r.val_mean_auc_ci[0]:.4f}, {r.val_mean_auc_ci[1]:.4f}]")
            oof_val_gap = r.mean_auc - r.val_mean_auc
            print(f"         OOF-Val Gap = {oof_val_gap:+.4f}")

        print(f"\nPer-Antibiotic (OOF):")
        for ab, metrics in r.per_antibiotic.items():
            mean = metrics.get('mean')
            std = metrics.get('std')
            if mean is not None:
                std_str = f" +/- {std:.4f}" if std is not None else ""
                print(f"  {ab:35} {mean:.4f}{std_str}")

        # Phase 5.5: Show validation metrics per antibiotic
        if r.val_per_antibiotic:
            print(f"\nPer-Antibiotic (Val):")
            for ab, metrics in r.val_per_antibiotic.items():
                mean = metrics.get('mean')
                oof_mean = r.per_antibiotic.get(ab, {}).get('mean')
                if mean is not None and oof_mean is not None:
                    gap = mean - oof_mean
                    print(f"  {ab:35} {mean:.4f} (OOF-Val: {gap:+.4f})")

        print(f"\nPer-Species (OOF):")
        for species_name, metrics in r.per_species.items():
            mean = metrics.get('mean')
            std = metrics.get('std')
            marker = " (monitoring)" if species_name == "K.pneumoniae" else ""
            if mean is not None:
                std_str = f" +/- {std:.4f}" if std is not None else ""
                print(f"  {species_name:15} {mean:.4f}{std_str}{marker}")
            else:
                print(f"  {species_name:15} N/A{marker}")

        # Phase 5.5: Show validation metrics per species
        if r.val_per_species:
            print(f"\nPer-Species (Val):")
            for species_name, metrics in r.val_per_species.items():
                mean = metrics.get('mean')
                oof_mean = r.per_species.get(species_name, {}).get('mean')
                marker = " (monitoring)" if species_name == "K.pneumoniae" else ""
                if mean is not None and oof_mean is not None:
                    gap = mean - oof_mean
                    print(f"  {species_name:15} {mean:.4f} (OOF-Val: {gap:+.4f}){marker}")
                elif mean is not None:
                    print(f"  {species_name:15} {mean:.4f}{marker}")

    def save_results(self, path: Path):
        """Save results to JSON file."""
        if self.result is None:
            raise ValueError("No results to save. Run the experiment first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.result.to_dict(), f, indent=2, default=str)

        print(f"Results saved to: {path}")

    def save_predictions(self, path: Path) -> None:
        """Save OOF and test predictions to npz file.

        Args:
            path: Path to save the npz file (e.g., "predictions/pls_n20.npz")
        """
        if self.result is None:
            raise ValueError("No results to save. Run the experiment first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            path,
            oof_predictions=self.result.oof_predictions,
            test_predictions=self.result.test_predictions,
            timestamp=self.result.timestamp,
            name=self.result.name
        )
        print(f"Predictions saved to: {path}")

    @classmethod
    def load_predictions(cls, path: Path) -> Dict[str, Any]:
        """Load predictions from npz file.

        Args:
            path: Path to the npz file (e.g., "predictions/pls_n20.npz")

        Returns:
            Dict with keys: 'oof_predictions', 'test_predictions', 'name', 'timestamp'
        """
        data = np.load(path, allow_pickle=True)
        return {
            'oof_predictions': data['oof_predictions'],
            'test_predictions': data['test_predictions'],
            'name': str(data['name']),
            'timestamp': str(data['timestamp'])
        }


if __name__ == "__main__":
    # Quick test
    exp = ControlledExperiment(name="test_baseline")
    results = exp.run()
    exp.print_summary()
