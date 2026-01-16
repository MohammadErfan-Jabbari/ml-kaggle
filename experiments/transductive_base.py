#!/usr/bin/env python3
"""
TRANSDUCTIVE DIMENSIONALITY REDUCTION + SELF-TRAINING BASE MODULE

Core module for combining unsupervised DR (PCA/KernelPCA/PPCA) trained on ALL data
(train+val+test) with self-training for AMR prediction.

Key Innovation:
- Unsupervised DR on train+val+test learns joint distribution → fixes covariate shift
- Self-training further adapts to test distribution with pseudo-labels
- Ensemble of LightGBM, XGBoost, CatBoost, MLP on reduced features

Target: Beat LB 0.83862
"""

import sys
import os
import warnings
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
import json
import pickle
import multiprocessing
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.neural_network import MLPClassifier
from scipy.stats import rankdata
import lightgbm as lgb

# Rich for beautiful terminal output
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.panel import Panel
from rich.text import Text
from rich import box

# Initialize console
console = Console()

warnings.filterwarnings("ignore")

# Try importing optional libraries
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    console.print("[yellow]⚠ XGBoost not installed, will skip XGB models[/]")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    console.print("[yellow]⚠ CatBoost not installed, will skip CatBoost models[/]")

try:
    from sklearn.decomposition import KernelPCA
    HAS_KPCA = True
except ImportError:
    HAS_KPCA = False

# Check for GPU availability
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    N_GPUS = torch.cuda.device_count() if HAS_CUDA else 0
except ImportError:
    HAS_CUDA = False
    N_GPUS = 0

# CPU cores for parallel processing
N_CPU = min(multiprocessing.cpu_count(), 32)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data.dataset import (
    ANTIBIOTICS,
    load_validation_split,
    create_test_distribution_split,
    load_train_data,
    split_features_targets
)

# =============================================================================
# CONSTANTS
# =============================================================================

DATA_DIR = Path("/home/erfan/data/ml_kaggle/raw")
BASE_OUTPUT_DIR = Path("/sata_disk/users/erfan/ml_kaggle/outputs")

SPECIES_NAMES = {0: "E.coli", 1: "K.pneumoniae", 2: "P.mirabilis", 3: "P.aeruginosa"}

ANTIBIOTIC_SHORT = {
    "Ampicillin": "AMP",
    "Amoxicillin_Clavulanic_acid": "AMC",
    "Cefotaxime": "CTX",
    "Cefuroxime": "CXM",
    "Ciprofloxacin": "CIP",
    "Ertapenem": "ETP",
    "Imipenem": "IPM",
    "Levofloxacin": "LVX"
}

N_FOLDS = 5
RANDOM_STATE = 42

# Match test distribution for sample weighting
SPECIES_WEIGHTS = {
    0: 1.5,   # E.coli
    1: 2.0,   # K.pneumoniae
    2: 1.5,   # P.mirabilis
    3: 0.1,   # P.aeruginosa
}

# Intrinsic resistance rules
INTRINSIC_RESISTANCE = {
    3: ["Ampicillin", "Amoxicillin_Clavulanic_acid", "Ertapenem", "Cefotaxime", "Cefuroxime"],  # P. aeruginosa
    2: ["Imipenem"],  # P. mirabilis
}


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TransductiveConfig:
    """Configuration for transductive DR + self-training pipeline."""

    # DR settings
    dr_method: str = "pca"  # 'pca', 'kpca', 'ppca'
    n_components: int = 100
    dr_params: dict = None  # Extra params, e.g., {'kernel': 'rbf'} for kpca

    # Self-training settings
    max_iterations: int = 5
    confidence_threshold_high: float = 0.85
    confidence_threshold_low: float = 0.15
    pseudo_label_weight: float = 0.5
    min_new_labels_per_iter: int = 10

    # Ensemble settings
    use_lgb: bool = True
    use_xgb: bool = True
    use_catboost: bool = True
    use_mlp: bool = True

    # Data paths
    data_dir: Path = DATA_DIR
    output_base: Path = BASE_OUTPUT_DIR

    # Runtime
    n_folds: int = N_FOLDS
    n_jobs: int = None
    random_state: int = RANDOM_STATE
    smoke_test: bool = False

    # LightGBM params
    lgb_params: dict = None

    # MLP params
    mlp_params: dict = None

    def __post_init__(self):
        if self.n_jobs is None:
            self.n_jobs = N_CPU
        if self.dr_params is None:
            self.dr_params = {}
        if self.lgb_params is None:
            self.lgb_params = {
                "n_estimators": 200,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_child_samples": 15,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
            }
        if self.mlp_params is None:
            self.mlp_params = {
                "hidden_layer_sizes": (256, 128),
                "activation": "relu",
                "alpha": 0.01,
                "learning_rate_init": 0.001,
                "max_iter": 500,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "n_iter_no_change": 10,
            }


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def print_banner(dr_method: str):
    """Print startup banner."""
    method_upper = dr_method.upper()
    banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ████████╗██████╗  █████╗ ███╗   ██╗███████╗██████╗ ██╗   ██╗ ██████╗       ║
║   ╚══██╔══╝██╔══██╗██╔══██╗████╗  ██║██╔════╝██╔══██╗██║   ██║██╔════╝       ║
║      ██║   ██████╔╝███████║██╔██╗ ██║███████╗██║  ██║██║   ██║██║            ║
║      ██║   ██╔══██╗██╔══██║██║╚██╗██║╚════██║██║  ██║██║   ██║██║            ║
║      ██║   ██║  ██║██║  ██║██║ ╚████║███████║██████╔╝╚██████╔╝╚██████╗       ║
║      ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚═════╝  ╚═════╝  ╚═════╝       ║
║                                                                              ║
║         Transductive {method_upper} + Self-Training for AMR Prediction             ║
║         Target: Beat LB 0.83862 with Val Mean AUC                            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    console.print(banner, style="cyan")


# =============================================================================
# DATA LOADING AND TRANSDUCTIVE DR
# =============================================================================

def load_raw_data(config: TransductiveConfig):
    """Load raw train and test data."""
    console.print("\n[bold cyan]📂 Loading Data...[/]")

    with console.status("[bold green]Reading CSV files...") as status:
        train_df = load_train_data()
        X_full, y_full, metadata = split_features_targets(train_df)
        species_full = metadata[:, 1].astype(np.int32)

        test_df = pd.read_csv(config.data_dir / "test.csv")
        feature_cols = [f"maldi_feature_{i}" for i in range(6000)]
        X_test = test_df[feature_cols].values.astype(np.float32)
        species_test = test_df["species_id"].values.astype(np.int32)
        sample_ids = test_df["sample_id"].values

    console.print(f"  ✓ Train (full): {X_full.shape[0]:,} samples, {X_full.shape[1]:,} features")
    console.print(f"  ✓ Test: {X_test.shape[0]:,} samples")

    return X_full, y_full, species_full, X_test, species_test, sample_ids


def create_validation_split(X_full, y_full, species_full, config: TransductiveConfig):
    """Create train/val split matching test species distribution."""
    try:
        X_train, X_val, y_train, y_val, species_train, species_val = load_validation_split()
        console.print("  ✓ Loaded existing validation split")
    except FileNotFoundError:
        console.print("  [yellow]Creating new validation split matching test distribution...[/]")
        X_train, X_val, y_train, y_val, species_train, species_val = create_test_distribution_split(
            X_full, y_full, species_full, val_size=0.2, random_state=config.random_state
        )

    console.print(f"  ✓ Train: {X_train.shape[0]:,}, Val: {X_val.shape[0]:,}")

    return X_train, X_val, y_train, y_val, species_train, species_val


def perform_transductive_dr(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    config: TransductiveConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, object, object, np.ndarray]:
    """
    Perform transductive dimensionality reduction on ALL data.

    The key innovation: fit DR on train+val+test combined,
    then split back. This learns the joint distribution.

    Returns:
        X_train_dr, X_val_dr, X_test_dr, dr_model, scaler, non_constant_mask
    """
    console.print(f"\n[bold cyan]🔬 Transductive {config.dr_method.upper()}...[/]")

    n_train = len(X_train)
    n_val = len(X_val)
    n_test = len(X_test)

    # 1. Combine ALL data
    X_all = np.vstack([X_train, X_val, X_test])
    console.print(f"  ✓ Combined: {X_all.shape[0]:,} samples × {X_all.shape[1]:,} features")

    # 2. Remove constant features (computed on ALL data)
    variance = np.var(X_all, axis=0)
    non_constant_mask = variance > 1e-8
    n_removed = (~non_constant_mask).sum()
    X_all_clean = X_all[:, non_constant_mask]
    console.print(f"  ✓ Removed {n_removed:,} constant features, keeping {non_constant_mask.sum():,}")

    # 3. StandardScaler on ALL data (critical for PCA/MLP)
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all_clean)
    console.print(f"  ✓ Standardized: mean={X_all_scaled.mean():.6f}, std={X_all_scaled.std():.6f}")

    # 4. Apply DR method
    dr_start = time.time()

    n_components = min(config.n_components, X_all_scaled.shape[1], X_all_scaled.shape[0] - 1)

    if config.dr_method == 'pca':
        dr_model = PCA(n_components=n_components, random_state=config.random_state)
        X_all_dr = dr_model.fit_transform(X_all_scaled)
        explained_var = dr_model.explained_variance_ratio_.sum()
        console.print(f"  ✓ PCA: {n_components} components, explained variance: {explained_var:.4f}")

    elif config.dr_method == 'kpca':
        if not HAS_KPCA:
            raise ImportError("KernelPCA not available")

        kernel = config.dr_params.get('kernel', 'rbf')
        gamma_param = config.dr_params.get('gamma', None)

        # KernelPCA doesn't accept 'scale' - compute it manually
        if gamma_param == 'scale':
            # gamma = 1 / (n_features * X.var())
            gamma = 1.0 / (X_all_scaled.shape[1] * X_all_scaled.var())
        elif gamma_param == 'auto':
            gamma = 1.0 / X_all_scaled.shape[1]
        else:
            gamma = gamma_param  # None or float

        dr_model = KernelPCA(
            n_components=n_components,
            kernel=kernel,
            gamma=gamma,
            fit_inverse_transform=False,
            n_jobs=config.n_jobs,
            random_state=config.random_state
        )
        X_all_dr = dr_model.fit_transform(X_all_scaled)
        console.print(f"  ✓ Kernel PCA: {n_components} components, kernel={kernel}, gamma={gamma}")

    elif config.dr_method == 'ppca':
        # Use FactorAnalysis for Probabilistic PCA
        dr_model = FactorAnalysis(
            n_components=n_components,
            random_state=config.random_state
        )
        X_all_dr = dr_model.fit_transform(X_all_scaled)
        noise_var = dr_model.noise_variance_.mean()
        console.print(f"  ✓ Probabilistic PCA (FA): {n_components} components, noise_var: {noise_var:.6f}")

    else:
        raise ValueError(f"Unknown DR method: {config.dr_method}")

    dr_time = time.time() - dr_start
    console.print(f"  ✓ DR completed in {dr_time:.1f}s")

    # 5. Split back
    X_train_dr = X_all_dr[:n_train]
    X_val_dr = X_all_dr[n_train:n_train+n_val]
    X_test_dr = X_all_dr[n_train+n_val:]

    console.print(f"  ✓ Split back: train={X_train_dr.shape}, val={X_val_dr.shape}, test={X_test_dr.shape}")

    return X_train_dr, X_val_dr, X_test_dr, dr_model, scaler, non_constant_mask


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_sample_weights(species: np.ndarray, is_pseudo: np.ndarray = None, pseudo_weight: float = 0.5) -> np.ndarray:
    """Get sample weights based on species and pseudo-label status."""
    weights = np.array([SPECIES_WEIGHTS.get(s, 1.0) for s in species])
    if is_pseudo is not None:
        weights[is_pseudo] *= pseudo_weight
    return weights


def apply_intrinsic_rules(predictions: np.ndarray, species: np.ndarray, antibiotic: str = None) -> np.ndarray:
    """Apply biological intrinsic resistance rules."""
    predictions = predictions.copy()

    if antibiotic is not None:
        # Single antibiotic mode
        for species_id, resistant_antibiotics in INTRINSIC_RESISTANCE.items():
            if antibiotic in resistant_antibiotics:
                species_mask = (species == species_id)
                predictions[species_mask] = 1.0
    else:
        # Full matrix mode (n_samples, n_antibiotics)
        antibiotic_indices = {ab: idx for idx, ab in enumerate(ANTIBIOTICS)}

        for species_id, resistant_antibiotics in INTRINSIC_RESISTANCE.items():
            species_mask = (species == species_id)
            for ab in resistant_antibiotics:
                if ab in antibiotic_indices:
                    predictions[species_mask, antibiotic_indices[ab]] = 1.0

    return predictions


def compute_val_mean_auc(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Compute Val Mean AUC - THE LEADERBOARD METRIC."""
    antibiotic_aucs = []
    per_ab_aucs = {}

    for idx, antibiotic in enumerate(ANTIBIOTICS):
        mask = ~np.isnan(y_true[:, idx])
        if mask.sum() > 10 and len(np.unique(y_true[mask, idx])) > 1:
            auc = roc_auc_score(y_true[mask, idx], y_pred[mask, idx])
            per_ab_aucs[antibiotic] = auc
            antibiotic_aucs.append(auc)
        else:
            per_ab_aucs[antibiotic] = np.nan

    mean_auc = np.mean(antibiotic_aucs) if antibiotic_aucs else 0.0
    return mean_auc, per_ab_aucs


def rank_average_predictions(pred_list: List[np.ndarray]) -> np.ndarray:
    """Rank average multiple prediction arrays."""
    if len(pred_list) == 1:
        return pred_list[0]

    n_samples = len(pred_list[0])
    ranks = [rankdata(p) / n_samples for p in pred_list]
    return np.mean(ranks, axis=0)


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_ensemble_for_antibiotic(
    antibiotic_idx: int,
    antibiotic: str,
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    species_labeled: np.ndarray,
    is_pseudo_labeled: np.ndarray,
    X_unlabeled: np.ndarray,
    species_unlabeled: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    species_val: np.ndarray,
    X_test: np.ndarray,
    species_test: np.ndarray,
    config: TransductiveConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Train ensemble of models for a single antibiotic.

    Returns:
        unlabeled_preds, val_preds, test_preds, val_auc
    """
    valid_mask = ~np.isnan(y_labeled)
    if valid_mask.sum() < 50:
        return (
            np.full(len(X_unlabeled), 0.5),
            np.full(len(X_val), 0.5),
            np.full(len(X_test), 0.5),
            0.5
        )

    X_train = X_labeled[valid_mask]
    y_train = y_labeled[valid_mask]
    species_train = species_labeled[valid_mask]
    is_pseudo_train = is_pseudo_labeled[valid_mask]

    if len(np.unique(y_train)) < 2:
        const = y_train[0]
        return (
            np.full(len(X_unlabeled), const),
            np.full(len(X_val), const),
            np.full(len(X_test), const),
            0.5
        )

    all_unlabeled_preds = []
    all_val_preds = []
    all_test_preds = []

    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_state)

    # LightGBM
    if config.use_lgb:
        lgb_unlabeled = []
        lgb_val = []
        lgb_test = []

        for train_idx, _ in skf.split(X_train, species_train):
            weights = get_sample_weights(species_train[train_idx], is_pseudo_train[train_idx], config.pseudo_label_weight)
            model = lgb.LGBMClassifier(
                random_state=config.random_state,
                verbose=-1,
                n_jobs=config.n_jobs,
                device='cpu',
                **config.lgb_params
            )
            model.fit(X_train[train_idx], y_train[train_idx], sample_weight=weights)
            lgb_unlabeled.append(model.predict_proba(X_unlabeled)[:, 1])
            lgb_val.append(model.predict_proba(X_val)[:, 1])
            lgb_test.append(model.predict_proba(X_test)[:, 1])

        all_unlabeled_preds.append(np.mean(lgb_unlabeled, axis=0))
        all_val_preds.append(np.mean(lgb_val, axis=0))
        all_test_preds.append(np.mean(lgb_test, axis=0))

    # XGBoost (CPU only to avoid GPU memory issues with parallel runs)
    if config.use_xgb and HAS_XGB:
        xgb_unlabeled = []
        xgb_val = []
        xgb_test = []

        xgb_params = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "random_state": config.random_state,
            "verbosity": 0,
            "n_jobs": config.n_jobs,
            "tree_method": "hist",  # Fast CPU histogram method
        }

        for train_idx, _ in skf.split(X_train, species_train):
            weights = get_sample_weights(species_train[train_idx], is_pseudo_train[train_idx], config.pseudo_label_weight)
            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X_train[train_idx], y_train[train_idx], sample_weight=weights)
            xgb_unlabeled.append(model.predict_proba(X_unlabeled)[:, 1])
            xgb_val.append(model.predict_proba(X_val)[:, 1])
            xgb_test.append(model.predict_proba(X_test)[:, 1])

        all_unlabeled_preds.append(np.mean(xgb_unlabeled, axis=0))
        all_val_preds.append(np.mean(xgb_val, axis=0))
        all_test_preds.append(np.mean(xgb_test, axis=0))

    # CatBoost (CPU only to avoid GPU memory issues with parallel runs)
    if config.use_catboost and HAS_CATBOOST:
        cat_unlabeled = []
        cat_val = []
        cat_test = []

        cat_params = {
            "iterations": 200,
            "learning_rate": 0.05,
            "depth": 6,
            "random_seed": config.random_state,
            "verbose": False,
            "thread_count": config.n_jobs,
        }

        for train_idx, _ in skf.split(X_train, species_train):
            weights = get_sample_weights(species_train[train_idx], is_pseudo_train[train_idx], config.pseudo_label_weight)
            model = CatBoostClassifier(**cat_params)
            model.fit(X_train[train_idx], y_train[train_idx], sample_weight=weights)
            cat_unlabeled.append(model.predict_proba(X_unlabeled)[:, 1])
            cat_val.append(model.predict_proba(X_val)[:, 1])
            cat_test.append(model.predict_proba(X_test)[:, 1])

        all_unlabeled_preds.append(np.mean(cat_unlabeled, axis=0))
        all_val_preds.append(np.mean(cat_val, axis=0))
        all_test_preds.append(np.mean(cat_test, axis=0))

    # MLP (works well on reduced PCA features)
    if config.use_mlp:
        mlp_unlabeled = []
        mlp_val = []
        mlp_test = []

        for train_idx, _ in skf.split(X_train, species_train):
            model = MLPClassifier(
                random_state=config.random_state,
                **config.mlp_params
            )
            model.fit(X_train[train_idx], y_train[train_idx].astype(int))
            mlp_unlabeled.append(model.predict_proba(X_unlabeled)[:, 1])
            mlp_val.append(model.predict_proba(X_val)[:, 1])
            mlp_test.append(model.predict_proba(X_test)[:, 1])

        all_unlabeled_preds.append(np.mean(mlp_unlabeled, axis=0))
        all_val_preds.append(np.mean(mlp_val, axis=0))
        all_test_preds.append(np.mean(mlp_test, axis=0))

    # Rank average ensemble
    if len(all_unlabeled_preds) > 0:
        unlabeled_preds = rank_average_predictions(all_unlabeled_preds)
        val_preds = rank_average_predictions(all_val_preds)
        test_preds = rank_average_predictions(all_test_preds)
    else:
        unlabeled_preds = np.full(len(X_unlabeled), 0.5)
        val_preds = np.full(len(X_val), 0.5)
        test_preds = np.full(len(X_test), 0.5)

    # Apply intrinsic rules
    val_preds = apply_intrinsic_rules(val_preds, species_val, antibiotic)
    test_preds = apply_intrinsic_rules(test_preds, species_test, antibiotic)

    # Compute val AUC
    val_labels = y_val[:, antibiotic_idx] if y_val.ndim > 1 else y_val
    val_mask = ~np.isnan(val_labels)

    if val_mask.sum() > 10 and len(np.unique(val_labels[val_mask])) > 1:
        val_auc = roc_auc_score(val_labels[val_mask], val_preds[val_mask])
    else:
        val_auc = 0.5

    return unlabeled_preds, val_preds, test_preds, val_auc


def select_pseudo_labels(
    predictions: np.ndarray,
    current_pseudo_mask: np.ndarray,
    config: TransductiveConfig
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Select new pseudo-labels based on confidence thresholds."""
    new_labels = np.full(len(predictions), np.nan)
    new_pseudo_mask = current_pseudo_mask.copy()

    high_conf_pos = (predictions >= config.confidence_threshold_high) & (~current_pseudo_mask)
    new_labels[high_conf_pos] = 1.0
    new_pseudo_mask[high_conf_pos] = True

    high_conf_neg = (predictions <= config.confidence_threshold_low) & (~current_pseudo_mask)
    new_labels[high_conf_neg] = 0.0
    new_pseudo_mask[high_conf_neg] = True

    n_new = high_conf_pos.sum() + high_conf_neg.sum()

    return new_labels, new_pseudo_mask, n_new


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_transductive_pipeline(config: TransductiveConfig) -> Dict:
    """
    Run the full transductive DR + self-training pipeline.

    Returns:
        Dictionary with results, paths, and metrics
    """
    start_time = datetime.now()

    # Print banner
    print_banner(config.dr_method)

    # Create run directory
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{config.dr_method}_{config.n_components}_{run_timestamp}"
    run_dir = config.output_base / "transductive_dr_runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold green]📁 Run Directory:[/] {run_dir}")
    console.print(f"[bold green]🖥️  Hardware:[/] {N_GPUS} GPUs, {N_CPU} CPU cores, CUDA: {HAS_CUDA}")
    console.print(f"[bold green]⚙️  Config:[/] {config.dr_method.upper()} n={config.n_components}, max_iter={config.max_iterations}")
    console.print(f"[bold green]🎯 Target:[/] Beat LB 0.83862 with Val Mean AUC")

    # Load raw data
    X_full, y_full, species_full, X_test_raw, species_test, sample_ids = load_raw_data(config)

    # Create validation split
    X_train_raw, X_val_raw, y_train, y_val, species_train, species_val = create_validation_split(
        X_full, y_full, species_full, config
    )

    # Show validation species distribution
    console.print("\n[bold]Validation Species Distribution:[/]")
    dist_table = Table(box=box.SIMPLE)
    dist_table.add_column("Species", style="cyan")
    dist_table.add_column("Val %", justify="right")
    dist_table.add_column("Target %", justify="right")

    target_pcts = {0: 27, 1: 51, 2: 19, 3: 3}
    for s_id, s_name in SPECIES_NAMES.items():
        pct = (species_val == s_id).sum() / len(species_val) * 100
        dist_table.add_row(s_name, f"{pct:.1f}%", f"{target_pcts[s_id]}%")

    console.print(dist_table)

    # Perform transductive DR on ALL data
    X_train_dr, X_val_dr, X_test_dr, dr_model, scaler, non_constant_mask = perform_transductive_dr(
        X_train_raw, X_val_raw, X_test_raw, config
    )

    # Storage for results
    final_val_preds = np.zeros((len(X_val_dr), len(ANTIBIOTICS)))
    final_test_preds = np.zeros((len(X_test_dr), len(ANTIBIOTICS)))
    antibiotic_results = {}

    console.print("\n")
    console.rule("[bold cyan]Starting Self-Training Loop[/]")
    console.print("\n")

    # Process each antibiotic
    for ab_idx, antibiotic in enumerate(ANTIBIOTICS):
        ab_start_time = time.time()
        short = ANTIBIOTIC_SHORT[antibiotic]

        console.print(f"\n[bold cyan]{'═' * 60}[/]")
        console.print(f"[bold cyan]  {short}: {antibiotic} ({ab_idx + 1}/{len(ANTIBIOTICS)})[/]")
        console.print(f"[bold cyan]{'═' * 60}[/]")

        # Initialize labeled pool
        X_labeled = X_train_dr.copy()
        y_labeled = y_train[:, ab_idx].copy()
        species_labeled = species_train.copy()
        is_pseudo_labeled = np.zeros(len(X_labeled), dtype=bool)

        # Initialize unlabeled pool (test samples)
        X_unlabeled = X_test_dr.copy()
        species_unlabeled = species_test.copy()
        unlabeled_pseudo_mask = np.zeros(len(X_unlabeled), dtype=bool)
        unlabeled_pseudo_labels = np.full(len(X_unlabeled), np.nan)

        n_labeled_initial = (~np.isnan(y_labeled)).sum()
        console.print(f"  📊 Initial: {n_labeled_initial:,} labeled, {len(X_unlabeled):,} unlabeled")

        # Self-training iterations
        best_val_auc = 0.0
        initial_val_auc = 0.0
        best_val_preds = None
        best_test_preds = None

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:

            task = progress.add_task(f"[cyan]Training {short}...", total=config.max_iterations)

            for iteration in range(config.max_iterations):
                # Train ensemble and predict
                unlabeled_preds, val_preds, test_preds, val_auc = train_ensemble_for_antibiotic(
                    antibiotic_idx=ab_idx,
                    antibiotic=antibiotic,
                    X_labeled=X_labeled,
                    y_labeled=y_labeled,
                    species_labeled=species_labeled,
                    is_pseudo_labeled=is_pseudo_labeled,
                    X_unlabeled=X_unlabeled,
                    species_unlabeled=species_unlabeled,
                    X_val=X_val_dr,
                    y_val=y_val,
                    species_val=species_val,
                    X_test=X_test_dr,
                    species_test=species_test,
                    config=config
                )

                if iteration == 0:
                    initial_val_auc = val_auc

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_val_preds = val_preds.copy()
                    best_test_preds = test_preds.copy()

                # Select new pseudo-labels
                new_labels, unlabeled_pseudo_mask, n_new = select_pseudo_labels(
                    unlabeled_preds, unlabeled_pseudo_mask, config
                )

                progress.update(task, advance=1, description=f"[cyan]{short} Iter {iteration+1}: AUC={val_auc:.4f}, +{n_new} pseudo")

                # Check stopping condition
                if n_new < config.min_new_labels_per_iter:
                    console.print(f"  ⏹️  Stopping early: only {n_new} new labels")
                    break

                # Add new pseudo-labels to training pool
                if n_new > 0:
                    new_indices = np.where(~np.isnan(new_labels) & (unlabeled_pseudo_labels != new_labels))[0]
                    if len(new_indices) > 0:
                        X_labeled = np.vstack([X_labeled, X_unlabeled[new_indices]])
                        y_labeled = np.concatenate([y_labeled, new_labels[new_indices]])
                        species_labeled = np.concatenate([species_labeled, species_unlabeled[new_indices]])
                        is_pseudo_labeled = np.concatenate([is_pseudo_labeled, np.ones(len(new_indices), dtype=bool)])
                        unlabeled_pseudo_labels[new_indices] = new_labels[new_indices]

        # Store results
        ab_time = time.time() - ab_start_time
        auc_improvement = best_val_auc - initial_val_auc

        antibiotic_results[antibiotic] = {
            'best_auc': best_val_auc,
            'initial_auc': initial_val_auc,
            'auc_improvement': auc_improvement,
            'total_pseudo': int(unlabeled_pseudo_mask.sum()),
            'time': ab_time
        }

        final_val_preds[:, ab_idx] = best_val_preds if best_val_preds is not None else val_preds
        final_test_preds[:, ab_idx] = best_test_preds if best_test_preds is not None else test_preds

        improvement_style = "green" if auc_improvement > 0 else "red"
        console.print(f"  ✅ Best AUC: [bold]{best_val_auc:.4f}[/] (Δ [{improvement_style}]{auc_improvement:+.4f}[/]) • {unlabeled_pseudo_mask.sum():,} pseudo-labels • {ab_time:.1f}s")

    # Final results
    console.print("\n")
    console.rule("[bold green]Training Complete[/]")
    console.print("\n")

    # Create summary table
    summary_table = Table(title="[bold]Final Results by Antibiotic[/]", box=box.DOUBLE_EDGE)
    summary_table.add_column("Antibiotic", style="cyan")
    summary_table.add_column("Initial AUC", justify="right")
    summary_table.add_column("Best AUC", justify="right", style="green")
    summary_table.add_column("Improvement", justify="right")
    summary_table.add_column("Pseudo Labels", justify="right")

    for ab in ANTIBIOTICS:
        result = antibiotic_results[ab]
        delta = result['auc_improvement']
        delta_style = "green" if delta > 0 else "red"
        summary_table.add_row(
            ANTIBIOTIC_SHORT[ab],
            f"{result['initial_auc']:.4f}",
            f"{result['best_auc']:.4f}",
            Text(f"{delta:+.4f}", style=delta_style),
            f"{result['total_pseudo']:,}"
        )

    console.print(summary_table)

    # Compute final Val Mean AUC
    mean_auc, per_ab_aucs = compute_val_mean_auc(y_val, final_val_preds)

    console.print(f"\n[bold]📊 VAL MEAN AUC: {mean_auc:.4f}[/]")
    console.print(f"[bold]🎯 TARGET LB: 0.83862[/]")

    diff = mean_auc - 0.83862
    if diff > 0:
        console.print(f"[bold green]✅ Above target by {diff:.4f}![/]")
    else:
        console.print(f"[bold yellow]📉 Below target by {abs(diff):.4f}[/]")

    # Save outputs
    console.print("\n[bold cyan]💾 Saving outputs...[/]")

    # Submissions
    submissions_dir = run_dir / "submissions"
    submissions_dir.mkdir(exist_ok=True)

    submission = pd.DataFrame({
        "sample_id": sample_ids,
        **{ab: final_test_preds[:, idx] for idx, ab in enumerate(ANTIBIOTICS)}
    })
    submission_path = submissions_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    console.print(f"  ✓ Submission: {submission_path}")

    # Predictions
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)

    np.save(predictions_dir / "final_val_preds.npy", final_val_preds)
    np.save(predictions_dir / "final_test_preds.npy", final_test_preds)
    console.print(f"  ✓ Predictions: {predictions_dir}")

    # Config
    config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}
    with open(run_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Artifacts
    artifacts = {
        'config': config_dict,
        'dr_model': dr_model,
        'scaler': scaler,
        'non_constant_mask': non_constant_mask,
        'final_val_preds': final_val_preds,
        'final_test_preds': final_test_preds,
        'antibiotic_results': antibiotic_results,
        'mean_auc': mean_auc,
        'per_ab_aucs': per_ab_aucs,
        'timestamp': datetime.now().isoformat(),
    }
    with open(run_dir / "artifacts.pkl", 'wb') as f:
        pickle.dump(artifacts, f)
    console.print(f"  ✓ Artifacts: {run_dir}/artifacts.pkl")

    # Results JSON
    results = {
        'run_dir': str(run_dir),
        'dr_method': config.dr_method,
        'n_components': config.n_components,
        'timestamp': datetime.now().isoformat(),
        'elapsed_time': str(datetime.now() - start_time).split('.')[0],
        'val_mean_auc': float(mean_auc),
        'per_antibiotic': {ab: float(v) for ab, v in per_ab_aucs.items() if not np.isnan(v)},
        'antibiotic_results': antibiotic_results,
    }

    with open(run_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    console.print(f"  ✓ Results: {run_dir}/results.json")

    # Final banner
    console.print(f"\n[bold green]{'═' * 60}[/]")
    console.print(f"[bold green]  ✅ COMPLETE! Elapsed: {str(datetime.now() - start_time).split('.')[0]}[/]")
    console.print(f"[bold green]  📁 All outputs in: {run_dir}[/]")
    console.print(f"[bold green]{'═' * 60}[/]")

    console.print(f"\n[bold]Kaggle submission command:[/]")
    console.print(f"[dim]kaggle competitions submit -c antimicrobial-resistance-prediction-from-maldi-tof -f {submission_path} -m 'Transductive {config.dr_method.upper()} + Self-Training'[/]")

    return results
