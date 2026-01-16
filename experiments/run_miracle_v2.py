#!/usr/bin/env python3
"""
MIRACLE BLEND v2: Using proper validation split and Val Mean AUC metric.

Key Changes from v1:
1. Uses validation split that matches test species distribution
2. Evaluates with VAL MEAN AUC (average across 8 antibiotics) - the LB metric
3. Trains on train subset only, validates on held-out val set
4. No OOF cheating - true held-out validation

Strategy:
- Maximum model diversity (LightGBM, XGBoost, CatBoost, MLP, PLS variants)
- Multiple hyperparameter configurations per model type
- Rank averaging (proven best ensemble method)
- Species-aware sample weighting
- GPU acceleration for tree models (L40s)
- Beautiful Rich terminal output

Target: Beat LB 0.83862
"""

import sys
import os
import warnings
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from datetime import datetime, timedelta
import json
import pickle
import multiprocessing
import traceback
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
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
from rich.live import Live

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

# Check for GPU availability
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    N_GPUS = torch.cuda.device_count() if HAS_CUDA else 0
except ImportError:
    HAS_CUDA = False
    N_GPUS = 0

# CPU cores for parallel processing - cap at 32 to avoid overload
N_CPU = min(multiprocessing.cpu_count(), 32)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data.dataset import (
    ANTIBIOTICS,
    load_validation_split,
    create_test_distribution_split,
    load_train_data,
    split_features_targets
)

DATA_DIR = Path("/home/erfan/data/ml_kaggle/raw")
BASE_OUTPUT_DIR = Path("/sata_disk/users/erfan/ml_kaggle/outputs")

SPECIES_NAMES = {0: "E.coli", 1: "K.pneumoniae", 2: "P.mirabilis", 3: "P.aeruginosa"}

# Short names for display
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
    0: 1.5,   # E.coli: upweight (27% test vs 17% train)
    1: 2.0,   # K.pneumoniae: upweight strongly (51% test vs 28% train)
    2: 1.5,   # P.mirabilis: upweight (19% test vs 12% train)
    3: 0.1,   # P.aeruginosa: downweight strongly (3% test vs 43% train)
}


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def print_banner():
    """Print startup banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ███╗   ███╗██╗██████╗  █████╗  ██████╗██╗     ███████╗                    ║
║   ████╗ ████║██║██╔══██╗██╔══██╗██╔════╝██║     ██╔════╝                    ║
║   ██╔████╔██║██║██████╔╝███████║██║     ██║     █████╗                      ║
║   ██║╚██╔╝██║██║██╔══██╗██╔══██║██║     ██║     ██╔══╝                      ║
║   ██║ ╚═╝ ██║██║██║  ██║██║  ██║╚██████╗███████╗███████╗                    ║
║   ╚═╝     ╚═╝╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚══════╝╚══════╝                    ║
║                                                                              ║
║              BLEND v2: Maximum Model Diversity Ensemble                      ║
║              Target: Beat LB 0.83862 with Val Mean AUC                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    console.print(banner, style="cyan")


def create_model_progress_table(model_results: Dict, current_model: str = None) -> Table:
    """Create table showing progress for all models."""
    table = Table(title="Model Training Progress", box=box.ROUNDED, show_header=True, header_style="bold magenta")

    table.add_column("Model", style="cyan", width=15)
    table.add_column("Status", justify="center", width=12)
    table.add_column("Val Mean AUC", justify="right", width=12)
    table.add_column("Time", justify="right", width=10)

    for name, result in model_results.items():
        if result is None:
            if name == current_model:
                status = Text("🔄 Training", style="yellow")
                auc = "-"
                time_str = "-"
            else:
                status = Text("⏳ Pending", style="dim")
                auc = "-"
                time_str = "-"
        elif result.get('failed'):
            status = Text("❌ Failed", style="red")
            auc = "-"
            time_str = f"{result.get('time', 0):.1f}s"
        else:
            status = Text("✅ Done", style="green")
            auc = f"{result.get('val_mean_auc', 0):.4f}"
            time_str = f"{result.get('time', 0):.1f}s"

        table.add_row(name, status, auc, time_str)

    return table


def create_ensemble_table(ensemble_results: Dict) -> Table:
    """Create table showing ensemble results."""
    table = Table(title="Ensemble Results", box=box.DOUBLE_EDGE, show_header=True, header_style="bold green")

    table.add_column("Ensemble", style="cyan", width=20)
    table.add_column("Val Mean AUC", justify="right", width=12)
    table.add_column("Rank", justify="center", width=6)

    sorted_results = sorted(ensemble_results.items(), key=lambda x: x[1], reverse=True)

    for rank, (name, auc) in enumerate(sorted_results, 1):
        style = "bold green" if rank == 1 else ""
        marker = " ★" if rank == 1 else ""
        table.add_row(
            f"{name}{marker}",
            Text(f"{auc:.4f}", style=style),
            str(rank)
        )

    return table


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data_with_val_split():
    """Load data with proper validation split matching test distribution."""
    console.print("\n[bold cyan]📂 Loading Data...[/]")

    with console.status("[bold green]Reading CSV files...") as status:
        train_df = load_train_data()
        X_full, y_full, metadata = split_features_targets(train_df)
        species_full = metadata[:, 1].astype(np.int32)

        test_df = pd.read_csv(DATA_DIR / "test.csv")
        feature_cols = [f"maldi_feature_{i}" for i in range(6000)]
        X_test = test_df[feature_cols].values.astype(np.float32)
        species_test = test_df["species_id"].values.astype(np.int32)
        sample_ids = test_df["sample_id"].values

    # Try to load existing validation split
    try:
        X_train, X_val, y_train, y_val, species_train, species_val = load_validation_split()
        console.print("  ✓ Loaded existing validation split")
    except FileNotFoundError:
        console.print("  [yellow]Creating new validation split matching test distribution...[/]")
        X_train, X_val, y_train, y_val, species_train, species_val = create_test_distribution_split(
            X_full, y_full, species_full, val_size=0.2, random_state=RANDOM_STATE
        )

    console.print(f"  ✓ Train: {X_train.shape[0]:,} samples")
    console.print(f"  ✓ Val: {X_val.shape[0]:,} samples")
    console.print(f"  ✓ Test: {X_test.shape[0]:,} samples")

    return X_train, X_val, X_test, y_train, y_val, species_train, species_val, species_test, sample_ids


def remove_constant_features(X_train, X_val, X_test, threshold=1e-5):
    """Remove constant/near-constant features."""
    variances = X_train.var(axis=0)
    mask = variances > threshold
    n_removed = (~mask).sum()
    console.print(f"  ✓ Removed {n_removed:,} constant features, keeping {mask.sum():,}")
    return X_train[:, mask], X_val[:, mask], X_test[:, mask], mask


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


def compute_val_mean_auc(y_true, y_pred, species=None):
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


# =============================================================================
# MODEL BUILDERS
# =============================================================================

def build_lgb_model(X_train, y_train, species_train, X_val, X_test, params: dict, name: str):
    """Build LightGBM model."""
    val_preds = np.zeros((len(X_val), len(ANTIBIOTICS)))
    test_preds = np.zeros((len(X_test), len(ANTIBIOTICS)))

    for idx, antibiotic in enumerate(ANTIBIOTICS):
        label_mask = ~np.isnan(y_train[:, idx])
        if label_mask.sum() < 50:
            val_preds[:, idx] = 0.5
            test_preds[:, idx] = 0.5
            continue

        X_ab = X_train[label_mask]
        y_ab = y_train[label_mask, idx]
        species_ab = species_train[label_mask]

        if len(np.unique(y_ab)) < 2:
            val_preds[:, idx] = y_ab[0]
            test_preds[:, idx] = y_ab[0]
            continue

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_val_preds = []
        fold_test_preds = []

        for train_idx, _ in skf.split(X_ab, species_ab):
            weights = get_sample_weights(species_ab[train_idx])
            model = lgb.LGBMClassifier(
                random_state=RANDOM_STATE,
                verbose=-1,
                n_jobs=N_CPU,
                device='cpu',
                **params,
            )
            model.fit(X_ab[train_idx], y_ab[train_idx], sample_weight=weights)
            fold_val_preds.append(model.predict_proba(X_val)[:, 1])
            fold_test_preds.append(model.predict_proba(X_test)[:, 1])

        val_preds[:, idx] = np.mean(fold_val_preds, axis=0)
        test_preds[:, idx] = np.mean(fold_test_preds, axis=0)

    return val_preds, test_preds


def build_xgb_model(X_train, y_train, species_train, X_val, X_test, params: dict, name: str):
    """Build XGBoost model with GPU support."""
    if not HAS_XGB:
        return None, None

    val_preds = np.zeros((len(X_val), len(ANTIBIOTICS)))
    test_preds = np.zeros((len(X_test), len(ANTIBIOTICS)))

    gpu_params = {}
    if HAS_CUDA and N_GPUS > 0:
        gpu_params = {"tree_method": "hist", "device": "cuda"}

    for idx, antibiotic in enumerate(ANTIBIOTICS):
        label_mask = ~np.isnan(y_train[:, idx])
        if label_mask.sum() < 50:
            val_preds[:, idx] = 0.5
            test_preds[:, idx] = 0.5
            continue

        X_ab = X_train[label_mask]
        y_ab = y_train[label_mask, idx]
        species_ab = species_train[label_mask]

        if len(np.unique(y_ab)) < 2:
            val_preds[:, idx] = y_ab[0]
            test_preds[:, idx] = y_ab[0]
            continue

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_val_preds = []
        fold_test_preds = []

        for train_idx, _ in skf.split(X_ab, species_ab):
            weights = get_sample_weights(species_ab[train_idx])
            model = xgb.XGBClassifier(
                random_state=RANDOM_STATE,
                verbosity=0,
                n_jobs=N_CPU,
                **params,
                **gpu_params
            )
            model.fit(X_ab[train_idx], y_ab[train_idx], sample_weight=weights)
            fold_val_preds.append(model.predict_proba(X_val)[:, 1])
            fold_test_preds.append(model.predict_proba(X_test)[:, 1])

        val_preds[:, idx] = np.mean(fold_val_preds, axis=0)
        test_preds[:, idx] = np.mean(fold_test_preds, axis=0)

    return val_preds, test_preds


def build_catboost_model(X_train, y_train, species_train, X_val, X_test, params: dict, name: str):
    """Build CatBoost model with GPU support."""
    if not HAS_CATBOOST:
        return None, None

    val_preds = np.zeros((len(X_val), len(ANTIBIOTICS)))
    test_preds = np.zeros((len(X_test), len(ANTIBIOTICS)))

    gpu_params = {}
    if HAS_CUDA and N_GPUS > 0:
        gpu_params = {"task_type": "GPU", "devices": "0"}

    for idx, antibiotic in enumerate(ANTIBIOTICS):
        label_mask = ~np.isnan(y_train[:, idx])
        if label_mask.sum() < 50:
            val_preds[:, idx] = 0.5
            test_preds[:, idx] = 0.5
            continue

        X_ab = X_train[label_mask]
        y_ab = y_train[label_mask, idx]
        species_ab = species_train[label_mask]

        if len(np.unique(y_ab)) < 2:
            val_preds[:, idx] = y_ab[0]
            test_preds[:, idx] = y_ab[0]
            continue

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_val_preds = []
        fold_test_preds = []

        for train_idx, _ in skf.split(X_ab, species_ab):
            weights = get_sample_weights(species_ab[train_idx])
            model = CatBoostClassifier(
                random_seed=RANDOM_STATE,
                verbose=False,
                thread_count=N_CPU,
                **params,
                **gpu_params
            )
            model.fit(X_ab[train_idx], y_ab[train_idx], sample_weight=weights)
            fold_val_preds.append(model.predict_proba(X_val)[:, 1])
            fold_test_preds.append(model.predict_proba(X_test)[:, 1])

        val_preds[:, idx] = np.mean(fold_val_preds, axis=0)
        test_preds[:, idx] = np.mean(fold_test_preds, axis=0)

    return val_preds, test_preds


def build_mlp_model(X_train, y_train, species_train, X_val, X_test, params: dict, name: str):
    """Build MLP neural network model."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    val_preds = np.zeros((len(X_val), len(ANTIBIOTICS)))
    test_preds = np.zeros((len(X_test), len(ANTIBIOTICS)))

    for idx, antibiotic in enumerate(ANTIBIOTICS):
        label_mask = ~np.isnan(y_train[:, idx])
        if label_mask.sum() < 50:
            val_preds[:, idx] = 0.5
            test_preds[:, idx] = 0.5
            continue

        X_ab = X_train_scaled[label_mask]
        y_ab = y_train[label_mask, idx].astype(int)
        species_ab = species_train[label_mask]

        if len(np.unique(y_ab)) < 2:
            val_preds[:, idx] = y_ab[0]
            test_preds[:, idx] = y_ab[0]
            continue

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_val_preds = []
        fold_test_preds = []

        for train_idx, _ in skf.split(X_ab, species_ab):
            model = MLPClassifier(
                random_state=RANDOM_STATE,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                **params
            )
            model.fit(X_ab[train_idx], y_ab[train_idx])
            fold_val_preds.append(model.predict_proba(X_val_scaled)[:, 1])
            fold_test_preds.append(model.predict_proba(X_test_scaled)[:, 1])

        val_preds[:, idx] = np.mean(fold_val_preds, axis=0)
        test_preds[:, idx] = np.mean(fold_test_preds, axis=0)

    return val_preds, test_preds


def build_pls_lgb_model(X_train, y_train, species_train, X_val, X_test, n_components: int, name: str):
    """Build PLS + LightGBM pipeline."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    val_preds = np.zeros((len(X_val), len(ANTIBIOTICS)))
    test_preds = np.zeros((len(X_test), len(ANTIBIOTICS)))

    for idx, antibiotic in enumerate(ANTIBIOTICS):
        label_mask = ~np.isnan(y_train[:, idx])
        if label_mask.sum() < 50:
            val_preds[:, idx] = 0.5
            test_preds[:, idx] = 0.5
            continue

        X_ab = X_train_scaled[label_mask]
        y_ab = y_train[label_mask, idx]
        species_ab = species_train[label_mask]

        if len(np.unique(y_ab)) < 2:
            val_preds[:, idx] = y_ab[0]
            test_preds[:, idx] = y_ab[0]
            continue

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_val_preds = []
        fold_test_preds = []

        for train_idx, _ in skf.split(X_ab, species_ab):
            n_comp = min(n_components, len(train_idx) - 1)
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X_ab[train_idx], y_ab[train_idx])

            X_train_pls = pls.transform(X_ab[train_idx])
            X_val_pls = pls.transform(X_val_scaled)
            X_test_pls = pls.transform(X_test_scaled)

            weights = get_sample_weights(species_ab[train_idx])
            model = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                random_state=RANDOM_STATE,
                verbose=-1,
                device='cpu',
            )
            model.fit(X_train_pls, y_ab[train_idx], sample_weight=weights)
            fold_val_preds.append(model.predict_proba(X_val_pls)[:, 1])
            fold_test_preds.append(model.predict_proba(X_test_pls)[:, 1])

        val_preds[:, idx] = np.mean(fold_val_preds, axis=0)
        test_preds[:, idx] = np.mean(fold_test_preds, axis=0)

    return val_preds, test_preds


# =============================================================================
# ENSEMBLE METHODS
# =============================================================================

def rank_average(predictions_list: List[Tuple[np.ndarray, np.ndarray]], species_val, species_test):
    """Rank averaging of multiple model predictions."""
    n_val = predictions_list[0][0].shape[0]
    n_test = predictions_list[0][1].shape[0]
    n_antibiotics = predictions_list[0][0].shape[1]

    rank_val = np.zeros((n_val, n_antibiotics))
    rank_test = np.zeros((n_test, n_antibiotics))

    for val_pred, test_pred in predictions_list:
        for idx in range(n_antibiotics):
            rank_val[:, idx] += rankdata(val_pred[:, idx]) / n_val
            rank_test[:, idx] += rankdata(test_pred[:, idx]) / n_test

    n_models = len(predictions_list)
    rank_val /= n_models
    rank_test /= n_models

    rank_val = apply_intrinsic_rules(rank_val, species_val)
    rank_test = apply_intrinsic_rules(rank_test, species_test)

    return rank_val, rank_test


def weighted_average(predictions_list: List[Tuple[np.ndarray, np.ndarray]], weights: List[float],
                     species_val, species_test):
    """Weighted probability averaging."""
    weights = np.array(weights)
    weights = weights / weights.sum()

    val_avg = np.zeros_like(predictions_list[0][0])
    test_avg = np.zeros_like(predictions_list[0][1])

    for (val_pred, test_pred), w in zip(predictions_list, weights):
        val_avg += val_pred * w
        test_avg += test_pred * w

    val_avg = apply_intrinsic_rules(val_avg, species_val)
    test_avg = apply_intrinsic_rules(test_avg, species_test)

    return val_avg, test_avg


# =============================================================================
# MAIN
# =============================================================================

def miracle_blend_v2():
    """Create the miracle blend using proper validation."""
    start_time = datetime.now()

    # Print banner
    print_banner()

    # Create run directory
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = BASE_OUTPUT_DIR / "miracle_v2_runs" / f"run_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold green]📁 Run Directory:[/] {run_dir}")
    console.print(f"[bold green]🖥️  Hardware:[/] {N_GPUS} GPUs, {N_CPU} CPU cores, CUDA: {HAS_CUDA}")
    console.print(f"[bold green]🎯 Target:[/] Beat LB 0.83862 with Val Mean AUC")

    # Load data
    X_train, X_val, X_test, y_train, y_val, species_train, species_val, species_test, sample_ids = load_data_with_val_split()
    X_train, X_val, X_test, feature_mask = remove_constant_features(X_train, X_val, X_test)

    # Show val distribution
    console.print("\n[bold]Validation Species Distribution:[/]")
    dist_table = Table(box=box.SIMPLE)
    dist_table.add_column("Species", style="cyan")
    dist_table.add_column("Val %", justify="right")
    dist_table.add_column("Target %", justify="right")

    for s_id, s_name in SPECIES_NAMES.items():
        pct = (species_val == s_id).sum() / len(species_val) * 100
        target_pcts = {0: 27, 1: 51, 2: 19, 3: 3}
        dist_table.add_row(s_name, f"{pct:.1f}%", f"{target_pcts[s_id]}%")

    console.print(dist_table)

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

    # Initialize results tracking
    model_results = {name: None for _, name, _ in models_to_build}
    all_predictions = []
    model_metrics = {}

    console.print("\n")
    console.rule(f"[bold cyan]Building {len(models_to_build)} Models[/]")
    console.print("\n")

    # Build models with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:

        task = progress.add_task("[cyan]Training models...", total=len(models_to_build))

        for model_idx, (model_type, name, config) in enumerate(models_to_build):
            model_start = time.time()

            progress.update(task, description=f"[cyan]Training {name}...")

            try:
                if model_type == "lgb":
                    val_pred, test_pred = build_lgb_model(X_train, y_train, species_train, X_val, X_test, config, name)
                elif model_type == "xgb":
                    val_pred, test_pred = build_xgb_model(X_train, y_train, species_train, X_val, X_test, config, name)
                elif model_type == "catboost":
                    val_pred, test_pred = build_catboost_model(X_train, y_train, species_train, X_val, X_test, config, name)
                elif model_type == "mlp":
                    val_pred, test_pred = build_mlp_model(X_train, y_train, species_train, X_val, X_test, config, name)
                elif model_type == "pls_lgb":
                    val_pred, test_pred = build_pls_lgb_model(X_train, y_train, species_train, X_val, X_test, config, name)

                model_time = time.time() - model_start

                if val_pred is not None and test_pred is not None:
                    val_pred = apply_intrinsic_rules(val_pred, species_val)
                    test_pred = apply_intrinsic_rules(test_pred, species_test)
                    all_predictions.append((val_pred, test_pred))

                    val_mean_auc, per_ab = compute_val_mean_auc(y_val, val_pred)
                    model_metrics[name] = {
                        'val_mean_auc': val_mean_auc,
                        'per_antibiotic': per_ab,
                        'time': model_time
                    }
                    model_results[name] = {'val_mean_auc': val_mean_auc, 'time': model_time}

                    progress.update(task, description=f"[green]✓ {name}: AUC={val_mean_auc:.4f}")
                else:
                    model_results[name] = {'failed': True, 'time': model_time}

            except Exception as e:
                model_time = time.time() - model_start
                model_results[name] = {'failed': True, 'time': model_time, 'error': str(e)}
                console.print(f"  [red]✗ {name} failed: {e}[/]")

            progress.update(task, advance=1)

    console.print(f"\n[bold green]✓ Successfully built {len(all_predictions)} models[/]")

    # Show top models
    console.print("\n[bold]Top 5 Individual Models:[/]")
    top_models_table = Table(box=box.SIMPLE)
    top_models_table.add_column("Rank", style="bold", width=4)
    top_models_table.add_column("Model", style="cyan", width=15)
    top_models_table.add_column("Val Mean AUC", justify="right", width=12)

    sorted_models = sorted(model_metrics.items(), key=lambda x: x[1]['val_mean_auc'], reverse=True)
    for i, (name, m) in enumerate(sorted_models[:5]):
        top_models_table.add_row(str(i+1), name, f"{m['val_mean_auc']:.4f}")

    console.print(top_models_table)

    # Create ensembles
    console.print("\n")
    console.rule("[bold cyan]Creating Ensembles[/]")
    console.print("\n")

    with console.status("[bold green]Computing ensemble predictions...") as status:
        # 1. Rank average of all
        rank_val, rank_test = rank_average(all_predictions, species_val, species_test)
        rank_mean_auc, _ = compute_val_mean_auc(y_val, rank_val)

        # 2. Weighted average by val mean AUC
        weights = [model_metrics[name]['val_mean_auc'] for name in model_metrics]
        weighted_val, weighted_test = weighted_average(all_predictions, weights, species_val, species_test)
        weighted_mean_auc, _ = compute_val_mean_auc(y_val, weighted_val)

        # 3. Top-N rank average
        top_n = 5
        top_names = [name for name, _ in sorted_models[:top_n]]
        model_names_list = [name for _, name, _ in models_to_build]
        top_indices = []
        for i, (model_type, name, cfg) in enumerate(models_to_build):
            if name in top_names and i < len(all_predictions):
                top_indices.append(i)

        if len(top_indices) >= 3:
            top_predictions = [all_predictions[i] for i in top_indices[:min(top_n, len(top_indices))]]
            top_rank_val, top_rank_test = rank_average(top_predictions, species_val, species_test)
            top_rank_mean_auc, _ = compute_val_mean_auc(y_val, top_rank_val)
        else:
            top_rank_val, top_rank_test = rank_val, rank_test
            top_rank_mean_auc = rank_mean_auc

        # 4. Meta-blend
        meta_val = 0.5 * rank_val + 0.5 * weighted_val
        meta_test = 0.5 * rank_test + 0.5 * weighted_test
        meta_val = apply_intrinsic_rules(meta_val, species_val)
        meta_test = apply_intrinsic_rules(meta_test, species_test)
        meta_mean_auc, _ = compute_val_mean_auc(y_val, meta_val)

    all_blends = {
        'Rank-All': (rank_val, rank_test, rank_mean_auc),
        'Weighted-All': (weighted_val, weighted_test, weighted_mean_auc),
        'Top-N-Rank': (top_rank_val, top_rank_test, top_rank_mean_auc),
        'Meta-Blend': (meta_val, meta_test, meta_mean_auc),
    }

    # Find best
    best_name = max(all_blends, key=lambda x: all_blends[x][2])
    best_val, best_test, best_mean_auc = all_blends[best_name]

    # Show ensemble results
    ensemble_table = create_ensemble_table({name: auc for name, (_, _, auc) in all_blends.items()})
    console.print(ensemble_table)

    # Final summary
    console.print("\n")
    console.rule("[bold green]Final Results[/]")
    console.print("\n")

    summary_panel = Panel(
        f"""[bold cyan]BEST ENSEMBLE:[/] {best_name}
[bold cyan]Val Mean AUC:[/] {best_mean_auc:.4f}
[bold cyan]Target LB:[/] 0.83862

[bold cyan]Comparison:[/]
  {'✅ Above target!' if best_mean_auc > 0.83862 else '📉 Below target'}
  Difference: {best_mean_auc - 0.83862:+.4f}

[bold cyan]Total Time:[/] {str(datetime.now() - start_time).split('.')[0]}
[bold cyan]Models Built:[/] {len(all_predictions)}""",
        title="[bold white]Summary[/]",
        border_style="green"
    )
    console.print(summary_panel)

    # Save outputs
    console.print("\n[bold cyan]💾 Saving outputs...[/]")

    submission_dir = run_dir / "submissions"
    submission_dir.mkdir(exist_ok=True)

    # Save best ensemble
    best_sub = pd.DataFrame({
        "sample_id": sample_ids,
        **{ab: best_test[:, idx] for idx, ab in enumerate(ANTIBIOTICS)}
    })
    best_path = submission_dir / f"best_{best_name.lower().replace('-', '_')}.csv"
    best_sub.to_csv(best_path, index=False)
    console.print(f"  ✓ Best submission: {best_path}")

    # Save all ensemble variants
    for name, (_, test, _) in all_blends.items():
        sub = pd.DataFrame({
            "sample_id": sample_ids,
            **{ab: test[:, idx] for idx, ab in enumerate(ANTIBIOTICS)}
        })
        path = submission_dir / f"{name.lower().replace('-', '_')}.csv"
        sub.to_csv(path, index=False)

    # Save all model predictions as numpy arrays and pickle for later analysis
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)

    # Save individual model predictions
    for i, ((val_pred, test_pred), (model_type, name, config)) in enumerate(zip(all_predictions, models_to_build)):
        if name in model_metrics:
            pred_data = {
                'model_name': name,
                'model_type': model_type,
                'config': config,
                'val_preds': val_pred,
                'test_preds': test_pred,
                'val_mean_auc': model_metrics[name]['val_mean_auc'],
            }
            with open(predictions_dir / f"{name}_predictions.pkl", 'wb') as f:
                pickle.dump(pred_data, f)

    console.print(f"  ✓ Model predictions: {predictions_dir}/ ({len(all_predictions)} models)")

    # Save ensemble predictions as numpy arrays
    for name, (val_pred, test_pred, mean_auc) in all_blends.items():
        np.save(predictions_dir / f"{name.lower().replace('-', '_')}_val.npy", val_pred)
        np.save(predictions_dir / f"{name.lower().replace('-', '_')}_test.npy", test_pred)

    console.print(f"  ✓ Ensemble predictions: {predictions_dir}/ ({len(all_blends)} ensembles)")

    # Save comprehensive artifact for later analysis
    artifacts = {
        'all_predictions': all_predictions,
        'model_metrics': model_metrics,
        'all_blends': {name: {'val': val, 'test': test, 'mean_auc': mean_auc} for name, (val, test, mean_auc) in all_blends.items()},
        'best_ensemble': best_name,
        'best_val_mean_auc': float(best_mean_auc),
        'models_config': models_to_build,
        'timestamp': datetime.now().isoformat(),
    }
    with open(run_dir / "artifacts.pkl", 'wb') as f:
        pickle.dump(artifacts, f)
    console.print(f"  ✓ Artifacts: {run_dir}/artifacts.pkl")

    # Save results JSON
    results = {
        'run_dir': str(run_dir),
        'timestamp': datetime.now().isoformat(),
        'elapsed_time': str(datetime.now() - start_time).split('.')[0],
        'n_models': len(all_predictions),
        'hardware': {
            'n_gpus': N_GPUS,
            'n_cpu': N_CPU,
            'cuda_available': HAS_CUDA
        },
        'model_metrics': {name: {'val_mean_auc': float(m['val_mean_auc'])} for name, m in model_metrics.items()},
        'ensemble_metrics': {name: float(mean_auc) for name, (_, _, mean_auc) in all_blends.items()},
        'best_ensemble': best_name,
        'best_val_mean_auc': float(best_mean_auc),
    }

    results_path = run_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    console.print(f"  ✓ Results: {results_path}")

    # Final banner
    console.print(f"\n[bold green]{'═' * 60}[/]")
    console.print(f"[bold green]  ✅ COMPLETE! Elapsed: {str(datetime.now() - start_time).split('.')[0]}[/]")
    console.print(f"[bold green]  📁 All outputs in: {run_dir}[/]")
    console.print(f"[bold green]{'═' * 60}[/]")

    console.print(f"\n[bold]Kaggle submission command:[/]")
    console.print(f"[dim]kaggle competitions submit -c antimicrobial-resistance-prediction-from-maldi-tof -f {best_path} -m 'Miracle v2: {best_name}'[/]")

    return all_blends, model_metrics


if __name__ == "__main__":
    try:
        miracle_blend_v2()
    except Exception as e:
        console.print(f"[bold red]FATAL ERROR: {e}[/]")
        console.print(traceback.format_exc())
        sys.exit(1)
