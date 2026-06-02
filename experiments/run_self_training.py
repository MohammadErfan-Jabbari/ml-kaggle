#!/usr/bin/env python3
"""
SELF-TRAINING PIPELINE: Semi-supervised learning for AMR prediction.

Key Innovation:
- Uses partially-labeled training samples as unlabeled data for missing antibiotics
- Uses test samples as unlabeled data for ALL antibiotics
- Iteratively pseudo-labels high-confidence predictions
- Validation holdout from FULLY-LABELED samples only (clean eval)

This addresses the species distribution shift by learning from test distribution.

Target: Beat LB 0.83862
"""

import sys
import os
import warnings
import logging
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
import multiprocessing
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Rich for beautiful terminal output
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box

warnings.filterwarnings("ignore")

# Initialize Rich console
console = Console()

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SelfTrainingConfig:
    """Configuration for self-training pipeline."""
    # Data paths
    data_dir: Path = Path("/home/centcom/data/ml_kaggle/raw")
    output_base: Path = Path("/home/centcom/data/ml_kaggle/outputs")

    # Validation split
    val_fraction: float = 0.2  # 20% of fully-labeled for validation

    # Self-training parameters
    max_iterations: int = 5
    confidence_threshold_high: float = 0.9  # prob > this → label=1
    confidence_threshold_low: float = 0.1   # prob < this → label=0
    pseudo_label_weight: float = 0.5        # weight for pseudo-labeled samples
    min_new_labels_per_iter: int = 10       # stop if fewer new labels

    # LightGBM parameters
    lgb_params: dict = None
    n_folds: int = 5

    # Runtime
    n_jobs: int = None
    random_state: int = 42

    # Test mode
    smoke_test: bool = False  # If True, use subset of data

    def __post_init__(self):
        if self.lgb_params is None:
            self.lgb_params = {
                "n_estimators": 300,
                "learning_rate": 0.03,
                "num_leaves": 31,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
            }
        if self.n_jobs is None:
            self.n_jobs = min(multiprocessing.cpu_count(), 32)


# =============================================================================
# CONSTANTS
# =============================================================================

ANTIBIOTICS = [
    "Ampicillin", "Amoxicillin_Clavulanic_acid", "Cefotaxime", "Cefuroxime",
    "Ciprofloxacin", "Ertapenem", "Imipenem", "Levofloxacin"
]

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

SPECIES_NAMES = {0: "E.coli", 1: "K.pneumoniae", 2: "P.mirabilis", 3: "P.aeruginosa"}

# Test species distribution (target for validation split)
TEST_SPECIES_DISTRIBUTION = {
    0: 0.269,  # E.coli
    1: 0.508,  # K.pneumoniae (MAJORITY)
    2: 0.193,  # P.mirabilis
    3: 0.030   # P.aeruginosa
}

# Species weights for training (counter distribution shift)
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
# TRAINING STATE TRACKER
# =============================================================================

@dataclass
class TrainingState:
    """Track training state for display."""
    run_dir: Path = None
    start_time: datetime = None

    # Current progress
    current_antibiotic: str = ""
    current_antibiotic_idx: int = 0
    current_iteration: int = 0
    max_iterations: int = 5

    # Timing
    antibiotic_times: Dict[str, float] = field(default_factory=dict)
    iteration_times: List[float] = field(default_factory=list)

    # Results
    results: Dict[str, Dict] = field(default_factory=dict)

    # Counts
    n_labeled: int = 0
    n_pseudo: int = 0
    n_unlabeled: int = 0
    n_new_pseudo: int = 0

    def elapsed_time(self) -> str:
        if self.start_time is None:
            return "00:00:00"
        delta = datetime.now() - self.start_time
        return str(delta).split('.')[0]

    def eta(self) -> str:
        if not self.antibiotic_times or self.current_antibiotic_idx == 0:
            return "calculating..."

        avg_time = np.mean(list(self.antibiotic_times.values()))
        remaining = len(ANTIBIOTICS) - self.current_antibiotic_idx
        eta_seconds = avg_time * remaining
        return str(timedelta(seconds=int(eta_seconds)))

    def mean_auc(self) -> float:
        if not self.results:
            return 0.0
        aucs = [r.get('best_auc', 0) for r in self.results.values()]
        return np.mean(aucs) if aucs else 0.0


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def create_header_panel(state: TrainingState, config: SelfTrainingConfig) -> Panel:
    """Create the header panel with run info."""
    header_text = Text()
    header_text.append("🧬 SELF-TRAINING PIPELINE ", style="bold cyan")
    header_text.append("for AMR Prediction\n", style="cyan")
    header_text.append(f"📁 Run: {state.run_dir.name if state.run_dir else 'N/A'}\n", style="dim")
    header_text.append(f"⏱️  Elapsed: {state.elapsed_time()} | ", style="yellow")
    header_text.append(f"ETA: {state.eta()}\n", style="yellow")
    header_text.append(f"🎯 Target LB: 0.83862 | ", style="green")
    header_text.append(f"Current Mean AUC: {state.mean_auc():.4f}", style="bold green")

    return Panel(header_text, title="[bold white]Training Status[/]", border_style="blue")


def create_progress_table(state: TrainingState) -> Table:
    """Create table showing progress for all antibiotics."""
    table = Table(title="Antibiotic Progress", box=box.ROUNDED, show_header=True, header_style="bold magenta")

    table.add_column("Antibiotic", style="cyan", width=12)
    table.add_column("Status", justify="center", width=12)
    table.add_column("Iter", justify="center", width=6)
    table.add_column("Best AUC", justify="right", width=10)
    table.add_column("Δ AUC", justify="right", width=8)
    table.add_column("Pseudo", justify="right", width=8)
    table.add_column("Time", justify="right", width=8)

    for ab in ANTIBIOTICS:
        short = ANTIBIOTIC_SHORT[ab]

        if ab in state.results:
            result = state.results[ab]
            status = "✅ Done"
            status_style = "green"
            best_auc = f"{result.get('best_auc', 0):.4f}"
            delta = result.get('auc_improvement', 0)
            delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
            delta_style = "green" if delta > 0 else "red"
            pseudo = str(result.get('total_pseudo', 0))
            time_str = f"{result.get('time', 0):.1f}s"
        elif ab == state.current_antibiotic:
            status = "🔄 Training"
            status_style = "yellow"
            best_auc = "-"
            delta_str = "-"
            delta_style = "dim"
            pseudo = str(state.n_pseudo)
            time_str = "-"
        else:
            status = "⏳ Pending"
            status_style = "dim"
            best_auc = "-"
            delta_str = "-"
            delta_style = "dim"
            pseudo = "-"
            time_str = "-"

        iter_str = f"{state.current_iteration}/{state.max_iterations}" if ab == state.current_antibiotic else "-"

        table.add_row(
            short,
            Text(status, style=status_style),
            iter_str,
            best_auc,
            Text(delta_str, style=delta_style),
            pseudo,
            time_str
        )

    return table


def create_current_status_panel(state: TrainingState) -> Panel:
    """Create panel showing current training status."""
    if not state.current_antibiotic:
        return Panel("Initializing...", title="Current Status")

    status_text = Text()
    status_text.append(f"📊 Antibiotic: ", style="bold")
    status_text.append(f"{state.current_antibiotic}\n", style="cyan bold")
    status_text.append(f"🔄 Iteration: {state.current_iteration}/{state.max_iterations}\n", style="yellow")
    status_text.append(f"\n")
    status_text.append(f"📈 Labeled samples: {state.n_labeled:,}\n", style="green")
    status_text.append(f"🏷️  Pseudo-labeled: {state.n_pseudo:,}\n", style="blue")
    status_text.append(f"❓ Unlabeled pool: {state.n_unlabeled:,}\n", style="dim")
    status_text.append(f"✨ New this iter: {state.n_new_pseudo:,}\n", style="magenta")

    return Panel(status_text, title=f"[bold]Training {ANTIBIOTIC_SHORT.get(state.current_antibiotic, '?')}[/]", border_style="yellow")


def create_summary_panel(state: TrainingState) -> Panel:
    """Create final summary panel."""
    summary = Text()
    summary.append("🎯 FINAL RESULTS\n\n", style="bold green")

    if state.results:
        for ab in ANTIBIOTICS:
            if ab in state.results:
                auc = state.results[ab].get('best_auc', 0)
                short = ANTIBIOTIC_SHORT[ab]
                bar_len = int(auc * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                summary.append(f"{short}: [{bar}] {auc:.4f}\n")

        summary.append(f"\n")
        summary.append(f"📊 Mean AUC: {state.mean_auc():.4f}\n", style="bold cyan")
        summary.append(f"🎯 Target: 0.83862\n", style="yellow")

        diff = state.mean_auc() - 0.83862
        if diff > 0:
            summary.append(f"✅ Above target by {diff:.4f}!", style="bold green")
        else:
            summary.append(f"📉 Below target by {abs(diff):.4f}", style="bold red")

    return Panel(summary, title="[bold]Summary[/]", border_style="green")


def print_banner():
    """Print startup banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ███████╗███████╗██╗     ███████╗    ████████╗██████╗  █████╗ ██╗███╗   ██╗  ║
║   ██╔════╝██╔════╝██║     ██╔════╝    ╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║  ║
║   ███████╗█████╗  ██║     █████╗         ██║   ██████╔╝███████║██║██╔██╗ ██║  ║
║   ╚════██║██╔══╝  ██║     ██╔══╝         ██║   ██╔══██╗██╔══██║██║██║╚██╗██║  ║
║   ███████║███████╗███████╗██║            ██║   ██║  ██║██║  ██║██║██║ ╚████║  ║
║   ╚══════╝╚══════╝╚══════╝╚═╝            ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝  ║
║                                                                               ║
║            Semi-Supervised Learning for AMR Prediction                        ║
║            Target: Beat LB 0.83862 using unlabeled test data                  ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
    console.print(banner, style="cyan")


# =============================================================================
# DATA LOADING AND PARTITIONING
# =============================================================================

@dataclass
class DataPartition:
    """Container for data partitions."""
    # Holdout validation (fully labeled, matches test distribution)
    X_val: np.ndarray
    y_val: np.ndarray
    species_val: np.ndarray
    val_indices: np.ndarray

    # Initial training (fully labeled, remaining after val split)
    X_train_labeled: np.ndarray
    y_train_labeled: np.ndarray
    species_train_labeled: np.ndarray
    train_labeled_indices: np.ndarray

    # Partially labeled (per antibiotic, some have labels, some don't)
    X_partial: np.ndarray
    y_partial: np.ndarray  # Contains NaN for missing labels
    species_partial: np.ndarray
    partial_indices: np.ndarray

    # Test data (unlabeled for all antibiotics)
    X_test: np.ndarray
    species_test: np.ndarray
    sample_ids_test: np.ndarray

    # Feature mask (for removing constant features)
    feature_mask: np.ndarray


def load_and_partition_data(config: SelfTrainingConfig) -> DataPartition:
    """Load data and create partitions for self-training."""
    console.print("\n[bold cyan]📂 Loading Data...[/]")

    with console.status("[bold green]Reading CSV files...") as status:
        train_df = pd.read_csv(config.data_dir / "train.csv")
        test_df = pd.read_csv(config.data_dir / "test.csv")

        # Extract features
        feature_cols = [f"maldi_feature_{i}" for i in range(6000)]

        X_train_full = train_df[feature_cols].values.astype(np.float32)
        y_train_full = train_df[ANTIBIOTICS].values.astype(np.float32)
        species_train_full = train_df["species_id"].values.astype(np.int32)

        X_test = test_df[feature_cols].values.astype(np.float32)
        species_test = test_df["species_id"].values.astype(np.int32)
        sample_ids_test = test_df["sample_id"].values

    console.print(f"  ✓ Train: {X_train_full.shape[0]:,} samples, {X_train_full.shape[1]:,} features")
    console.print(f"  ✓ Test: {X_test.shape[0]:,} samples")

    # Identify fully-labeled vs partially-labeled samples
    n_labels_per_sample = np.sum(~np.isnan(y_train_full), axis=1)
    fully_labeled_mask = (n_labels_per_sample == len(ANTIBIOTICS))

    n_fully_labeled = fully_labeled_mask.sum()
    n_partially_labeled = (~fully_labeled_mask).sum()
    console.print(f"  ✓ Fully labeled: {n_fully_labeled:,} | Partially labeled: {n_partially_labeled:,}")

    # Split fully-labeled into validation + training
    fully_labeled_indices = np.where(fully_labeled_mask)[0]

    X_fully = X_train_full[fully_labeled_mask]
    y_fully = y_train_full[fully_labeled_mask]
    species_fully = species_train_full[fully_labeled_mask]

    val_indices_local, train_indices_local = create_test_distribution_split(
        species_fully,
        val_fraction=config.val_fraction,
        random_state=config.random_state
    )

    # Map back to original indices
    val_indices = fully_labeled_indices[val_indices_local]
    train_labeled_indices = fully_labeled_indices[train_indices_local]
    partial_indices = np.where(~fully_labeled_mask)[0]

    console.print(f"  ✓ Validation: {len(val_indices):,} | Training: {len(train_labeled_indices):,}")

    # Remove constant features
    feature_variances = X_train_full.var(axis=0)
    feature_mask = feature_variances > 1e-5
    n_removed = (~feature_mask).sum()
    console.print(f"  ✓ Removed {n_removed:,} constant features, keeping {feature_mask.sum():,}")

    # Apply feature mask
    X_train_full = X_train_full[:, feature_mask]
    X_test = X_test[:, feature_mask]

    # Verify validation species distribution
    val_species = species_train_full[val_indices]
    console.print("\n[bold]Validation Species Distribution:[/]")

    dist_table = Table(box=box.SIMPLE)
    dist_table.add_column("Species", style="cyan")
    dist_table.add_column("Actual", justify="right")
    dist_table.add_column("Target", justify="right")
    dist_table.add_column("Match", justify="center")

    for s_id, s_name in SPECIES_NAMES.items():
        actual = (val_species == s_id).sum() / len(val_species)
        target = TEST_SPECIES_DISTRIBUTION[s_id]
        match = "✅" if abs(actual - target) < 0.02 else "⚠️"
        dist_table.add_row(s_name, f"{actual*100:.1f}%", f"{target*100:.1f}%", match)

    console.print(dist_table)

    return DataPartition(
        X_val=X_train_full[val_indices],
        y_val=y_train_full[val_indices],
        species_val=species_train_full[val_indices],
        val_indices=val_indices,

        X_train_labeled=X_train_full[train_labeled_indices],
        y_train_labeled=y_train_full[train_labeled_indices],
        species_train_labeled=species_train_full[train_labeled_indices],
        train_labeled_indices=train_labeled_indices,

        X_partial=X_train_full[partial_indices],
        y_partial=y_train_full[partial_indices],
        species_partial=species_train_full[partial_indices],
        partial_indices=partial_indices,

        X_test=X_test,
        species_test=species_test,
        sample_ids_test=sample_ids_test,

        feature_mask=feature_mask
    )


def create_test_distribution_split(
    species: np.ndarray,
    val_fraction: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Split indices to create validation set matching test species distribution."""
    rng = np.random.default_rng(random_state)
    n_total = len(species)
    n_val = int(n_total * val_fraction)

    val_indices = []
    train_indices = []

    for species_id in range(4):
        species_mask = (species == species_id)
        species_indices = np.where(species_mask)[0]

        target_count = int(n_val * TEST_SPECIES_DISTRIBUTION[species_id])
        target_count = min(target_count, len(species_indices))

        rng.shuffle(species_indices)
        val_indices.extend(species_indices[:target_count])
        train_indices.extend(species_indices[target_count:])

    return np.array(val_indices), np.array(train_indices)


# =============================================================================
# SELF-TRAINING CORE
# =============================================================================

def get_sample_weights(species: np.ndarray, is_pseudo: np.ndarray = None, pseudo_weight: float = 0.5) -> np.ndarray:
    """Get sample weights based on species and pseudo-label status."""
    weights = np.array([SPECIES_WEIGHTS.get(s, 1.0) for s in species])
    if is_pseudo is not None:
        weights[is_pseudo] *= pseudo_weight
    return weights


def train_single_antibiotic_iteration(
    antibiotic_idx: int,
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    species_labeled: np.ndarray,
    is_pseudo_labeled: np.ndarray,
    X_unlabeled: np.ndarray,
    species_unlabeled: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    config: SelfTrainingConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Train model for single antibiotic and predict on unlabeled + val + test."""

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
        return (
            np.full(len(X_unlabeled), y_train[0]),
            np.full(len(X_val), y_train[0]),
            np.full(len(X_test), y_train[0]),
            0.5
        )

    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_state)

    unlabeled_preds_folds = []
    val_preds_folds = []
    test_preds_folds = []

    for fold_idx, (train_idx, _) in enumerate(skf.split(X_train, species_train)):
        weights = get_sample_weights(
            species_train[train_idx],
            is_pseudo_train[train_idx],
            config.pseudo_label_weight
        )

        model = lgb.LGBMClassifier(
            random_state=config.random_state,
            verbose=-1,
            n_jobs=config.n_jobs,
            device='cpu',
            **config.lgb_params
        )

        model.fit(X_train[train_idx], y_train[train_idx], sample_weight=weights)

        unlabeled_preds_folds.append(model.predict_proba(X_unlabeled)[:, 1])
        val_preds_folds.append(model.predict_proba(X_val)[:, 1])
        test_preds_folds.append(model.predict_proba(X_test)[:, 1])

    unlabeled_preds = np.mean(unlabeled_preds_folds, axis=0)
    val_preds = np.mean(val_preds_folds, axis=0)
    test_preds = np.mean(test_preds_folds, axis=0)

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
    config: SelfTrainingConfig
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


def apply_intrinsic_rules(predictions: np.ndarray, species: np.ndarray, antibiotic: str) -> np.ndarray:
    """Apply biological intrinsic resistance rules."""
    predictions = predictions.copy()
    for species_id, resistant_antibiotics in INTRINSIC_RESISTANCE.items():
        if antibiotic in resistant_antibiotics:
            species_mask = (species == species_id)
            predictions[species_mask] = 1.0
    return predictions


# =============================================================================
# MAIN SELF-TRAINING LOOP
# =============================================================================

def run_self_training(config: SelfTrainingConfig) -> Dict:
    """Run the full self-training pipeline with live display."""

    # Print banner
    print_banner()

    # Setup output directory
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = config.output_base / "self_training_runs" / f"run_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Initialize state
    state = TrainingState(
        run_dir=run_dir,
        start_time=datetime.now(),
        max_iterations=config.max_iterations
    )

    console.print(f"\n[bold green]📁 Run Directory:[/] {run_dir}")
    console.print(f"[bold green]⚙️  Config:[/] max_iter={config.max_iterations}, conf=[{config.confidence_threshold_low}, {config.confidence_threshold_high}], pseudo_weight={config.pseudo_label_weight}")

    # Load data
    data = load_and_partition_data(config)

    # Storage for results
    final_val_preds = np.zeros((len(data.X_val), len(ANTIBIOTICS)))
    final_test_preds = np.zeros((len(data.X_test), len(ANTIBIOTICS)))

    console.print("\n")
    console.rule("[bold cyan]Starting Self-Training Loop[/]")
    console.print("\n")

    # Process each antibiotic with progress display
    for ab_idx, antibiotic in enumerate(ANTIBIOTICS):
        ab_start_time = time.time()

        state.current_antibiotic = antibiotic
        state.current_antibiotic_idx = ab_idx
        state.current_iteration = 0

        short = ANTIBIOTIC_SHORT[antibiotic]
        console.print(f"\n[bold cyan]{'═' * 60}[/]")
        console.print(f"[bold cyan]  {short}: {antibiotic} ({ab_idx + 1}/{len(ANTIBIOTICS)})[/]")
        console.print(f"[bold cyan]{'═' * 60}[/]")

        # Initialize labeled pool
        X_labeled = data.X_train_labeled.copy()
        y_labeled = data.y_train_labeled[:, ab_idx].copy()
        species_labeled = data.species_train_labeled.copy()
        is_pseudo_labeled = np.zeros(len(X_labeled), dtype=bool)

        # Add partially-labeled samples that have labels for this antibiotic
        partial_has_label = ~np.isnan(data.y_partial[:, ab_idx])
        if partial_has_label.sum() > 0:
            X_labeled = np.vstack([X_labeled, data.X_partial[partial_has_label]])
            y_labeled = np.concatenate([y_labeled, data.y_partial[partial_has_label, ab_idx]])
            species_labeled = np.concatenate([species_labeled, data.species_partial[partial_has_label]])
            is_pseudo_labeled = np.concatenate([is_pseudo_labeled, np.zeros(partial_has_label.sum(), dtype=bool)])

        # Initialize unlabeled pool
        partial_no_label = np.isnan(data.y_partial[:, ab_idx])
        X_unlabeled_partial = data.X_partial[partial_no_label]
        species_unlabeled_partial = data.species_partial[partial_no_label]

        X_unlabeled = np.vstack([X_unlabeled_partial, data.X_test])
        species_unlabeled = np.concatenate([species_unlabeled_partial, data.species_test])

        unlabeled_pseudo_mask = np.zeros(len(X_unlabeled), dtype=bool)
        unlabeled_pseudo_labels = np.full(len(X_unlabeled), np.nan)

        state.n_labeled = len(X_labeled)
        state.n_unlabeled = len(X_unlabeled)
        state.n_pseudo = 0

        console.print(f"  📊 Initial: {len(X_labeled):,} labeled, {len(X_unlabeled):,} unlabeled")

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
                state.current_iteration = iteration + 1
                iter_start = time.time()

                # Train and predict
                unlabeled_preds, val_preds, test_preds, val_auc = train_single_antibiotic_iteration(
                    antibiotic_idx=ab_idx,
                    X_labeled=X_labeled,
                    y_labeled=y_labeled,
                    species_labeled=species_labeled,
                    is_pseudo_labeled=is_pseudo_labeled,
                    X_unlabeled=X_unlabeled,
                    species_unlabeled=species_unlabeled,
                    X_val=data.X_val,
                    y_val=data.y_val,
                    X_test=data.X_test,
                    config=config,
                )

                # Apply intrinsic rules
                val_preds = apply_intrinsic_rules(val_preds, data.species_val, antibiotic)
                test_preds = apply_intrinsic_rules(test_preds, data.species_test, antibiotic)

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

                state.n_new_pseudo = n_new
                state.n_pseudo = unlabeled_pseudo_mask.sum()

                iter_time = time.time() - iter_start

                # Update progress
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
                        state.n_labeled = len(X_labeled)

        # Store results
        ab_time = time.time() - ab_start_time
        auc_improvement = best_val_auc - initial_val_auc

        state.results[antibiotic] = {
            'best_auc': best_val_auc,
            'initial_auc': initial_val_auc,
            'auc_improvement': auc_improvement,
            'total_pseudo': int(unlabeled_pseudo_mask.sum()),
            'time': ab_time
        }
        state.antibiotic_times[antibiotic] = ab_time

        final_val_preds[:, ab_idx] = best_val_preds if best_val_preds is not None else val_preds
        final_test_preds[:, ab_idx] = best_test_preds if best_test_preds is not None else test_preds

        # Save predictions for this antibiotic (incremental saving)
        predictions_dir = run_dir / "predictions"
        predictions_dir.mkdir(exist_ok=True)

        ab_pred_data = {
            'antibiotic': antibiotic,
            'val_preds': best_val_preds if best_val_preds is not None else val_preds,
            'test_preds': best_test_preds if best_test_preds is not None else test_preds,
            'best_auc': best_val_auc,
            'initial_auc': initial_val_auc,
            'n_pseudo_labels': int(unlabeled_pseudo_mask.sum()),
        }
        with open(predictions_dir / f"{ANTIBIOTIC_SHORT[antibiotic]}_predictions.pkl", 'wb') as f:
            pickle.dump(ab_pred_data, f)

        # Print antibiotic summary
        improvement_style = "green" if auc_improvement > 0 else "red"
        console.print(f"  ✅ Best AUC: [bold]{best_val_auc:.4f}[/] (Δ [{improvement_style}]{auc_improvement:+.4f}[/]) • {unlabeled_pseudo_mask.sum():,} pseudo-labels • {ab_time:.1f}s")

    # Final results
    console.print("\n")
    console.rule("[bold green]Training Complete[/]")
    console.print("\n")

    # Create final summary table
    summary_table = Table(title="[bold]Final Results by Antibiotic[/]", box=box.DOUBLE_EDGE)
    summary_table.add_column("Antibiotic", style="cyan")
    summary_table.add_column("Initial AUC", justify="right")
    summary_table.add_column("Best AUC", justify="right", style="green")
    summary_table.add_column("Improvement", justify="right")
    summary_table.add_column("Pseudo Labels", justify="right")

    for ab in ANTIBIOTICS:
        result = state.results[ab]
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

    # Mean AUC
    mean_auc = state.mean_auc()
    console.print(f"\n[bold]📊 MEAN AUC: {mean_auc:.4f}[/]")
    console.print(f"[bold]🎯 TARGET LB: 0.83862[/]")

    diff = mean_auc - 0.83862
    if diff > 0:
        console.print(f"[bold green]✅ Above target by {diff:.4f}![/]")
    else:
        console.print(f"[bold yellow]📉 Below target by {abs(diff):.4f}[/]")

    # Save outputs
    console.print("\n[bold cyan]💾 Saving outputs...[/]")

    submissions_dir = run_dir / "submissions"
    submissions_dir.mkdir(exist_ok=True)

    submission = pd.DataFrame({
        "sample_id": data.sample_ids_test,
        **{ab: final_test_preds[:, idx] for idx, ab in enumerate(ANTIBIOTICS)}
    })
    submission_path = submissions_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    console.print(f"  ✓ Submission: {submission_path}")

    # Save all predictions as numpy arrays for further analysis
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)

    np.save(predictions_dir / "final_val_preds.npy", final_val_preds)
    np.save(predictions_dir / "final_test_preds.npy", final_test_preds)
    console.print(f"  ✓ Predictions: {predictions_dir}/final_*.npy")

    # Save comprehensive run artifact
    all_artifacts = {
        'config': asdict(config),
        'final_val_preds': final_val_preds,
        'final_test_preds': final_test_preds,
        'antibiotic_results': state.results,
        'mean_auc': mean_auc,
        'timestamp': datetime.now().isoformat(),
    }
    with open(run_dir / "artifacts.pkl", 'wb') as f:
        pickle.dump(all_artifacts, f)
    console.print(f"  ✓ Artifacts: {run_dir}/artifacts.pkl")

    results = {
        'run_dir': str(run_dir),
        'timestamp': datetime.now().isoformat(),
        'elapsed_time': state.elapsed_time(),
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        'mean_auc': mean_auc,
        'per_antibiotic': state.results
    }

    results_path = run_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    console.print(f"  ✓ Results: {results_path}")

    # Final banner
    console.print(f"\n[bold green]{'═' * 60}[/]")
    console.print(f"[bold green]  ✅ COMPLETE! Elapsed: {state.elapsed_time()}[/]")
    console.print(f"[bold green]  📁 All outputs in: {run_dir}[/]")
    console.print(f"[bold green]{'═' * 60}[/]")

    console.print(f"\n[bold]Kaggle submission command:[/]")
    console.print(f"[dim]kaggle competitions submit -c antimicrobial-resistance-prediction-from-maldi-tof -f {submission_path} -m 'Self-training'[/]")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Self-training for AMR prediction")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test")
    parser.add_argument("--max-iter", type=int, default=5, help="Max self-training iterations")
    parser.add_argument("--conf-high", type=float, default=0.9, help="High confidence threshold")
    parser.add_argument("--conf-low", type=float, default=0.1, help="Low confidence threshold")
    parser.add_argument("--pseudo-weight", type=float, default=0.5, help="Pseudo-label weight")

    args = parser.parse_args()

    config = SelfTrainingConfig(
        smoke_test=args.smoke_test,
        max_iterations=args.max_iter,
        confidence_threshold_high=args.conf_high,
        confidence_threshold_low=args.conf_low,
        pseudo_label_weight=args.pseudo_weight
    )

    results = run_self_training(config)

    return results


if __name__ == "__main__":
    main()
