#!/usr/bin/env python3
"""
SPECIES-SPECIFIC MODELS: 32 models (4 species × 8 antibiotics)

Key Innovation:
- Train separate model for each (species, antibiotic) pair
- Routes test samples to their species-specific model at inference
- Addresses species distribution shift directly

This should help because:
- K.pneumoniae (51% of test) gets its own dedicated models
- No interference from P.aeruginosa patterns (43% of train but 3% of test)

Target: Beat LB 0.83862
"""

import sys
import os
import warnings
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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.panel import Panel
from rich.text import Text
from rich import box

warnings.filterwarnings("ignore")

# Initialize Rich console
console = Console()

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SpeciesSpecificConfig:
    """Configuration for species-specific training."""
    # Data paths
    data_dir: Path = Path("/home/erfan/data/ml_kaggle/raw")
    output_base: Path = Path("/sata_disk/users/erfan/ml_kaggle/outputs")

    # Validation split
    val_fraction: float = 0.2  # 20% for validation

    # Minimum samples per species for dedicated model
    min_samples_per_species: int = 50

    # LightGBM parameters
    lgb_params: dict = None
    n_folds: int = 5

    # Fallback to global model
    use_global_fallback: bool = True

    # Runtime
    n_jobs: int = None
    random_state: int = 42

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
SPECIES_SHORT = {0: "EC", 1: "KP", 2: "PM", 3: "PA"}

# Test species distribution (for validation stratification)
TEST_SPECIES_DISTRIBUTION = {
    0: 0.269,  # E.coli
    1: 0.508,  # K.pneumoniae
    2: 0.193,  # P.mirabilis
    3: 0.030   # P.aeruginosa
}

# Intrinsic resistance rules
INTRINSIC_RESISTANCE = {
    3: ["Ampicillin", "Amoxicillin_Clavulanic_acid", "Ertapenem", "Cefotaxime", "Cefuroxime"],  # P. aeruginosa
    2: ["Imipenem"],  # P. mirabilis
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(config: SpeciesSpecificConfig):
    """Load and prepare data."""
    console.print("\n[bold cyan]📂 Loading Data...[/]")

    train_df = pd.read_csv(config.data_dir / "train.csv")
    test_df = pd.read_csv(config.data_dir / "test.csv")

    feature_cols = [f"maldi_feature_{i}" for i in range(6000)]

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[ANTIBIOTICS].values.astype(np.float32)
    species_train = train_df["species_id"].values.astype(np.int32)

    X_test = test_df[feature_cols].values.astype(np.float32)
    species_test = test_df["species_id"].values.astype(np.int32)
    sample_ids_test = test_df["sample_id"].values

    console.print(f"  ✓ Train: {X_train.shape[0]:,} samples, {X_train.shape[1]:,} features")
    console.print(f"  ✓ Test: {X_test.shape[0]:,} samples")

    # Remove constant features
    feature_variances = X_train.var(axis=0)
    feature_mask = feature_variances > 1e-5
    n_removed = (~feature_mask).sum()
    console.print(f"  ✓ Removed {n_removed:,} constant features, keeping {feature_mask.sum():,}")

    X_train = X_train[:, feature_mask]
    X_test = X_test[:, feature_mask]

    # Create validation split (stratified by species to match test distribution)
    val_indices, train_indices = create_stratified_split(
        species_train,
        val_fraction=config.val_fraction,
        target_distribution=TEST_SPECIES_DISTRIBUTION,
        random_state=config.random_state
    )

    console.print(f"  ✓ Train split: {len(train_indices):,} | Val split: {len(val_indices):,}")

    # Print species distribution comparison
    console.print("\n[bold]Species Distribution:[/]")
    dist_table = Table(box=box.SIMPLE)
    dist_table.add_column("Species", style="cyan")
    dist_table.add_column("Train", justify="right")
    dist_table.add_column("Val", justify="right")
    dist_table.add_column("Test", justify="right")

    for s_id, s_name in SPECIES_NAMES.items():
        train_pct = (species_train[train_indices] == s_id).sum() / len(train_indices) * 100
        val_pct = (species_train[val_indices] == s_id).sum() / len(val_indices) * 100
        test_pct = (species_test == s_id).sum() / len(species_test) * 100
        dist_table.add_row(s_name, f"{train_pct:.1f}%", f"{val_pct:.1f}%", f"{test_pct:.1f}%")

    console.print(dist_table)

    return {
        'X_train': X_train[train_indices],
        'y_train': y_train[train_indices],
        'species_train': species_train[train_indices],
        'X_val': X_train[val_indices],
        'y_val': y_train[val_indices],
        'species_val': species_train[val_indices],
        'X_test': X_test,
        'species_test': species_test,
        'sample_ids_test': sample_ids_test,
        'feature_mask': feature_mask
    }


def create_stratified_split(
    species: np.ndarray,
    val_fraction: float,
    target_distribution: Dict[int, float],
    random_state: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Create validation split matching target species distribution."""
    rng = np.random.default_rng(random_state)
    n_total = len(species)
    n_val = int(n_total * val_fraction)

    val_indices = []
    train_indices = []

    for species_id in range(4):
        species_mask = (species == species_id)
        species_indices = np.where(species_mask)[0]

        target_count = int(n_val * target_distribution.get(species_id, 0.25))
        target_count = min(target_count, len(species_indices))

        rng.shuffle(species_indices)
        val_indices.extend(species_indices[:target_count])
        train_indices.extend(species_indices[target_count:])

    return np.array(val_indices), np.array(train_indices)


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_species_antibiotic_model(
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: SpeciesSpecificConfig,
    species_name: str,
    antibiotic_name: str
) -> Tuple[Optional[object], float, np.ndarray]:
    """Train model for specific (species, antibiotic) pair."""

    # Check for valid labels
    valid_mask = ~np.isnan(y)
    if valid_mask.sum() < 30:
        return None, 0.5, np.full(len(X_val), 0.5)

    X_train = X[valid_mask]
    y_train = y[valid_mask]

    # Check for class balance
    if len(np.unique(y_train)) < 2:
        # Single class - return constant prediction
        pred_val = np.full(len(X_val), y_train[0])
        return None, 0.5, pred_val

    # 5-fold CV for robust predictions
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_state)

    val_preds_folds = []
    models = []

    for fold_idx, (train_idx, _) in enumerate(skf.split(X_train, y_train)):
        model = lgb.LGBMClassifier(
            random_state=config.random_state + fold_idx,
            verbose=-1,
            n_jobs=config.n_jobs,
            **config.lgb_params
        )

        model.fit(X_train[train_idx], y_train[train_idx])
        models.append(model)

        if len(X_val) > 0:
            val_preds_folds.append(model.predict_proba(X_val)[:, 1])

    # Average predictions across folds
    if val_preds_folds:
        val_preds = np.mean(val_preds_folds, axis=0)

        # Calculate AUC on valid validation samples
        val_valid = ~np.isnan(y_val)
        if val_valid.sum() > 10 and len(np.unique(y_val[val_valid])) > 1:
            val_auc = roc_auc_score(y_val[val_valid], val_preds[val_valid])
        else:
            val_auc = 0.5
    else:
        val_preds = np.full(len(X_val), 0.5)
        val_auc = 0.5

    return models, val_auc, val_preds


def train_global_model(
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: SpeciesSpecificConfig
) -> Tuple[List, float, np.ndarray]:
    """Train global model (fallback when species has too few samples)."""

    valid_mask = ~np.isnan(y)
    if valid_mask.sum() < 30:
        return [], 0.5, np.full(len(X_val), 0.5)

    X_train = X[valid_mask]
    y_train = y[valid_mask]

    if len(np.unique(y_train)) < 2:
        return [], 0.5, np.full(len(X_val), y_train[0])

    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_state)

    val_preds_folds = []
    models = []

    for fold_idx, (train_idx, _) in enumerate(skf.split(X_train, y_train)):
        model = lgb.LGBMClassifier(
            random_state=config.random_state + fold_idx,
            verbose=-1,
            n_jobs=config.n_jobs,
            **config.lgb_params
        )
        model.fit(X_train[train_idx], y_train[train_idx])
        models.append(model)

        if len(X_val) > 0:
            val_preds_folds.append(model.predict_proba(X_val)[:, 1])

    if val_preds_folds:
        val_preds = np.mean(val_preds_folds, axis=0)
        val_valid = ~np.isnan(y_val)
        if val_valid.sum() > 10 and len(np.unique(y_val[val_valid])) > 1:
            val_auc = roc_auc_score(y_val[val_valid], val_preds[val_valid])
        else:
            val_auc = 0.5
    else:
        val_preds = np.full(len(X_val), 0.5)
        val_auc = 0.5

    return models, val_auc, val_preds


def apply_intrinsic_rules(predictions: np.ndarray, species: np.ndarray, antibiotic: str) -> np.ndarray:
    """Apply biological intrinsic resistance rules."""
    predictions = predictions.copy()
    for species_id, resistant_antibiotics in INTRINSIC_RESISTANCE.items():
        if antibiotic in resistant_antibiotics:
            species_mask = (species == species_id)
            predictions[species_mask] = 1.0
    return predictions


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_species_specific_training(config: SpeciesSpecificConfig) -> Dict:
    """Run the species-specific training pipeline."""

    # Print banner
    banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ███████╗██████╗ ███████╗ ██████╗██╗███████╗███████╗                         ║
║   ██╔════╝██╔══██╗██╔════╝██╔════╝██║██╔════╝██╔════╝                         ║
║   ███████╗██████╔╝█████╗  ██║     ██║█████╗  ███████╗                         ║
║   ╚════██║██╔═══╝ ██╔══╝  ██║     ██║██╔══╝  ╚════██║                         ║
║   ███████║██║     ███████╗╚██████╗██║███████╗███████║                         ║
║   ╚══════╝╚═╝     ╚══════╝ ╚═════╝╚═╝╚══════╝╚══════╝                         ║
║                                                                               ║
║            Species-Specific Models (4 species × 8 antibiotics = 32 models)   ║
║            Target: Beat LB 0.83862 by addressing species distribution shift  ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
    console.print(banner, style="cyan")

    # Setup output directory
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = config.output_base / "species_specific_runs" / f"run_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold green]📁 Run Directory:[/] {run_dir}")

    start_time = time.time()

    # Load data
    data = load_data(config)

    # Storage for results
    all_models = {}  # (species, antibiotic) -> models
    all_val_preds = np.zeros((len(data['X_val']), len(ANTIBIOTICS)))
    all_test_preds = np.zeros((len(data['X_test']), len(ANTIBIOTICS)))
    results_per_model = {}

    console.print("\n")
    console.rule("[bold cyan]Training Species-Specific Models[/]")
    console.print("\n")

    # Calculate total models to train
    total_models = len(SPECIES_NAMES) * len(ANTIBIOTICS)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        main_task = progress.add_task("[cyan]Training models...", total=total_models)

        for ab_idx, antibiotic in enumerate(ANTIBIOTICS):
            ab_short = ANTIBIOTIC_SHORT[antibiotic]

            # First, train a global model for this antibiotic (fallback)
            global_models, global_val_auc, global_val_preds = train_global_model(
                data['X_train'],
                data['y_train'][:, ab_idx],
                data['X_val'],
                data['y_val'][:, ab_idx],
                config
            )

            # Initialize predictions with global model
            val_preds_combined = global_val_preds.copy()
            test_preds_combined = np.zeros(len(data['X_test']))

            # Predict on test with global model
            if global_models:
                test_preds_folds = []
                for model in global_models:
                    test_preds_folds.append(model.predict_proba(data['X_test'])[:, 1])
                test_preds_combined = np.mean(test_preds_folds, axis=0)
            else:
                test_preds_combined = np.full(len(data['X_test']), 0.5)

            results_per_model[f"GLOBAL_{ab_short}"] = {
                'val_auc': global_val_auc,
                'n_samples': (~np.isnan(data['y_train'][:, ab_idx])).sum()
            }

            # Now train species-specific models
            for species_id, species_name in SPECIES_NAMES.items():
                sp_short = SPECIES_SHORT[species_id]
                model_name = f"{sp_short}_{ab_short}"

                # Get species-specific data
                train_species_mask = (data['species_train'] == species_id)
                val_species_mask = (data['species_val'] == species_id)
                test_species_mask = (data['species_test'] == species_id)

                X_train_sp = data['X_train'][train_species_mask]
                y_train_sp = data['y_train'][train_species_mask, ab_idx]
                X_val_sp = data['X_val'][val_species_mask]
                y_val_sp = data['y_val'][val_species_mask, ab_idx]

                n_labeled = (~np.isnan(y_train_sp)).sum()

                # Check if we have enough samples
                if n_labeled >= config.min_samples_per_species:
                    # Train species-specific model
                    models, val_auc, val_preds = train_species_antibiotic_model(
                        X_train_sp, y_train_sp,
                        X_val_sp, y_val_sp,
                        config, species_name, antibiotic
                    )

                    if models:
                        all_models[(species_id, ab_idx)] = models

                        # Update validation predictions for this species
                        val_preds_combined[val_species_mask] = val_preds

                        # Update test predictions for this species
                        test_preds_sp = []
                        for model in models:
                            X_test_sp = data['X_test'][test_species_mask]
                            if len(X_test_sp) > 0:
                                test_preds_sp.append(model.predict_proba(X_test_sp)[:, 1])

                        if test_preds_sp:
                            test_preds_combined[test_species_mask] = np.mean(test_preds_sp, axis=0)

                        results_per_model[model_name] = {
                            'val_auc': val_auc,
                            'n_samples': n_labeled,
                            'used': 'species-specific'
                        }
                    else:
                        results_per_model[model_name] = {
                            'val_auc': 0.5,
                            'n_samples': n_labeled,
                            'used': 'global-fallback'
                        }
                else:
                    # Use global model (already set as default)
                    results_per_model[model_name] = {
                        'val_auc': global_val_auc,
                        'n_samples': n_labeled,
                        'used': 'global-fallback'
                    }

                progress.update(main_task, advance=1,
                               description=f"[cyan]{model_name}: {n_labeled} samples")

            # Apply intrinsic resistance rules
            val_preds_combined = apply_intrinsic_rules(val_preds_combined, data['species_val'], antibiotic)
            test_preds_combined = apply_intrinsic_rules(test_preds_combined, data['species_test'], antibiotic)

            all_val_preds[:, ab_idx] = val_preds_combined
            all_test_preds[:, ab_idx] = test_preds_combined

    # Calculate final metrics
    console.print("\n")
    console.rule("[bold green]Results[/]")
    console.print("\n")

    # Per-antibiotic AUC
    results_table = Table(title="[bold]Per-Antibiotic Validation AUC[/]", box=box.DOUBLE_EDGE)
    results_table.add_column("Antibiotic", style="cyan")
    results_table.add_column("Global AUC", justify="right")
    results_table.add_column("E.coli", justify="right")
    results_table.add_column("K.pn", justify="right")
    results_table.add_column("P.mir", justify="right")
    results_table.add_column("P.aer", justify="right")
    results_table.add_column("Combined", justify="right", style="green bold")

    antibiotic_aucs = []

    for ab_idx, antibiotic in enumerate(ANTIBIOTICS):
        ab_short = ANTIBIOTIC_SHORT[antibiotic]

        # Calculate combined AUC for this antibiotic
        y_val = data['y_val'][:, ab_idx]
        valid_mask = ~np.isnan(y_val)

        if valid_mask.sum() > 10 and len(np.unique(y_val[valid_mask])) > 1:
            combined_auc = roc_auc_score(y_val[valid_mask], all_val_preds[valid_mask, ab_idx])
        else:
            combined_auc = 0.5

        antibiotic_aucs.append(combined_auc)

        # Get per-species AUCs
        global_auc = results_per_model.get(f"GLOBAL_{ab_short}", {}).get('val_auc', 0.5)

        species_aucs = []
        for species_id in range(4):
            sp_short = SPECIES_SHORT[species_id]
            model_name = f"{sp_short}_{ab_short}"
            auc = results_per_model.get(model_name, {}).get('val_auc', 0.5)
            species_aucs.append(f"{auc:.3f}")

        results_table.add_row(
            ab_short,
            f"{global_auc:.4f}",
            species_aucs[0],
            species_aucs[1],
            species_aucs[2],
            species_aucs[3],
            f"{combined_auc:.4f}"
        )

    console.print(results_table)

    # Mean AUC
    mean_auc = np.mean(antibiotic_aucs)
    console.print(f"\n[bold]📊 MEAN AUC: {mean_auc:.4f}[/]")
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
        "sample_id": data['sample_ids_test'],
        **{ab: all_test_preds[:, idx] for idx, ab in enumerate(ANTIBIOTICS)}
    })
    submission_path = submissions_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    console.print(f"  ✓ Submission: {submission_path}")

    # Predictions
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    np.save(predictions_dir / "val_preds.npy", all_val_preds)
    np.save(predictions_dir / "test_preds.npy", all_test_preds)
    console.print(f"  ✓ Predictions: {predictions_dir}")

    # Results JSON
    elapsed = time.time() - start_time
    results = {
        'run_dir': str(run_dir),
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'elapsed_formatted': str(timedelta(seconds=int(elapsed))),
        'mean_auc': float(mean_auc),
        'per_antibiotic_auc': {ab: float(auc) for ab, auc in zip(ANTIBIOTICS, antibiotic_aucs)},
        'per_model_results': {k: {kk: int(vv) if isinstance(vv, (np.integer,)) else (float(vv) if isinstance(vv, (np.floating, float)) else vv)
                                   for kk, vv in v.items()}
                              for k, v in results_per_model.items()},
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}
    }

    results_path = run_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    console.print(f"  ✓ Results: {results_path}")

    # Artifacts (models + predictions)
    artifacts = {
        'models': all_models,
        'val_preds': all_val_preds,
        'test_preds': all_test_preds,
        'results': results
    }
    with open(run_dir / "artifacts.pkl", 'wb') as f:
        pickle.dump(artifacts, f)
    console.print(f"  ✓ Artifacts: {run_dir}/artifacts.pkl")

    # Final summary
    console.print(f"\n[bold green]{'═' * 60}[/]")
    console.print(f"[bold green]  ✅ COMPLETE! Elapsed: {timedelta(seconds=int(elapsed))}[/]")
    console.print(f"[bold green]  📁 All outputs in: {run_dir}[/]")
    console.print(f"[bold green]{'═' * 60}[/]")

    console.print(f"\n[bold]Kaggle submission command:[/]")
    console.print(f"[dim]kaggle competitions submit -c antimicrobial-resistance-prediction-from-maldi-tof -f {submission_path} -m 'Species-specific 32 models'[/]")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Species-specific models for AMR prediction")
    parser.add_argument("--min-samples", type=int, default=50,
                        help="Minimum samples per species for dedicated model")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction")

    args = parser.parse_args()

    config = SpeciesSpecificConfig(
        min_samples_per_species=args.min_samples,
        n_folds=args.n_folds,
        val_fraction=args.val_fraction
    )

    results = run_species_specific_training(config)

    return results


if __name__ == "__main__":
    main()
