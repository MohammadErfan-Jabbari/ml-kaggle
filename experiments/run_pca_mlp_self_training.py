#!/usr/bin/env python3
"""
PCA + MLP Only Self-Training

Key idea: MLP benefits more from additional training data than trees.
Use PCA to reduce dimensions, then MLP with self-training.
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime
import json
import pickle
import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, MofNCompleteColumn
from rich import box

warnings.filterwarnings("ignore")
console = Console()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data.dataset import ANTIBIOTICS, load_validation_split, load_train_data, split_features_targets

DATA_DIR = Path("/home/centcom/data/ml_kaggle/raw")
OUTPUT_DIR = Path("/home/centcom/data/ml_kaggle/outputs")

ANTIBIOTIC_SHORT = {
    "Ampicillin": "AMP", "Amoxicillin_Clavulanic_acid": "AMC",
    "Cefotaxime": "CTX", "Cefuroxime": "CXM", "Ciprofloxacin": "CIP",
    "Ertapenem": "ETP", "Imipenem": "IPM", "Levofloxacin": "LVX"
}

SPECIES_NAMES = {0: "E.coli", 1: "K.pneumoniae", 2: "P.mirabilis", 3: "P.aeruginosa"}

INTRINSIC_RESISTANCE = {
    3: ["Ampicillin", "Amoxicillin_Clavulanic_acid", "Ertapenem", "Cefotaxime", "Cefuroxime"],
    2: ["Imipenem"],
}


def apply_intrinsic_rules(preds, species, antibiotic):
    preds = preds.copy()
    for species_id, abs_list in INTRINSIC_RESISTANCE.items():
        if antibiotic in abs_list:
            preds[species == species_id] = 1.0
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-components", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--conf-high", type=float, default=0.85)
    parser.add_argument("--conf-low", type=float, default=0.15)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    console.print("\n[bold cyan]🧠 PCA + MLP Self-Training[/]")
    console.print(f"Config: {args.n_components} components, {args.max_iter} iterations\n")

    # Create run directory
    run_name = f"pca_mlp_{args.n_components}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = OUTPUT_DIR / "pca_mlp_runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    console.print("[bold]Loading data...[/]")
    train_df = load_train_data()
    X_full, y_full, metadata = split_features_targets(train_df)
    species_full = metadata[:, 1].astype(np.int32)

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    feature_cols = [f"maldi_feature_{i}" for i in range(6000)]
    X_test = test_df[feature_cols].values.astype(np.float32)
    species_test = test_df["species_id"].values.astype(np.int32)
    sample_ids = test_df["sample_id"].values

    # Load validation split
    X_train, X_val, y_train, y_val, species_train, species_val = load_validation_split()
    console.print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Combine all data for PCA
    X_all = np.vstack([X_train, X_val, X_test])
    n_train, n_val, n_test = len(X_train), len(X_val), len(X_test)

    # Remove constant features
    variance = np.var(X_all, axis=0)
    mask = variance > 1e-8
    X_all = X_all[:, mask]
    console.print(f"  Features after removing constants: {X_all.shape[1]}")

    # Standardize
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)

    # PCA on ALL data (transductive)
    console.print(f"\n[bold]Fitting PCA on all data ({args.n_components} components)...[/]")
    pca = PCA(n_components=args.n_components, random_state=42)
    X_all_pca = pca.fit_transform(X_all_scaled)
    console.print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    # Split back
    X_train_pca = X_all_pca[:n_train]
    X_val_pca = X_all_pca[n_train:n_train+n_val]
    X_test_pca = X_all_pca[n_train+n_val:]

    # Results storage
    final_val_preds = np.zeros((n_val, len(ANTIBIOTICS)))
    final_test_preds = np.zeros((n_test, len(ANTIBIOTICS)))
    results = {}

    console.print("\n[bold cyan]Starting self-training loop...[/]\n")

    for ab_idx, antibiotic in enumerate(ANTIBIOTICS):
        short = ANTIBIOTIC_SHORT[antibiotic]
        console.print(f"[bold]═══ {short}: {antibiotic} ({ab_idx+1}/8) ═══[/]")

        # Initialize labeled pool
        X_labeled = X_train_pca.copy()
        y_labeled = y_train[:, ab_idx].copy()
        species_labeled = species_train.copy()

        # Unlabeled pool (test)
        X_unlabeled = X_test_pca.copy()
        species_unlabeled = species_test.copy()
        pseudo_mask = np.zeros(n_test, dtype=bool)
        pseudo_labels = np.full(n_test, np.nan)

        # Get valid labeled samples
        valid_mask = ~np.isnan(y_labeled)
        n_valid = valid_mask.sum()
        console.print(f"  Initial labeled: {n_valid}")

        best_val_auc = 0.0
        best_val_preds = None
        best_test_preds = None

        for iteration in range(args.max_iter):
            # Current training data
            X_curr = X_labeled[valid_mask]
            y_curr = y_labeled[valid_mask]
            species_curr = species_labeled[valid_mask]

            if len(np.unique(y_curr)) < 2:
                console.print(f"  [yellow]Only one class, skipping[/]")
                break

            # K-fold MLP training
            skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
            val_preds_folds = []
            test_preds_folds = []
            unlabeled_preds_folds = []

            for train_idx, _ in skf.split(X_curr, species_curr):
                mlp = MLPClassifier(
                    hidden_layer_sizes=(256, 128),
                    activation='relu',
                    alpha=0.01,
                    learning_rate_init=0.001,
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=10,
                    random_state=42
                )
                mlp.fit(X_curr[train_idx], y_curr[train_idx].astype(int))

                val_preds_folds.append(mlp.predict_proba(X_val_pca)[:, 1])
                test_preds_folds.append(mlp.predict_proba(X_test_pca)[:, 1])
                unlabeled_preds_folds.append(mlp.predict_proba(X_unlabeled)[:, 1])

            val_preds = np.mean(val_preds_folds, axis=0)
            test_preds = np.mean(test_preds_folds, axis=0)
            unlabeled_preds = np.mean(unlabeled_preds_folds, axis=0)

            # Apply intrinsic rules
            val_preds = apply_intrinsic_rules(val_preds, species_val, antibiotic)
            test_preds = apply_intrinsic_rules(test_preds, species_test, antibiotic)

            # Compute val AUC
            val_labels = y_val[:, ab_idx]
            val_mask_valid = ~np.isnan(val_labels)
            if val_mask_valid.sum() > 10 and len(np.unique(val_labels[val_mask_valid])) > 1:
                val_auc = roc_auc_score(val_labels[val_mask_valid], val_preds[val_mask_valid])
            else:
                val_auc = 0.5

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_val_preds = val_preds.copy()
                best_test_preds = test_preds.copy()

            # Select new pseudo-labels
            high_conf_pos = (unlabeled_preds >= args.conf_high) & (~pseudo_mask)
            high_conf_neg = (unlabeled_preds <= args.conf_low) & (~pseudo_mask)
            n_new = high_conf_pos.sum() + high_conf_neg.sum()

            console.print(f"  Iter {iteration+1}: Val AUC={val_auc:.4f}, +{n_new} pseudo-labels")

            if n_new < 10:
                console.print(f"  [yellow]Stopping early: only {n_new} new labels[/]")
                break

            # Add pseudo-labels
            new_indices = np.where(high_conf_pos | high_conf_neg)[0]
            new_labels = np.where(high_conf_pos[new_indices], 1.0, 0.0)

            X_labeled = np.vstack([X_labeled, X_unlabeled[new_indices]])
            y_labeled = np.concatenate([y_labeled, new_labels])
            species_labeled = np.concatenate([species_labeled, species_unlabeled[new_indices]])
            valid_mask = np.concatenate([valid_mask, np.ones(len(new_indices), dtype=bool)])

            pseudo_mask[new_indices] = True
            pseudo_labels[new_indices] = new_labels

        # Store results
        final_val_preds[:, ab_idx] = best_val_preds if best_val_preds is not None else val_preds
        final_test_preds[:, ab_idx] = best_test_preds if best_test_preds is not None else test_preds

        results[antibiotic] = {
            'best_auc': float(best_val_auc),
            'n_pseudo': int(pseudo_mask.sum())
        }
        console.print(f"  [green]Best AUC: {best_val_auc:.4f}, Pseudo-labels: {pseudo_mask.sum()}[/]\n")

    # Final summary
    console.print("\n[bold green]═══ Final Results ═══[/]")

    table = Table(box=box.ROUNDED)
    table.add_column("Antibiotic")
    table.add_column("Val AUC", justify="right")

    for ab in ANTIBIOTICS:
        table.add_row(ANTIBIOTIC_SHORT[ab], f"{results[ab]['best_auc']:.4f}")

    console.print(table)

    mean_auc = np.mean([r['best_auc'] for r in results.values()])
    console.print(f"\n[bold]Val Mean AUC: {mean_auc:.4f}[/]")
    console.print(f"Target: 0.83862")
    console.print(f"Gap: {mean_auc - 0.83862:+.4f}")

    # Save submission
    submissions_dir = run_dir / "submissions"
    submissions_dir.mkdir(exist_ok=True)

    submission = pd.DataFrame({
        "sample_id": sample_ids,
        **{ab: final_test_preds[:, idx] for idx, ab in enumerate(ANTIBIOTICS)}
    })
    sub_path = submissions_dir / "submission.csv"
    submission.to_csv(sub_path, index=False)
    console.print(f"\n[bold]Saved: {sub_path}[/]")

    # Save results
    with open(run_dir / "results.json", 'w') as f:
        json.dump({
            'mean_auc': float(mean_auc),
            'n_components': args.n_components,
            'per_antibiotic': results
        }, f, indent=2)

    console.print(f"\n[bold green]Done! Run dir: {run_dir}[/]")


if __name__ == "__main__":
    main()
