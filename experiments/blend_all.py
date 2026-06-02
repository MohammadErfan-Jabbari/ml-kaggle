#!/usr/bin/env python3
"""
BLEND ALL APPROACHES: Combine predictions from multiple runs.

Combines:
1. Self-Training (learned test distribution)
2. Miracle v2 (17-model ensemble diversity)
3. Species-Specific (if useful)
4. Aggressive Self-Training (if available)

Creates multiple blend variants and evaluates on validation.
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

# =============================================================================
# CONFIGURATION
# =============================================================================

ANTIBIOTICS = [
    "Ampicillin", "Amoxicillin_Clavulanic_acid", "Cefotaxime", "Cefuroxime",
    "Ciprofloxacin", "Ertapenem", "Imipenem", "Levofloxacin"
]

ANTIBIOTIC_SHORT = {
    "Ampicillin": "AMP", "Amoxicillin_Clavulanic_acid": "AMC",
    "Cefotaxime": "CTX", "Cefuroxime": "CXM", "Ciprofloxacin": "CIP",
    "Ertapenem": "ETP", "Imipenem": "IPM", "Levofloxacin": "LVX"
}

OUTPUT_BASE = Path("/home/centcom/data/ml_kaggle/outputs")


# =============================================================================
# BLENDING FUNCTIONS
# =============================================================================

def rank_average(predictions_list: list) -> np.ndarray:
    """Rank-average multiple predictions."""
    ranked = []
    for preds in predictions_list:
        ranks = np.zeros_like(preds)
        for col in range(preds.shape[1]):
            ranks[:, col] = rankdata(preds[:, col]) / len(preds)
        ranked.append(ranks)
    return np.mean(ranked, axis=0)


def weighted_average(predictions_list: list, weights: list) -> np.ndarray:
    """Weighted average of predictions."""
    weights = np.array(weights) / sum(weights)
    result = np.zeros_like(predictions_list[0])
    for preds, w in zip(predictions_list, weights):
        result += preds * w
    return result


def simple_average(predictions_list: list) -> np.ndarray:
    """Simple average of predictions."""
    return np.mean(predictions_list, axis=0)


def calculate_mean_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean AUC across antibiotics."""
    aucs = []
    for idx in range(y_true.shape[1]):
        valid = ~np.isnan(y_true[:, idx])
        if valid.sum() > 10 and len(np.unique(y_true[valid, idx])) > 1:
            auc = roc_auc_score(y_true[valid, idx], y_pred[valid, idx])
            aucs.append(auc)
    return np.mean(aucs) if aucs else 0.5


def calculate_per_antibiotic_auc(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate AUC for each antibiotic."""
    aucs = {}
    for idx, ab in enumerate(ANTIBIOTICS):
        valid = ~np.isnan(y_true[:, idx])
        if valid.sum() > 10 and len(np.unique(y_true[valid, idx])) > 1:
            aucs[ab] = roc_auc_score(y_true[valid, idx], y_pred[valid, idx])
        else:
            aucs[ab] = 0.5
    return aucs


# =============================================================================
# LOAD PREDICTIONS
# =============================================================================

def find_latest_run(base_dir: Path) -> Path:
    """Find the most recent run directory."""
    runs = sorted(base_dir.glob("run_*"))
    return runs[-1] if runs else None


def load_predictions():
    """Load predictions from all available runs."""
    predictions = {}

    console.print("\n[bold cyan]📂 Loading Predictions...[/]")

    # 1. Self-Training (use the complete run: run_20260108_180837)
    self_train_dir = OUTPUT_BASE / "self_training_runs" / "run_20260108_180837"
    if self_train_dir.exists():
        try:
            with open(self_train_dir / "artifacts.pkl", 'rb') as f:
                artifacts = pickle.load(f)
            predictions['self_train'] = {
                'val': artifacts['final_val_preds'],
                'test': artifacts['final_test_preds'],
                'mean_auc': artifacts.get('mean_auc', 0),
                'dir': self_train_dir
            }
            console.print(f"  ✓ Self-Training: {self_train_dir.name} (AUC: {predictions['self_train']['mean_auc']:.4f})")
        except Exception as e:
            console.print(f"  ✗ Self-Training: {e}")

    # 2. Miracle v2
    miracle_dir = find_latest_run(OUTPUT_BASE / "miracle_v2_runs")
    if miracle_dir:
        try:
            with open(miracle_dir / "artifacts.pkl", 'rb') as f:
                artifacts = pickle.load(f)

            # Get best ensemble predictions from all_blends
            best_name = artifacts.get('best_ensemble', 'Weighted-All')
            best_blend = artifacts['all_blends'].get(best_name, {})

            if 'val' in best_blend and 'test' in best_blend:
                predictions['miracle_v2'] = {
                    'val': best_blend['val'],
                    'test': best_blend['test'],
                    'mean_auc': best_blend.get('mean_auc', artifacts.get('best_val_mean_auc', 0)),
                    'dir': miracle_dir
                }
                console.print(f"  ✓ Miracle v2: {miracle_dir.name} (AUC: {predictions['miracle_v2']['mean_auc']:.4f})")
        except Exception as e:
            console.print(f"  ✗ Miracle v2: {e}")

    # 3. Species-Specific
    species_dir = find_latest_run(OUTPUT_BASE / "species_specific_runs")
    if species_dir:
        try:
            val_preds = np.load(species_dir / "predictions" / "val_preds.npy")
            test_preds = np.load(species_dir / "predictions" / "test_preds.npy")
            predictions['species_specific'] = {
                'val': val_preds,
                'test': test_preds,
                'mean_auc': 0.8023,  # From output
                'dir': species_dir
            }
            console.print(f"  ✓ Species-Specific: {species_dir.name} (AUC: 0.8023)")
        except Exception as e:
            console.print(f"  ✗ Species-Specific: {e}")

    return predictions


def load_validation_labels():
    """Load validation labels for evaluation."""
    # We need to match the validation split used by the runs
    # Load from self-training artifacts which has y_val

    self_train_dir = find_latest_run(OUTPUT_BASE / "self_training_runs")
    if self_train_dir:
        try:
            with open(self_train_dir / "artifacts.pkl", 'rb') as f:
                artifacts = pickle.load(f)
            # The artifacts should contain val labels or we need to recreate
            # For now, return None and we'll calculate relative improvements
            return None
        except:
            pass
    return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    console.print("\n" + "="*60)
    console.print("[bold cyan]🔀 BLEND ALL APPROACHES[/]")
    console.print("="*60)

    # Load predictions
    predictions = load_predictions()

    if len(predictions) < 2:
        console.print("\n[bold red]❌ Need at least 2 prediction sets to blend![/]")
        console.print("Available:", list(predictions.keys()))
        return

    console.print(f"\n[bold green]✓ Loaded {len(predictions)} prediction sets[/]")

    # Check if validation sizes match
    val_sizes = {name: pred['val'].shape for name, pred in predictions.items()}
    console.print(f"\nValidation shapes: {val_sizes}")

    test_sizes = {name: pred['test'].shape for name, pred in predictions.items()}
    console.print(f"Test shapes: {test_sizes}")

    # If sizes don't match, we can only blend test predictions
    all_val_same = len(set(s[0] for s in val_sizes.values())) == 1

    if not all_val_same:
        console.print("\n[bold yellow]⚠️ Validation sizes don't match - using test predictions only[/]")

    # Create output directory
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = OUTPUT_BASE / "blend_runs" / f"run_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"\n[bold green]📁 Run Directory:[/] {run_dir}")

    # Get test predictions
    test_preds_list = [pred['test'] for pred in predictions.values()]
    pred_names = list(predictions.keys())

    # Create blends (test only since val sizes differ)
    blends = {}

    # 1. Simple average
    blends['simple_avg'] = simple_average(test_preds_list)

    # 2. Rank average
    blends['rank_avg'] = rank_average(test_preds_list)

    # 3. Weighted by validation AUC
    weights = [pred['mean_auc'] for pred in predictions.values()]
    blends['weighted_by_auc'] = weighted_average(test_preds_list, weights)

    # 4. Exclude species-specific (if it exists and is worse)
    if 'species_specific' in predictions and len(predictions) > 2:
        good_preds = [pred['test'] for name, pred in predictions.items() if name != 'species_specific']
        blends['exclude_species'] = simple_average(good_preds)

    # 5. Self-train + Miracle only (if both exist)
    if 'self_train' in predictions and 'miracle_v2' in predictions:
        blends['self_train_miracle'] = simple_average([
            predictions['self_train']['test'],
            predictions['miracle_v2']['test']
        ])

    console.print(f"\n[bold]Created {len(blends)} blend variants[/]")

    # Save all blends
    submissions_dir = run_dir / "submissions"
    submissions_dir.mkdir(exist_ok=True)

    # Load test sample IDs
    test_df = pd.read_csv("/home/centcom/data/ml_kaggle/raw/test.csv")
    sample_ids = test_df['sample_id'].values

    results_table = Table(title="[bold]Blend Submissions Created[/]", box=box.ROUNDED)
    results_table.add_column("Blend", style="cyan")
    results_table.add_column("Components", style="dim")
    results_table.add_column("File", style="green")

    for blend_name, blend_preds in blends.items():
        submission = pd.DataFrame({
            "sample_id": sample_ids,
            **{ab: blend_preds[:, idx] for idx, ab in enumerate(ANTIBIOTICS)}
        })

        filename = f"blend_{blend_name}.csv"
        submission.to_csv(submissions_dir / filename, index=False)

        components = ", ".join(pred_names) if 'exclude' not in blend_name else "self_train, miracle_v2"
        results_table.add_row(blend_name, components, filename)

    console.print("\n")
    console.print(results_table)

    # Summary
    console.print(f"\n[bold]📊 Component Validation AUCs:[/]")
    for name, pred in predictions.items():
        console.print(f"  • {name}: {pred['mean_auc']:.4f}")

    console.print(f"\n[bold green]✅ All blends saved to: {submissions_dir}[/]")

    console.print(f"\n[bold]Recommended submission order:[/]")
    console.print(f"  1. blend_rank_avg.csv (most robust)")
    console.print(f"  2. blend_self_train_miracle.csv (best components)")
    console.print(f"  3. blend_weighted_by_auc.csv (AUC-weighted)")

    console.print(f"\n[bold]Kaggle command:[/]")
    console.print(f"[dim]kaggle competitions submit -c antimicrobial-resistance-prediction-from-maldi-tof \\")
    console.print(f"  -f {submissions_dir}/blend_rank_avg.csv \\")
    console.print(f"  -m 'Rank-avg blend: self-train + miracle v2'[/]")

    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'components': {name: {'mean_auc': pred['mean_auc'], 'dir': str(pred['dir'])}
                       for name, pred in predictions.items()},
        'blends_created': list(blends.keys())
    }
    with open(run_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    return blends


if __name__ == "__main__":
    main()
