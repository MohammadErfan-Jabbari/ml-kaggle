#!/usr/bin/env python3
"""
EVALUATE ALL BLENDS: Compare all runs and create optimal blends.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

OUTPUT_BASE = Path("/home/centcom/data/ml_kaggle/outputs")
DATA_DIR = Path("/home/centcom/data/ml_kaggle/raw")

ANTIBIOTICS = [
    "Ampicillin", "Amoxicillin_Clavulanic_acid", "Cefotaxime", "Cefuroxime",
    "Ciprofloxacin", "Ertapenem", "Imipenem", "Levofloxacin"
]

AB_SHORT = ["AMP", "AMC", "CTX", "CXM", "CIP", "ETP", "IPM", "LVX"]


def rank_transform(preds: np.ndarray) -> np.ndarray:
    """Rank transform predictions (per column)."""
    ranks = np.zeros_like(preds)
    for col in range(preds.shape[1]):
        ranks[:, col] = rankdata(preds[:, col]) / len(preds)
    return ranks


def main():
    console.print("\n" + "="*70)
    console.print("[bold cyan]📊 COMPREHENSIVE BLEND EVALUATION[/]")
    console.print("="*70)

    # =========================================================================
    # LOAD ALL COMPONENTS
    # =========================================================================
    console.print("\n[bold]Step 1: Load all run results[/]")

    components = {}

    # 1. Self-training (original - conservative thresholds)
    st_orig_dir = OUTPUT_BASE / "self_training_runs" / "run_20260108_180837"
    with open(st_orig_dir / "artifacts.pkl", 'rb') as f:
        st_orig = pickle.load(f)
    components['st_original'] = {
        'test_preds': st_orig['final_test_preds'],
        'mean_auc': st_orig['mean_auc'],
        'desc': 'Self-Train (0.85/0.15)'
    }
    console.print(f"  ✓ ST Original: {components['st_original']['mean_auc']:.4f}")

    # 2. Self-training (aggressive thresholds)
    st_agg_dir = OUTPUT_BASE / "self_training_runs" / "run_20260108_193910"
    with open(st_agg_dir / "artifacts.pkl", 'rb') as f:
        st_agg = pickle.load(f)
    components['st_aggressive'] = {
        'test_preds': st_agg['final_test_preds'],
        'mean_auc': st_agg['mean_auc'],
        'desc': 'Self-Train (0.70/0.30)'
    }
    console.print(f"  ✓ ST Aggressive: {components['st_aggressive']['mean_auc']:.4f}")

    # 3. Miracle v2
    m2_dir = OUTPUT_BASE / "miracle_v2_runs" / "run_20260108_180757"
    with open(m2_dir / "artifacts.pkl", 'rb') as f:
        m2 = pickle.load(f)
    best_blend = m2['all_blends']['Weighted-All']
    components['miracle_v2'] = {
        'test_preds': best_blend['test'],
        'mean_auc': best_blend['mean_auc'],
        'desc': 'Miracle v2 (17 models)'
    }
    console.print(f"  ✓ Miracle v2: {components['miracle_v2']['mean_auc']:.4f}")

    # 4. Species-specific
    sp_dir = OUTPUT_BASE / "species_specific_runs" / "run_20260108_194209"
    sp_test = np.load(sp_dir / "predictions" / "test_preds.npy")
    components['species_specific'] = {
        'test_preds': sp_test,
        'mean_auc': 0.8023,
        'desc': 'Species-Specific (32 models)'
    }
    console.print(f"  ✓ Species-Specific: {components['species_specific']['mean_auc']:.4f}")

    # =========================================================================
    # COMPONENT COMPARISON TABLE
    # =========================================================================
    console.print("\n[bold]Component Performance Ranking:[/]")

    comp_table = Table(box=box.DOUBLE_EDGE)
    comp_table.add_column("Rank", justify="center", style="bold")
    comp_table.add_column("Component", style="cyan")
    comp_table.add_column("Val Mean AUC", justify="right", style="green bold")
    comp_table.add_column("vs Target", justify="right")

    sorted_comps = sorted(components.items(), key=lambda x: x[1]['mean_auc'], reverse=True)
    for rank, (name, data) in enumerate(sorted_comps, 1):
        diff = data['mean_auc'] - 0.83862
        diff_str = f"{diff:+.4f}"
        diff_style = "green" if diff > 0 else "red"
        comp_table.add_row(
            str(rank),
            f"{name}: {data['desc']}",
            f"{data['mean_auc']:.4f}",
            f"[{diff_style}]{diff_str}[/]"
        )

    console.print(comp_table)

    # =========================================================================
    # CREATE BLENDS
    # =========================================================================
    console.print("\n[bold]Step 2: Create blend variants[/]")

    test_preds = {name: data['test_preds'] for name, data in components.items()}
    aucs = {name: data['mean_auc'] for name, data in components.items()}

    blends = {}

    # Best 2 components (ST original + Miracle v2)
    blends['best2_avg'] = {
        'preds': np.mean([test_preds['st_original'], test_preds['miracle_v2']], axis=0),
        'components': ['st_original', 'miracle_v2'],
        'weights': [0.5, 0.5]
    }

    blends['best2_rank'] = {
        'preds': np.mean([rank_transform(test_preds['st_original']),
                         rank_transform(test_preds['miracle_v2'])], axis=0),
        'components': ['st_original', 'miracle_v2'],
        'weights': [0.5, 0.5]
    }

    # Weight heavily on best single (ST original)
    blends['st_heavy'] = {
        'preds': 0.7 * test_preds['st_original'] + 0.3 * test_preds['miracle_v2'],
        'components': ['st_original', 'miracle_v2'],
        'weights': [0.7, 0.3]
    }

    # All 3 self-train variants + miracle (exclude species-specific)
    blends['all_good'] = {
        'preds': np.mean([test_preds['st_original'], test_preds['st_aggressive'],
                         test_preds['miracle_v2']], axis=0),
        'components': ['st_original', 'st_aggressive', 'miracle_v2'],
        'weights': [1/3, 1/3, 1/3]
    }

    # Weighted by AUC (exclude species-specific)
    good_comps = ['st_original', 'st_aggressive', 'miracle_v2']
    good_aucs = [aucs[c] for c in good_comps]
    total_auc = sum(good_aucs)
    good_weights = [a/total_auc for a in good_aucs]
    blends['weighted_good'] = {
        'preds': sum(w * test_preds[c] for w, c in zip(good_weights, good_comps)),
        'components': good_comps,
        'weights': good_weights
    }

    # Include all 4
    all_comps = list(test_preds.keys())
    all_aucs = [aucs[c] for c in all_comps]
    total_all = sum(all_aucs)
    all_weights = [a/total_all for a in all_aucs]
    blends['weighted_all'] = {
        'preds': sum(w * test_preds[c] for w, c in zip(all_weights, all_comps)),
        'components': all_comps,
        'weights': all_weights
    }

    # Rank average of all 4
    blends['rank_all'] = {
        'preds': np.mean([rank_transform(p) for p in test_preds.values()], axis=0),
        'components': all_comps,
        'weights': [0.25]*4
    }

    # =========================================================================
    # ESTIMATE BLEND PERFORMANCE
    # =========================================================================
    console.print("\n[bold]Blend Performance Estimates:[/]")
    console.print("[dim](Estimated as weighted average of component validation AUCs)[/]")

    blend_table = Table(box=box.ROUNDED)
    blend_table.add_column("Blend", style="cyan", width=20)
    blend_table.add_column("Components", width=30)
    blend_table.add_column("Est. AUC", justify="right", style="green bold")
    blend_table.add_column("vs Target", justify="right")

    blend_results = []
    for blend_name, blend_data in blends.items():
        comp_aucs = [aucs[c] for c in blend_data['components']]
        est_auc = sum(w * a for w, a in zip(blend_data['weights'], comp_aucs))
        blend_results.append((blend_name, est_auc, blend_data))

    # Sort by estimated AUC
    blend_results.sort(key=lambda x: x[1], reverse=True)

    for blend_name, est_auc, blend_data in blend_results:
        diff = est_auc - 0.83862
        diff_str = f"{diff:+.4f}"
        diff_style = "green" if diff > 0 else "red"
        comp_str = " + ".join([c.replace('_', ' ').title()[:12] for c in blend_data['components']])

        blend_table.add_row(
            blend_name,
            comp_str[:30],
            f"{est_auc:.4f}",
            f"[{diff_style}]{diff_str}[/]"
        )

    console.print(blend_table)

    # =========================================================================
    # SAVE SUBMISSION FILES
    # =========================================================================
    console.print("\n[bold]Step 3: Save submission files[/]")

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    sample_ids = test_df['sample_id'].values

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    blend_dir = OUTPUT_BASE / "blend_runs" / f"run_{timestamp}"
    blend_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir = blend_dir / "submissions"
    submissions_dir.mkdir(exist_ok=True)

    for blend_name, est_auc, blend_data in blend_results:
        submission = pd.DataFrame({
            "sample_id": sample_ids,
            **{ab: blend_data['preds'][:, idx] for idx, ab in enumerate(ANTIBIOTICS)}
        })
        path = submissions_dir / f"{blend_name}.csv"
        submission.to_csv(path, index=False)

    console.print(f"  ✓ Saved {len(blends)} blends to: {submissions_dir}")

    # Also save individual component submissions
    for comp_name, comp_data in components.items():
        submission = pd.DataFrame({
            "sample_id": sample_ids,
            **{ab: comp_data['test_preds'][:, idx] for idx, ab in enumerate(ANTIBIOTICS)}
        })
        path = submissions_dir / f"component_{comp_name}.csv"
        submission.to_csv(path, index=False)

    console.print(f"  ✓ Saved {len(components)} component submissions")

    # =========================================================================
    # FINAL RECOMMENDATIONS
    # =========================================================================
    console.print("\n" + "="*70)
    console.print("[bold green]📋 SUBMISSION RECOMMENDATIONS[/]")
    console.print("="*70)

    console.print("\n[bold]Top 3 submissions to try:[/]")
    for i, (blend_name, est_auc, _) in enumerate(blend_results[:3], 1):
        console.print(f"  {i}. [cyan]{blend_name}.csv[/] (Est. AUC: {est_auc:.4f})")

    console.print(f"\n[bold]All files in:[/] {submissions_dir}")

    console.print("\n[bold]Quick submit command:[/]")
    best_blend = blend_results[0][0]
    console.print(f"kaggle competitions submit -c antimicrobial-resistance-prediction-from-maldi-tof \\")
    console.print(f"  -f {submissions_dir}/{best_blend}.csv \\")
    console.print(f"  -m 'Blend: {best_blend}'")

    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'components': {name: {'mean_auc': float(data['mean_auc']), 'desc': data['desc']}
                       for name, data in components.items()},
        'blends': {name: {'est_auc': float(auc), 'components': data['components']}
                   for name, auc, data in blend_results}
    }
    import json
    with open(blend_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    return blend_results


if __name__ == "__main__":
    main()
