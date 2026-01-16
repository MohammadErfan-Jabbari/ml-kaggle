#!/usr/bin/env python3
"""
Phase 5.5: Re-run Phase 2 (top 3 methods) with validation.

Based on Phase 5.5 findings:
- OOF is unreliable (gap = +0.0985)
- Must use Val as the primary metric

This script re-runs the top 3 methods from Phase 2 with validation:
1. PLSRegression(n=20) - best single method from Phase 2 (OOF K.pn=0.7431)
2. LGBImportanceSelector(k=500) - second best (OOF K.pn=0.7050)
3. VarianceThreshold(t=0.005) - third best (OOF K.pn=0.7076)

Usage:
    uv run python experiments/run_phase5.5_phase2_top3.py
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent and src directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.controlled_experiment import (
    ControlledExperiment,
    OUTPUT_DIR
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import VarianceThreshold
from src.features.reducers import LGBImportanceSelector

# =============================================================================
# TOP 3 METHODS FROM PHASE 2
# =============================================================================

TOP3_METHODS = [
    {
        "name": "pls_n20_val",
        "display_name": "PLS(n=20)",
        "transformer": PLSRegression(n_components=20),
        "use_scaler": True,
        "oof_kpn_auc": 0.7431,
        "phase": "2A"
    },
    {
        "name": "lgb_k500_val",
        "display_name": "LGBImportanceSelector(k=500)",
        "transformer": LGBImportanceSelector(k=500),
        "use_scaler": False,
        "oof_kpn_auc": 0.7050,
        "phase": "1D3"
    },
    {
        "name": "var_t005_val",
        "display_name": "VarianceThreshold(t=0.005)",
        "transformer": VarianceThreshold(threshold=0.005),
        "use_scaler": False,
        "oof_kpn_auc": 0.7076,
        "phase": "2C"
    },
]

# =============================================================================
# RUN TOP 3 METHODS WITH VALIDATION
# =============================================================================

def main():
    """Run top 3 methods from Phase 2 with validation."""
    start_time = datetime.now()

    print("=" * 80)
    print("PHASE 5.5: RE-RUN PHASE 2 (TOP 3) WITH VALIDATION")
    print("=" * 80)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    results = {}

    for method_config in TOP3_METHODS:
        name = method_config["name"]
        display_name = method_config["display_name"]

        print("\n" + "=" * 80)
        print(f"Running: {display_name}")
        print("=" * 80)

        try:
            exp = ControlledExperiment(
                name=name,
                transformer=method_config["transformer"],
                use_scaler=method_config["use_scaler"],
                use_validation=True  # KEY: enables validation metrics
            )

            result = exp.run()
            exp.print_summary()

            # Save results
            output_path = OUTPUT_DIR / f"phase5.5_{name}.json"
            exp.save_results(output_path)

            # Store result
            results[name] = {
                'name': name,
                'display_name': display_name,
                'oof_mean_auc': result.mean_auc,
                'oof_K_pneumoniae_auc': result.per_species.get('K.pneumoniae', {}).get('mean'),
                'val_mean_auc': result.val_mean_auc,
                'val_K_pneumoniae_auc': result.val_K_pneumoniae_auc,
                'oof_val_gap': result.mean_auc - result.val_mean_auc if result.val_mean_auc else None,
                'config': method_config
            }

            print(f"\n✓ Results saved to: {output_path}")

        except Exception as e:
            print(f"\n✗ ERROR running {display_name}: {e}")
            import traceback
            traceback.print_exc()

    # Print comparison table
    print("\n" + "=" * 80)
    print("PHASE 5.5: TOP 3 METHODS COMPARISON")
    print("=" * 80)

    print(f"\nBaseline (from Phase 5.5):")
    print(f"  Val Mean AUC:     0.8030")
    print(f"  Val K.pn AUC:      0.6595")

    print(f"\nTop 3 Methods (with validation):")
    print(f"\n{'Method':<35} {'OOF Mean':>10} {'Val Mean':>10} {'Val K.pn':>10} {'OOF-Val':>10}")
    print("-" * 80)

    baseline_val_kpn = 0.6595

    for name, result in results.items():
        oof_mean = result['oof_mean_auc']
        val_mean = result['val_mean_auc']
        val_kpn = result['val_K_pneumoniae_auc']
        oof_val_gap = result['oof_val_gap']

        marker = " ★" if val_kpn and val_kpn > baseline_val_kpn else ""

        print(f"{result['display_name']:<35} {oof_mean:>10.4f} {val_mean:>10.4f} {val_kpn or 0:>10.4f} {oof_val_gap:>+10.4f}{marker}")

    # Find best method based on Val K.pn AUC
    best_method = max(results.items(), key=lambda x: x[1]['val_K_pneumoniae_auc'] or 0)
    best_name = best_method[0]
    best_result = best_method[1]

    print("\n" + "=" * 80)
    print("BEST METHOD (BASED ON VAL K.PNEUMONIAE AUC)")
    print("=" * 80)
    print(f"\nBest: {best_result['display_name']}")
    print(f"  Val K.pn AUC:  {best_result['val_K_pneumoniae_auc']:.4f}")
    print(f"  Val Mean AUC:  {best_result['val_mean_auc']:.4f}")
    print(f"  OOF-Val Gap:   {best_result['oof_val_gap']:+.4f}")

    baseline_kpn = 0.6595
    delta = best_result['val_K_pneumoniae_auc'] - baseline_kpn
    print(f"  Δ from baseline: {delta:+.4f}")

    # Save combined results
    combined_results = {
        'timestamp': datetime.now().isoformat(),
        'phase': '5.5_phase2_top3',
        'baseline': {
            'val_mean_auc': 0.8030,
            'val_K_pneumoniae_auc': 0.6595
        },
        'results': results,
        'best_method': best_name
    }

    output_path = OUTPUT_DIR / "phase5.5_phase2_top3_combined.json"
    with open(output_path, 'w') as f:
        json.dump(combined_results, f, indent=2, default=str)

    print(f"\n✓ Combined results saved to: {output_path}")

    # Print timing
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 80)
    print("PHASE 5.5: RE-RUN PHASE 2 (TOP 3) COMPLETE")
    print("=" * 80)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration:   {duration}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
