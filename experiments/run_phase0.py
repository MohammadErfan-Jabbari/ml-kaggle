#!/usr/bin/env python3
"""
Phase 0: Establish baseline with controlled experiment framework.

This script runs the baseline experiment (raw features, no DR) to establish
reference metrics for the Dimensionality Reduction Research Pipeline.

Expected Results:
- Mean AUC: ~0.8978
- K.pneumoniae AUC: ~0.7313
"""

from pathlib import Path
from controlled_experiment import ControlledExperiment, OUTPUT_DIR


def main():
    print("=" * 60)
    print("PHASE 0: Baseline Establishment")
    print("=" * 60)
    print("\nThis establishes reference metrics for DR comparison.")
    print("No dimensionality reduction or feature scaling applied.\n")

    # Run baseline (no transformer, no scaler)
    exp = ControlledExperiment(
        name="baseline_raw_features",
        transformer=None,
        use_scaler=False
    )

    results = exp.run()

    # Print detailed summary
    exp.print_summary()

    # Save results
    output_path = OUTPUT_DIR / "phase0_baseline.json"
    exp.save_results(output_path)

    # Print key metrics for quick verification
    print("\n" + "=" * 60)
    print("BASELINE ESTABLISHED")
    print("=" * 60)
    print(f"\n  Mean AUC:         {results.mean_auc:.4f} +/- {results.mean_auc_std:.4f}")

    kpn_auc = results.per_species.get('K.pneumoniae', {}).get('mean')
    kpn_std = results.per_species.get('K.pneumoniae', {}).get('std')
    if kpn_auc is not None:
        std_str = f" +/- {kpn_std:.4f}" if kpn_std else ""
        print(f"  K.pneumoniae AUC: {kpn_auc:.4f}{std_str}")

    print(f"\n  Expected: Mean ~0.8978, K.pn ~0.7313")
    print(f"\n  Results saved to: {output_path}")

    # Validation check
    print("\n" + "-" * 60)
    print("VALIDATION:")
    if abs(results.mean_auc - 0.8978) < 0.02:
        print("  [OK] Mean AUC within expected range")
    else:
        print(f"  [WARNING] Mean AUC {results.mean_auc:.4f} differs from expected 0.8978")

    if kpn_auc and abs(kpn_auc - 0.7313) < 0.03:
        print("  [OK] K.pneumoniae AUC within expected range")
    else:
        kpn_val = kpn_auc if kpn_auc else "N/A"
        print(f"  [WARNING] K.pn AUC {kpn_val} differs from expected 0.7313")

    return results


if __name__ == "__main__":
    main()
