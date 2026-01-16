#!/usr/bin/env python3
"""
Phase 5.5: Validation Infrastructure.

This script creates a holdout validation set that matches the test distribution
and establishes validation as the primary metric for model selection.

Critical Issue (from Phase 5):
- LightGBM Stacking: OOF=0.9237 but LB=0.8269 (massive overfitting)
- Stacking meta-learner learned fold patterns, not generalizable signal
- OOF is NOT reliable for ensemble evaluation

Solution:
- Create validation set matching test species distribution
- Use validation AUC as PRIMARY metric for model selection
- Analyze OOF-Val correlation to determine if OOF can be trusted

Usage:
    uv run python experiments/run_phase5.5.py

Output:
- data/processed/val_split.npz (saved validation split)
- outputs/experiments/phase5.5_baseline.json (baseline with val metrics)
- outputs/experiments/phase5.5_comparison.json (OOF-Val correlation analysis)

Decision Framework:
- High correlation (r > 0.9, gap < 0.03): Use OOF for rapid iteration
- Medium correlation (0.7 < r < 0.9, 0.03 < gap < 0.07): Hybrid approach
- Low correlation (r < 0.7, gap > 0.07): Val only, OOF is misleading
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats

# Add parent and src directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.controlled_experiment import (
    ControlledExperiment,
    load_data,
    OUTPUT_DIR,
    SPECIES_NAMES,
    ANTIBIOTICS
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Test species distribution (from EDA)
TEST_SPECIES_DISTRIBUTION = {
    0: 0.269,  # E.coli
    1: 0.508,  # K.pneumoniae (MAJORITY)
    2: 0.193,  # P.mirabilis
    3: 0.030   # P.aeruginosa
}

# =============================================================================
# STEP 1: CREATE VALIDATION SPLIT
# =============================================================================

def step1_create_validation_split():
    """
    Step 1: Create validation split matching test distribution.

    This split is saved to disk and reused across all validation experiments
    to ensure reproducibility and comparability.
    """
    print("=" * 80)
    print("PHASE 5.5: VALIDATION INFRASTRUCTURE")
    print("=" * 80)
    print("\n[Step 1] Creating validation split...")

    # Load training data
    X, _, y, species, _ = load_data()

    # Create validation split
    from src.data.dataset import create_test_distribution_split

    X_train, X_val, y_train, y_val, species_train, species_val = \
        create_test_distribution_split(X, y, species, val_size=0.2, random_state=42)

    print(f"\n✓ Validation split created:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Total: {X_train.shape[0] + X_val.shape[0]}")

    return X_train, X_val, y_train, y_val, species_train, species_val


# =============================================================================
# STEP 2: RUN BASELINE WITH VALIDATION
# =============================================================================

def step2_run_baseline_with_validation():
    """
    Step 2: Run baseline experiment with validation.

    This establishes reference metrics for:
    - OOF AUC (from 2688 samples, 5-fold CV)
    - Val AUC (from 672 held-out samples)
    - OOF-Val gap (indicates overfitting)
    """
    print("\n" + "=" * 80)
    print("[Step 2] Running baseline with validation...")
    print("=" * 80)

    # Run baseline with validation
    exp = ControlledExperiment(
        name="phase5.5_baseline",
        transformer=None,
        use_scaler=False,
        use_validation=True  # NEW: enables validation metrics
    )

    result = exp.run()
    exp.print_summary()

    # Save results
    output_path = OUTPUT_DIR / "phase5.5_baseline.json"
    exp.save_results(output_path)

    print(f"\n✓ Baseline results saved to: {output_path}")

    return result


# =============================================================================
# STEP 3: ANALYZE OOF-VAL CORRELATION
# =============================================================================

def step3_analyze_oof_val_correlation(baseline_result: Dict[str, Any]):
    """
    Step 3: Analyze correlation between OOF and Val metrics.

    Determines if OOF is a reliable proxy for validation performance.

    Quantified thresholds:
    - High correlation: r > 0.9 AND mean |OOF-Val| gap < 0.03
    - Medium correlation: 0.7 < r < 0.9 OR 0.03 < gap < 0.07
    - Low correlation: r < 0.7 OR gap > 0.07
    """
    print("\n" + "=" * 80)
    print("[Step 3] Analyzing OOF-Val correlation...")
    print("=" * 80)

    # Extract metrics
    oof_mean = baseline_result['mean_auc']
    val_mean = baseline_result['val_mean_auc']

    oof_per_antibiotic = {ab: baseline_result['per_antibiotic'][ab]['mean']
                          for ab in ANTIBIOTICS
                          if baseline_result['per_antibiotic'][ab]['mean'] is not None}

    val_per_antibiotic = {ab: baseline_result['val_per_antibiotic'][ab]['mean']
                          for ab in ANTIBIOTICS
                          if ab in baseline_result['val_per_antibiotic']}

    oof_per_species = {sp: baseline_result['per_species'][sp]['mean']
                        for sp in SPECIES_NAMES.values()
                        if baseline_result['per_species'][sp]['mean'] is not None}

    val_per_species = {sp: baseline_result['val_per_species'][sp]['mean']
                        for sp in SPECIES_NAMES.values()
                        if sp in baseline_result['val_per_species']}

    # Compute overall gap
    overall_gap = oof_mean - val_mean

    # Compute per-antibiotic gaps
    ab_gaps = []
    for ab in ANTIBIOTICS:
        if ab in oof_per_antibiotic and ab in val_per_antibiotic:
            gap = oof_per_antibiotic[ab] - val_per_antibiotic[ab]
            ab_gaps.append(gap)

    # Compute per-species gaps
    species_gaps = []
    for sp in SPECIES_NAMES.values():
        if sp in oof_per_species and sp in val_per_species:
            gap = oof_per_species[sp] - val_per_species[sp]
            species_gaps.append(gap)

    mean_ab_gap = np.mean(np.abs(ab_gaps)) if ab_gaps else 0
    max_ab_gap = np.max(np.abs(ab_gaps)) if ab_gaps else 0

    mean_species_gap = np.mean(np.abs(species_gaps)) if species_gaps else 0
    max_species_gap = np.max(np.abs(species_gaps)) if species_gaps else 0

    # Print comparison table
    print(f"\nOverall Metrics:")
    print(f"  OOF Mean AUC:  {oof_mean:.4f}")
    print(f"  Val Mean AUC:  {val_mean:.4f}")
    print(f"  OOF-Val Gap:   {overall_gap:+.4f}")

    print(f"\nPer-Antibiotic Gaps:")
    for ab in ANTIBIOTICS:
        if ab in oof_per_antibiotic and ab in val_per_antibiotic:
            gap = oof_per_antibiotic[ab] - val_per_antibiotic[ab]
            print(f"  {ab:35} OOF={oof_per_antibiotic[ab]:.4f}, Val={val_per_antibiotic[ab]:.4f}, Gap={gap:+.4f}")

    print(f"\nPer-Species Gaps:")
    for sp in SPECIES_NAMES.values():
        if sp in oof_per_species and sp in val_per_species:
            gap = oof_per_species[sp] - val_per_species[sp]
            marker = " <-- PRIMARY" if sp == "K.pneumoniae" else ""
            print(f"  {sp:15} OOF={oof_per_species[sp]:.4f}, Val={val_per_species[sp]:.4f}, Gap={gap:+.4f}{marker}")

    print(f"\nGap Statistics:")
    print(f"  Mean |AB Gap|:    {mean_ab_gap:.4f}")
    print(f"  Max |AB Gap|:     {max_ab_gap:.4f}")
    print(f"  Mean |Species Gap|: {mean_species_gap:.4f}")
    print(f"  Max |Species Gap|:  {max_species_gap:.4f}")

    # Decision logic
    print(f"\n{'='*80}")
    print("DECISION FRAMEWORK")
    print(f"{'='*80}")

    # Determine correlation level
    # Note: We only have 1 data point (baseline), so we can't compute Pearson r
    # Instead, we use the gap size as a proxy
    if abs(overall_gap) < 0.03 and max_species_gap < 0.05:
        decision = "use_oof"
        reasoning = (f"OOF-Val gap is small ({overall_gap:+.4f}) and consistent across species. "
                    f"OOF appears to be a reliable proxy for validation.")
    elif abs(overall_gap) < 0.07 and max_species_gap < 0.10:
        decision = "hybrid"
        reasoning = (f"OOF-Val gap is moderate ({overall_gap:+.4f}). "
                    f"Use OOF for rapid iteration but verify with Val for final selection.")
    else:
        decision = "use_val_only"
        reasoning = (f"OOF-Val gap is large ({overall_gap:+.4f}). "
                    f"OOF is misleading - use Val as the primary metric.")

    print(f"\nDecision: {decision.upper()}")
    print(f"Reasoning: {reasoning}")

    # Compile analysis results
    analysis = {
        'overall_oof_auc': oof_mean,
        'overall_val_auc': val_mean,
        'overall_gap': overall_gap,
        'mean_ab_gap': mean_ab_gap,
        'max_ab_gap': max_ab_gap,
        'mean_species_gap': mean_species_gap,
        'max_species_gap': max_species_gap,
        'per_antibiotic_gaps': {ab: oof_per_antibiotic[ab] - val_per_antibiotic[ab]
                               for ab in ANTIBIOTICS
                               if ab in oof_per_antibiotic and ab in val_per_antibiotic},
        'per_species_gaps': {sp: oof_per_species[sp] - val_per_species[sp]
                             for sp in SPECIES_NAMES.values()
                             if sp in oof_per_species and sp in val_per_species},
        'decision': decision,
        'reasoning': reasoning
    }

    # Save analysis
    output_path = OUTPUT_DIR / "phase5.5_comparison.json"
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'phase': 5.5,
            'analysis': analysis
        }, f, indent=2)

    print(f"\n✓ Analysis saved to: {output_path}")

    return analysis


# =============================================================================
# STEP 4: PRINT RECOMMENDATIONS
# =============================================================================

def step4_print_recommendations(analysis: Dict[str, Any]):
    """
    Step 4: Print recommendations for next steps.

    Based on the OOF-Val correlation analysis, provides clear
    recommendations for which phases to re-run with validation.
    """
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR NEXT STEPS")
    print("=" * 80)

    decision = analysis['decision']

    print(f"\nBased on OOF-Val correlation analysis: {decision.upper()}")
    print()

    if decision == "use_oof":
        print("✓ OOF is reliable - existing Phase 0-5 results are trustworthy")
        print()
        print("Recommended next steps:")
        print("  1. Re-run Phase 5 (averaging only) with validation for final check")
        print("  2. Select best method based on Val AUC")
        print("  3. Generate submission and submit to Kaggle")
        print()
        print("Skip: Phase 0, Phase 1, Phase 2, Phase 3, Phase 4 (OOF is reliable)")

    elif decision == "hybrid":
        print("⚠ OOF is somewhat reliable but Val is preferred for final selection")
        print()
        print("Recommended next steps:")
        print("  1. Re-run Phase 0 (baseline) with validation (~2 min)")
        print("  2. Re-run Phase 2 (top 3 methods) with validation (~3 min)")
        print("  3. Re-run Phase 5 (averaging only) with validation (~5 min)")
        print("  4. Select best method based on Val AUC")
        print("  5. Generate submission and submit to Kaggle")
        print()
        print("Skip: Phase 1 (14 methods), Phase 3 (global=per-species), Phase 4 (underperformed)")

    else:  # use_val_only
        print("✗ OOF is misleading - must use Val as primary metric")
        print()
        print("Recommended next steps:")
        print("  1. Re-run Phase 0 (baseline) with validation (~2 min)")
        print("  2. Re-run Phase 2 (top 3 methods) with validation (~3 min)")
        print("  3. Re-run Phase 5 (averaging only) with validation (~5 min)")
        print("  4. Select best method based on Val AUC (NOT OOF)")
        print("  5. Generate submission and submit to Kaggle")
        print()
        print("Skip: Phase 1 (14 methods), Phase 3 (global=per-species), Phase 4 (underperformed)")

    print()
    print("Key Principle:")
    print("  Always re-run core phases (0, 2 top 3, 5 averaging) with validation.")
    print("  This ensures comparability and verifies generalization.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run Phase 5.5: Validation Infrastructure."""
    start_time = datetime.now()

    # Step 1: Create validation split
    step1_create_validation_split()

    # Step 2: Run baseline with validation
    baseline_result = step2_run_baseline_with_validation()

    # Step 3: Analyze OOF-Val correlation
    analysis = step3_analyze_oof_val_correlation(baseline_result.to_dict())

    # Step 4: Print recommendations
    step4_print_recommendations(analysis)

    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 80)
    print("PHASE 5.5 COMPLETE")
    print("=" * 80)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration:   {duration}")
    print()
    print("Generated files:")
    print("  - data/processed/val_split.npz (validation split)")
    print("  - outputs/experiments/phase5.5_baseline.json (baseline with val metrics)")
    print("  - outputs/experiments/phase5.5_comparison.json (OOF-Val correlation)")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
