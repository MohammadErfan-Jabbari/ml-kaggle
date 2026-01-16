#!/usr/bin/env python3
"""
Phase 5: Ensemble & Final Selection.

This script implements ensemble methods to combine top single methods from
Phases 1-2. It generates final submission candidates for the Kaggle competition.

Usage:
    uv run python experiments/run_phase5.py

Expected runtime: ~35 minutes (or ~10 min if predictions cached)

Output:
- outputs/experiments/predictions/*.npz (5 files)
- outputs/submissions/ensemble_*.csv (5 files)
- outputs/experiments/phase5_combined_results.json
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Ensure proper imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import VarianceThreshold

from experiments.controlled_experiment import (
    ControlledExperiment,
    load_data,
    OUTPUT_DIR,
    PREDICTIONS_DIR
)
from experiments.ensemble_utils import (
    average_predictions,
    compute_weights_by_kpn_auc,
    train_stacking_lr,
    train_stacking_lgb,
    predict_stacking,
    evaluate_ensemble,
    print_ensemble_summary,
    create_submission,
    save_combined_results
)
from src.features.reducers import LGBImportanceSelector

# =============================================================================
# CONFIGURATION
# =============================================================================

# Top 5 methods from Phases 0-2 (using Phase 2 optimized configs)
TOP_METHODS = [
    {
        "name": "baseline",
        "display_name": "Baseline",
        "transformer": None,
        "use_scaler": False,
        "kpn_auc": 0.6946,
        "filename": "baseline"
    },
    {
        "name": "pls_n20",
        "display_name": "PLS(n=20)",
        "transformer": PLSRegression(n_components=20),
        "use_scaler": True,
        "kpn_auc": 0.7431,
        "filename": "pls_n20"
    },
    {
        "name": "var_t005",
        "display_name": "VarianceThreshold(t=0.005)",
        "transformer": VarianceThreshold(threshold=0.005),
        "use_scaler": False,
        "kpn_auc": 0.7076,
        "filename": "var_t005"
    },
    {
        "name": "lgb_k500",
        "display_name": "LGBImportanceSelector(k=500)",
        "transformer": LGBImportanceSelector(k=500),
        "use_scaler": False,
        "kpn_auc": 0.7050,
        "filename": "lgb_k500"
    },
    {
        "name": "pls_n15",
        "display_name": "PLS(n=15)",
        "transformer": PLSRegression(n_components=15),
        "use_scaler": True,
        "kpn_auc": 0.7329,
        "filename": "pls_n15"
    },
]

# Top 3 for averaging
TOP3_METHODS = ["pls_n20", "var_t005", "lgb_k500"]

# =============================================================================
# PREDICTION REGENERATION
# =============================================================================

def regenerate_predictions_for_top_methods(
    force_regenerate: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Regenerate predictions for top 5 methods.

    This runs ControlledExperiment for each method and saves predictions
    to .npz files. Skips if predictions already exist (unless force_regenerate=True).

    Args:
        force_regenerate: If True, regenerate even if predictions exist

    Returns:
        Dict mapping method name to prediction dict with keys:
        - 'oof_predictions': (3360, 8) array
        - 'test_predictions': (1000, 8) array
        - 'metrics': dict of metrics
    """
    predictions_cache = {}

    for method_config in TOP_METHODS:
        name = method_config["name"]
        display_name = method_config["display_name"]
        pred_path = PREDICTIONS_DIR / f"{method_config['filename']}.npz"

        # Check if predictions already exist
        if pred_path.exists() and not force_regenerate:
            print(f"\n[{name}] Loading cached predictions from {pred_path}")
            try:
                loaded = ControlledExperiment.load_predictions(pred_path)
                predictions_cache[name] = {
                    'oof_predictions': loaded['oof_predictions'],
                    'test_predictions': loaded['test_predictions'],
                    'name': loaded['name']
                }
                print(f"  Loaded: OOF shape={loaded['oof_predictions'].shape}, "
                      f"Test shape={loaded['test_predictions'].shape}")
                continue
            except Exception as e:
                print(f"  ERROR loading cache: {e}, regenerating...")

        # Regenerate predictions
        print(f"\n[{name}] Regenerating predictions: {display_name}")
        print("-" * 60)

        try:
            exp = ControlledExperiment(
                name=f"phase5_regen_{name}",
                transformer=method_config["transformer"],
                use_scaler=method_config["use_scaler"]
            )
            result = exp.run()
            exp.print_summary()

            # Save predictions to .npz
            exp.save_predictions(pred_path)

            # Cache in memory
            predictions_cache[name] = {
                'oof_predictions': result.oof_predictions,
                'test_predictions': result.test_predictions,
                'name': result.name,
                'metrics': result.to_dict()
            }

            print(f"  Predictions saved to: {pred_path}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise

    return predictions_cache


# =============================================================================
# 5A: PREDICTION AVERAGING
# =============================================================================

def run_phase5a_averaging(
    predictions_cache: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Run 5A1-5A3 averaging experiments.

    Args:
        predictions_cache: Dict of method_name -> prediction_dict

    Returns:
        Dict of experiment_id -> prediction_dict
    """
    print("\n" + "=" * 80)
    print("5A: PREDICTION AVERAGING")
    print("=" * 80)

    avg_results = {}

    # Prepare top 3 predictions
    top3_preds = [predictions_cache[name] for name in TOP3_METHODS]
    top3_kpn_aucs = [next(m['kpn_auc'] for m in TOP_METHODS if m['name'] == name)
                      for name in TOP3_METHODS]

    # 5A1: Equal weights
    print("\n[5A1] Equal weights averaging")
    print(f"  Methods: {[m['display_name'] for m in TOP_METHODS if m['name'] in TOP3_METHODS]}")
    print(f"  Weights: [1/3, 1/3, 1/3]")

    pred_5a1 = average_predictions(top3_preds, weights=[1/3, 1/3, 1/3])
    avg_results['5A1'] = {
        **pred_5a1,
        'name': '5A1_equal_weights',
        'display_name': '5A1: Equal Weights',
        'config': {
            'ensemble_type': 'averaging',
            'weighting': 'equal',
            'base_models': TOP3_METHODS,
            'weights': [1/3, 1/3, 1/3]
        }
    }

    # 5A2: Weighted by K.pn AUC
    print("\n[5A2] Weighted by K.pn AUC averaging")
    print(f"  Methods: {[m['display_name'] for m in TOP_METHODS if m['name'] in TOP3_METHODS]}")
    print(f"  K.pn AUCs: {top3_kpn_aucs}")

    weights_5a2 = compute_weights_by_kpn_auc(top3_kpn_aucs)
    print(f"  Weights: {[f'{w:.3f}' for w in weights_5a2]}")

    pred_5a2 = average_predictions(top3_preds, weights=weights_5a2)
    avg_results['5A2'] = {
        **pred_5a2,
        'name': '5A2_weighted_by_kpn',
        'display_name': '5A2: Weighted by K.pn AUC',
        'config': {
            'ensemble_type': 'averaging',
            'weighting': 'kpn_auc',
            'base_models': TOP3_METHODS,
            'kpn_aucs': top3_kpn_aucs,
            'weights': weights_5a2
        }
    }

    # 5A3: Best single only (PLS20)
    print("\n[5A3] Best single method (PLS n=20)")
    best_name = 'pls_n20'
    best_method = next(m for m in TOP_METHODS if m['name'] == best_name)
    print(f"  Method: {best_method['display_name']}")
    print(f"  K.pn AUC: {best_method['kpn_auc']:.4f}")

    pred_5a3 = predictions_cache[best_name].copy()
    pred_5a3['name'] = '5A3_best_single'
    pred_5a3['display_name'] = '5A3: Best Single (PLS n=20)'
    pred_5a3['config'] = {
        'ensemble_type': 'single',
        'base_model': best_name,
        'kpn_auc': best_method['kpn_auc']
    }

    avg_results['5A3'] = pred_5a3

    return avg_results


# =============================================================================
# 5B: STACKING
# =============================================================================

def run_phase5b_stacking(
    predictions_cache: Dict[str, Dict[str, Any]],
    y_train: np.ndarray,
    species_train: np.ndarray
) -> Dict[str, Dict[str, Any]]:
    """
    Run 5B1-5B2 stacking experiments.

    Args:
        predictions_cache: Dict of method_name -> prediction_dict
        y_train: True labels (3360, 8)
        species_train: Species IDs (3360,)

    Returns:
        Dict of experiment_id -> prediction_dict
    """
    print("\n" + "=" * 80)
    print("5B: STACKING")
    print("=" * 80)

    stack_results = {}

    # Prepare base model predictions (use all 5 for diversity)
    all_names = [m['name'] for m in TOP_METHODS]
    oof_preds = [predictions_cache[name]['oof_predictions'] for name in all_names]
    test_preds = [predictions_cache[name]['test_predictions'] for name in all_names]

    print(f"\nUsing {len(all_names)} base models:")
    for i, name in enumerate(all_names):
        method = next(m for m in TOP_METHODS if m['name'] == name)
        print(f"  {i+1}. {method['display_name']} (K.pn AUC = {method['kpn_auc']:.4f})")

    # 5B1: Logistic Regression
    print("\n[5B1] LogisticRegression meta-learner")
    print("-" * 60)

    meta_lr = train_stacking_lr(oof_preds, y_train, species_train, verbose=True)

    # Generate predictions
    test_5b1, oof_5b1 = predict_stacking(
        meta_lr,
        test_preds,
        oof_predictions_list=oof_preds,
        fallback_to_averaging=True
    )

    stack_results['5B1'] = {
        'oof_predictions': oof_5b1,
        'test_predictions': test_5b1,
        'name': '5B1_lr_stacking',
        'display_name': '5B1: LogisticRegression Stacking',
        'config': {
            'ensemble_type': 'stacking',
            'meta_learner': 'LogisticRegression',
            'base_models': all_names,
            'n_base_models': len(all_names)
        }
    }

    # 5B2: LightGBM
    print("\n[5B2] LightGBM meta-learner")
    print("-" * 60)

    meta_lgb = train_stacking_lgb(oof_preds, y_train, species_train, verbose=True)

    # Generate predictions
    test_5b2, oof_5b2 = predict_stacking(
        meta_lgb,
        test_preds,
        oof_predictions_list=oof_preds,
        fallback_to_averaging=True
    )

    stack_results['5B2'] = {
        'oof_predictions': oof_5b2,
        'test_predictions': test_5b2,
        'name': '5B2_lgb_stacking',
        'display_name': '5B2: LightGBM Stacking',
        'config': {
            'ensemble_type': 'stacking',
            'meta_learner': 'LightGBM',
            'base_models': all_names,
            'n_base_models': len(all_names)
        }
    }

    return stack_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run Phase 5 ensemble experiments."""
    start_time = datetime.now()
    print("=" * 80)
    print("PHASE 5: ENSEMBLE & FINAL SELECTION")
    print("=" * 80)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Load data (needed for evaluation and submission)
    print("\n[Loading data...]")
    X_train, X_test, y_train, species_train, species_test = load_data()
    test_df = pd.read_csv("/home/erfan/data/ml_kaggle/raw/test.csv")
    sample_ids = test_df["sample_id"].values

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Test samples: {len(sample_ids)}")

    # Step 1: Regenerate predictions for top methods
    print("\n" + "=" * 80)
    print("STEP 1: GENERATE PREDICTIONS FOR TOP 5 METHODS")
    print("=" * 80)

    predictions_cache = regenerate_predictions_for_top_methods(force_regenerate=False)

    # Verify all predictions loaded/generated
    print(f"\n✓ Loaded {len(predictions_cache)} prediction sets:")
    for name, pred_dict in predictions_cache.items():
        oof_shape = pred_dict['oof_predictions'].shape
        test_shape = pred_dict['test_predictions'].shape
        print(f"  {name:15} OOF: {oof_shape}, Test: {test_shape}")

    # Step 2: Run averaging experiments
    avg_results = run_phase5a_averaging(predictions_cache)

    # Step 3: Run stacking experiments
    stack_results = run_phase5b_stacking(predictions_cache, y_train, species_train)

    # Combine all results
    all_predictions = {**avg_results, **stack_results}

    # Step 4: Evaluate all ensembles
    print("\n" + "=" * 80)
    print("STEP 4: EVALUATE ENSEMBLES")
    print("=" * 80)

    all_results = {}

    for exp_id, pred_dict in all_predictions.items():
        print(f"\n[{exp_id}] {pred_dict.get('display_name', exp_id)}")

        result = evaluate_ensemble(
            pred_dict['oof_predictions'],
            y_train,
            species_train,
            pred_dict['test_predictions'],
            species_test,
            pred_dict['name'],
            config=pred_dict.get('config', {})
        )

        all_results[exp_id] = result

        print(f"  Mean AUC: {result['mean_auc']:.4f}")
        print(f"  K.pn AUC: {result.get('K.pneumoniae_auc', 0):.4f}")

    # Step 5: Generate submissions
    print("\n" + "=" * 80)
    print("STEP 5: GENERATE SUBMISSIONS")
    print("=" * 80)

    submission_dir = Path("/sata_disk/users/erfan/ml_kaggle/outputs/submissions")
    submission_dir.mkdir(parents=True, exist_ok=True)

    for exp_id, result in all_results.items():
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f"ensemble_{exp_id.lower()}_{timestamp}.csv"
        output_path = submission_dir / filename

        print(f"\n[{exp_id}] Creating submission: {filename}")
        create_submission(
            result['test_predictions'],
            sample_ids,
            output_path,
            result['name']
        )

    # Step 6: Print summary and save combined results
    baseline_kpn = 0.6946
    print_ensemble_summary(all_results, baseline_kpn=baseline_kpn)

    # Save combined results
    print("\n" + "=" * 80)
    print("STEP 6: SAVE COMBINED RESULTS")
    print("=" * 80)

    save_combined_results(all_results)

    # Print timing
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 80)
    print("PHASE 5 COMPLETE")
    print("=" * 80)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration}")

    print("\nGenerated files:")
    print(f"  Predictions: {len(predictions_cache)} .npz files in {PREDICTIONS_DIR}/")
    print(f"  Submissions: {len(all_results)} .csv files in {submission_dir}/")
    print(f"  Results: {OUTPUT_DIR}/phase5_combined_results.json")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Review submission files in outputs/submissions/")
    print("2. Select best ensemble based on K.pn AUC")
    print("3. Submit to Kaggle:")
    for exp_id, result in all_results.items():
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f"ensemble_{exp_id.lower()}_{timestamp}.csv"
        print(f"   kaggle competitions submit -c antimicrobial-resistance-prediction-from-maldi-tof \\")
        print(f"     -f outputs/submissions/{filename} \\")
        print(f"     -m \"Phase 5 {result['name']}\"")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
