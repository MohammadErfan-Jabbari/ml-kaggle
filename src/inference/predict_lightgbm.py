#!/usr/bin/env python3
"""
Generate predictions using trained LightGBM baseline models.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import ANTIBIOTICS
from utils.submission_tracker import generate_submission_name, log_submission


# Antibiotic index mapping for intrinsic resistance rules
ANTIBIOTIC_INDICES = {antibiotic: idx for idx, antibiotic in enumerate(ANTIBIOTICS)}


def apply_intrinsic_resistance_rules(predictions: np.ndarray, species_ids: np.ndarray) -> np.ndarray:
    """
    Override ML predictions with biologically-guaranteed intrinsic resistance.

    Rules (from EDA analysis):
    - P. aeruginosa (species_id=3): 100% resistant to 5 antibiotics
    - P. mirabilis (species_id=2): 97.2% resistant to Imipenem

    Args:
        predictions: (n_samples, 8) numpy array of ML predictions
        species_ids: (n_samples,) numpy array of species identifiers

    Returns:
        Modified predictions with rules applied
    """
    predictions = predictions.copy()

    # P. aeruginosa (species_id=3) intrinsic resistance
    # These bacteria are ALWAYS resistant to these antibiotics
    pa_mask = (species_ids == 3)
    predictions[pa_mask, ANTIBIOTIC_INDICES["Ampicillin"]] = 1.0
    predictions[pa_mask, ANTIBIOTIC_INDICES["Amoxicillin_Clavulanic_acid"]] = 1.0
    predictions[pa_mask, ANTIBIOTIC_INDICES["Ertapenem"]] = 1.0
    predictions[pa_mask, ANTIBIOTIC_INDICES["Cefotaxime"]] = 1.0
    predictions[pa_mask, ANTIBIOTIC_INDICES["Cefuroxime"]] = 1.0

    # P. mirabilis (species_id=2) intrinsic resistance
    # 97.2% resistant to Imipenem in training data
    pm_mask = (species_ids == 2)
    predictions[pm_mask, ANTIBIOTIC_INDICES["Imipenem"]] = 1.0

    n_pa_overrides = pa_mask.sum() * 5
    n_pm_overrides = pm_mask.sum()
    total_overrides = n_pa_overrides + n_pm_overrides
    total_predictions = predictions.size

    print(f"  P. aeruginosa: {pa_mask.sum()} samples × 5 antibiotics = {n_pa_overrides} overrides")
    print(f"  P. mirabilis:  {pm_mask.sum()} samples × 1 antibiotic  = {n_pm_overrides} overrides")
    print(f"  Total: {total_overrides}/{total_predictions} predictions ({100*total_overrides/total_predictions:.1f}%)")

    return predictions


def predict_lightgbm_baseline(
    data_dir: str = "/home/erfan/data/ml_kaggle/raw",
    model_dir: str = "outputs/models",
    output_dir: str = "outputs/submissions",
    model_type: str = "lgb",
    version_desc: str = "reweight_intrinsic",
):
    """
    Generate predictions using trained LightGBM models.

    Args:
        data_dir: Directory containing test.csv
        model_dir: Directory containing trained models
        output_dir: Directory to save submission.csv
        model_type: Model type abbreviation for filename
        version_desc: Description suffix for filename
    """
    print("=" * 60)
    print("LightGBM Baseline Inference")
    print("=" * 60)

    # Load test data
    print("\n[Step 1] Loading test data...")
    test_df = pd.read_csv(f"{data_dir}/test.csv")

    feature_cols = [f"maldi_feature_{i}" for i in range(6000)]
    X_test_full = test_df[feature_cols].values
    sample_ids = test_df["sample_id"].values

    print(f"Test samples: {len(sample_ids)}")

    # Load training data to identify constant features
    print("\n[Step 2] Identifying constant features...")
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    X_train_full = train_df[feature_cols].values

    # Remove constant features (same as training)
    variances = X_train_full.var(axis=0)
    constant_mask = variances > 1e-5
    X_test = X_test_full[:, constant_mask]

    print(f"Removed {(~constant_mask).sum()} constant features")
    print(f"Remaining features: {constant_mask.sum()}")

    # Load models and predict
    print("\n[Step 3] Loading models and generating predictions...")
    print("-" * 60)

    predictions = np.zeros((len(sample_ids), len(ANTIBIOTICS)))

    for idx, antibiotic in enumerate(ANTIBIOTICS):
        model_path = f"{model_dir}/lgb_baseline_{antibiotic}.txt"

        if not Path(model_path).exists():
            print(f"WARNING: Model not found for {antibiotic}: {model_path}")
            continue

        # Load model
        model = lgb.Booster(model_file=model_path)

        # Predict
        pred_proba = model.predict(X_test)
        predictions[:, idx] = pred_proba

        print(f"  [{idx+1}/8] {antibiotic}")

    # Apply intrinsic resistance rules
    print("\n[Step 3.5] Applying intrinsic resistance rules...")
    species_ids = test_df["species_id"].values
    predictions = apply_intrinsic_resistance_rules(predictions, species_ids)

    # Create submission dataframe
    print("\n[Step 4] Creating submission file...")

    submission = pd.DataFrame({
        "sample_id": sample_ids,
        **{antibiotic: predictions[:, idx] for idx, antibiotic in enumerate(ANTIBIOTICS)}
    })

    # Generate unique filename using naming convention
    filename = generate_submission_name(model_type, version_desc)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    submission_path = output_path / filename
    submission.to_csv(submission_path, index=False)

    print(f"Saved submission to {submission_path}")

    # Validate submission format
    print("\n[Step 5] Validating submission format...")
    sample_submission = pd.read_csv(f"{data_dir}/sample_submission.csv")

    if submission.shape != sample_submission.shape:
        print(f"WARNING: Shape mismatch!")
        print(f"  Expected: {sample_submission.shape}")
        print(f"  Got: {submission.shape}")

    if list(submission.columns) != list(sample_submission.columns):
        print(f"WARNING: Column mismatch!")
        print(f"  Expected: {list(sample_submission.columns)}")
        print(f"  Got: {list(submission.columns)}")

    # Check prediction ranges
    print("\nPrediction statistics:")
    for antibiotic in ANTIBIOTICS:
        preds = submission[antibiotic].values
        print(f"  {antibiotic:30} min={preds.min():.4f}, max={preds.max():.4f}, mean={preds.mean():.4f}")

    # Log submission to tracker
    print("\n[Step 6] Logging submission to tracker...")

    # Load OOF metrics from training (for now, use placeholder)
    # TODO: Pass OOF metrics from training or load from saved file
    metrics = {
        "mean_auc": 0.8978,  # From training, should be passed in
        "per_antibiotic": {
            "Ampicillin": 0.9282,
            "Levofloxacin": 0.8461,
            "Ciprofloxacin": 0.8454,
            "Imipenem": 0.9911,
            "Amoxicillin_Clavulanic_acid": 0.7078,
            "Ertapenem": 0.9862,
            "Cefotaxime": 0.9349,
            "Cefuroxime": 0.9426
        }
    }

    config = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "sample_weight": "0.3x for P.aeruginosa",
        "intrinsic_rules": True
    }

    notes = "LightGBM with sample reweighting (0.3x P.aeruginosa) + intrinsic resistance rules"

    sub_id = log_submission(
        filename=filename,
        model_type=model_type,
        version_desc=version_desc,
        metrics=metrics,
        config=config,
        notes=notes
    )

    print(f"Logged as Submission #{sub_id}")

    print("\n" + "=" * 60)
    print("Inference complete!")
    print("=" * 60)

    # Print submission command
    print("\nTo submit to Kaggle, run:")
    print(f"  kaggle competitions submit -c antimicrobial-resistance-prediction-from-maldi-tof \\")
    print(f"    -f {submission_path} \\")
    print(f"    -m \"LightGBM baseline (submission #{sub_id})\"")

    print(f"\nAfter submission, update leaderboard score:")
    print(f"  uv run python scripts/update_leaderboard.py --sub-id {sub_id} --public-lb <SCORE>")

    return submission


if __name__ == "__main__":
    submission = predict_lightgbm_baseline()
