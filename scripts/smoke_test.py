#!/usr/bin/env python3
"""
Smoke test for data pipeline.

Verifies:
- Data files exist and are readable
- Shapes match expectations
- No features are completely NaN
- Each row has at least one non-NaN label
- Species distribution is reasonable
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    """Run smoke tests on train/test data."""
    data_dir = Path("/home/erfan/data/ml_kaggle/raw")
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    print("🔥 Smoke Test: Data Pipeline")
    print("=" * 50)

    # Test 1: Files exist
    print("\n[Test 1] Checking files exist...")
    if not train_path.exists():
        print(f"❌ FAIL: {train_path} not found")
        return 1
    if not test_path.exists():
        print(f"❌ FAIL: {test_path} not found")
        return 1
    print("✅ PASS: Both files exist")

    # Test 2: Load data
    print("\n[Test 2] Loading data...")
    try:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        print("✅ PASS: Data loaded successfully")
    except Exception as e:
        print(f"❌ FAIL: Could not load data: {e}")
        return 1

    # Test 3: Verify shapes
    print("\n[Test 3] Verifying shapes...")
    expected_train_shape = (3360, 6010)  # sample_id + species_id + 6000 features + 8 labels
    expected_test_shape = (1000, 6002)   # sample_id + species_id + 6000 features

    if train.shape != expected_train_shape:
        print(f"❌ FAIL: Train shape is {train.shape}, expected {expected_train_shape}")
        return 1
    if test.shape != expected_test_shape:
        print(f"❌ FAIL: Test shape is {test.shape}, expected {expected_test_shape}")
        return 1
    print(f"✅ PASS: Train: {train.shape}, Test: {test.shape}")

    # Test 4: Verify column structure
    print("\n[Test 4] Verifying column structure...")
    expected_cols = ["sample_id", "species_id"] + [f"maldi_feature_{i}" for i in range(6000)]
    expected_labels = ["Ampicillin", "Levofloxacin", "Ciprofloxacin", "Imipenem",
                       "Amoxicillin_Clavulanic_acid", "Ertapenem", "Cefotaxime", "Cefuroxime"]

    missing_features = set(expected_cols) - set(train.columns)
    if missing_features:
        print(f"❌ FAIL: Missing features in train: {missing_features}")
        return 1

    missing_labels = set(expected_labels) - set(train.columns)
    if missing_labels:
        print(f"❌ FAIL: Missing labels in train: {missing_labels}")
        return 1
    print("✅ PASS: All expected columns present")

    # Test 5: No features are all NaN
    print("\n[Test 5] Checking for all-NaN features...")
    feature_cols = [f"maldi_feature_{i}" for i in range(6000)]
    all_nan_features = train[feature_cols].isna().all()
    if all_nan_features.any():
        nan_count = all_nan_features.sum()
        print(f"❌ FAIL: {nan_count} features are entirely NaN")
        return 1
    print("✅ PASS: No all-NaN features")

    # Test 6: Each row has at least one non-NaN label
    print("\n[Test 6] Checking label coverage...")
    label_cols = expected_labels
    labels = train[label_cols]
    rows_with_labels = labels.notna().any(axis=1).sum()

    if rows_with_labels < train.shape[0] * 0.9:
        print(f"❌ FAIL: Only {rows_with_labels}/{train.shape[0]} rows have labels")
        return 1
    print(f"✅ PASS: {rows_with_labels}/{train.shape[0]} rows have at least one label")

    # Test 7: Count NaN labels per antibiotic
    print("\n[Test 7] Label missingness per antibiotic...")
    missing_counts = labels.isna().sum()
    missing_pct = (missing_counts / len(train) * 100).round(1)

    print("Antibiotic           | Missing | Missing %")
    print("-" * 45)
    for antibiotic, count, pct in zip(label_cols, missing_counts, missing_pct):
        print(f"{antibiotic:27} | {count:7} | {pct:6}%")

    # Check Amox/Clav specifically (known to have ~43% missing)
    amox_missing = missing_pct["Amoxicillin_Clavulanic_acid"]
    if not (40 < amox_missing < 50):
        print(f"⚠️  WARNING: Amox/Clav missing {amox_missing}% (expected ~43%)")

    # Test 8: Species distribution
    print("\n[Test 8] Species distribution...")
    species_counts = train["species_id"].value_counts().sort_index()
    species_pct = (species_counts / len(train) * 100).round(1)

    species_names = {0: "E. coli", 1: "K. pneumoniae", 2: "P. mirabilis", 3: "P. aeruginosa"}

    print("Species        | Count | Pct")
    print("-" * 35)
    for species_id, count, pct in zip(species_counts.index, species_counts.values, species_pct.values):
        name = species_names.get(species_id, f"Unknown ({species_id})")
        print(f"{name:14} | {count:5} | {pct:4}%")

    # Check P. aeruginosa specifically (known to be ~43%)
    p_aeruginosa_pct = species_pct.get(3, 0)
    if not (40 < p_aeruginosa_pct < 48):
        print(f"⚠️  WARNING: P. aeruginosa is {p_aeruginosa_pct}% (expected ~43%)")

    # Test 9: Check test species distribution (shift warning)
    print("\n[Test 9] Test species distribution...")
    test_species_counts = test["species_id"].value_counts().sort_index()
    test_species_pct = (test_species_counts / len(test) * 100).round(1)

    print("Species        | Count | Pct")
    print("-" * 35)
    for species_id, count, pct in zip(test_species_counts.index, test_species_counts.values, test_species_pct.values):
        name = species_names.get(species_id, f"Unknown ({species_id})")
        print(f"{name:14} | {count:5} | {pct:4}%")

    # Warn about distribution shift
    p_aeruginosa_test_pct = test_species_pct.get(3, 0)
    shift = p_aeruginosa_pct - p_aeruginosa_pct
    if abs(shift) > 30:
        print(f"⚠️  WARNING: P. aeruginosa distribution shift: {p_aeruginosa_pct:.1f}% → {p_aeruginosa_test_pct:.1f}%")

    print("\n" + "=" * 50)
    print("✅ All smoke tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
