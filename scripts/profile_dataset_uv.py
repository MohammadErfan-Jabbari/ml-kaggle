from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path("raw")
TRAIN_PATH = RAW / "train.csv"
TEST_PATH = RAW / "test.csv"

ANTIBIOTICS = [
    "Ampicillin",
    "Levofloxacin",
    "Ciprofloxacin",
    "Imipenem",
    "Amoxicillin_Clavulanic_acid",
    "Ertapenem",
    "Cefotaxime",
    "Cefuroxime",
]


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    feat_cols = [c for c in train.columns if c.startswith("maldi_feature_")]

    print("train shape:", train.shape)
    print("test shape:", test.shape)
    print("n_features:", len(feat_cols))

    print("\nSpecies distribution (train):")
    print(train["species_id"].value_counts(normalize=True).sort_index().to_string())

    print("\nSpecies distribution (test):")
    print(test["species_id"].value_counts(normalize=True).sort_index().to_string())

    print("\nLabel missingness and prevalence (train):")
    for c in ANTIBIOTICS:
        labeled = int(train[c].notna().sum())
        missing = int(train[c].isna().sum())
        pos_rate = float(train[c].dropna().mean()) if labeled else float("nan")
        print(f"{c:28s} labeled={labeled:4d} missing={missing:4d} pos_rate={pos_rate:.3f}")

    labels_per_sample = train[ANTIBIOTICS].notna().sum(axis=1)
    print("\nLabels-per-sample counts:")
    print(labels_per_sample.value_counts().sort_index().to_string())
    fully = int((labels_per_sample == len(ANTIBIOTICS)).sum())
    partial = int((labels_per_sample < len(ANTIBIOTICS)).sum())
    print(f"fully_labeled={fully} partial_labeled={partial}")

    full_mask = labels_per_sample == len(ANTIBIOTICS)
    y_full = train.loc[full_mask, ANTIBIOTICS].astype(float)
    print("\nInter-antibiotic correlation (Pearson on binary, fully-labeled subset):")
    print(y_full.corr().round(2).to_string())

    intrinsic = {
        3: [
            "Ampicillin",
            "Amoxicillin_Clavulanic_acid",
            "Ertapenem",
            "Cefotaxime",
            "Cefuroxime",
        ],
        2: ["Imipenem"],
    }

    print("\nIntrinsic rule coverage (test rows affected):")
    for sp, cols in intrinsic.items():
        n = int((test["species_id"] == sp).sum())
        print(f"species_id={sp} n_test={n} forced_1_for={cols}")


if __name__ == "__main__":
    main()
