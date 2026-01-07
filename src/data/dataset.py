"""Data loading and preprocessing for MALDI-TOF AMR prediction."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DIR = PROJECT_ROOT / "raw"

ANTIBIOTICS = [
    "Ampicillin",
    "Levofloxacin",
    "Ciprofloxacin",
    "Imipenem",
    "Amoxicillin_Clavulanic_acid",
    "Ertapenem",
    "Cefotaxime",
    "Cefuroxime"
]


def load_species_mapping() -> dict:
    """Load species_id to species_name mapping."""
    df = pd.read_csv(RAW_DIR / "species_mapping.csv")
    return dict(zip(df["species_id"], df["species_name"]))


def load_train_data() -> pd.DataFrame:
    """Load training data."""
    return pd.read_csv(RAW_DIR / "train.csv")


def load_test_data() -> pd.DataFrame:
    """Load test data."""
    return pd.read_csv(RAW_DIR / "test.csv")


def load_sample_submission() -> pd.DataFrame:
    """Load sample submission format."""
    return pd.read_csv(RAW_DIR / "sample_submission.csv")


def split_features_targets(df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Split dataframe into MALDI features, target labels, and metadata.

    Args:
        df: DataFrame with sample_id, species_id, maldi_features, and optional antibiotic labels

    Returns:
        features: (n_samples, 6000) MALDI feature array
        targets: (n_samples, 8) antibiotic labels, or None if test data
        metadata: (n_samples, 2) array with [sample_id, species_id]
    """
    # Extract MALDI features (columns 2 to 6001)
    maldi_cols = [c for c in df.columns if c.startswith("maldi_feature_")]
    features = df[maldi_cols].values.astype(np.float32)

    # Metadata
    metadata = df[["sample_id", "species_id"]].values

    # Targets (if present)
    if all(antibiotic in df.columns for antibiotic in ANTIBIOTICS):
        targets = df[ANTIBIOTICS].values
        # Keep NaN values for semi-supervised handling
        targets = targets.astype(np.float32)
    else:
        targets = None

    return features, targets, metadata


class MaldiDataset(Dataset):
    """PyTorch Dataset for MALDI-TOF data."""

    def __init__(self, features: np.ndarray, targets: Optional[np.ndarray],
                 species_id: np.ndarray, use_species: bool = True):
        """
        Args:
            features: (n_samples, 6000) MALDI features
            targets: (n_samples, 8) antibiotic labels (may contain NaN)
            species_id: (n_samples,) species identifiers
            use_species: Whether to include species_id as a feature
        """
        self.features = torch.from_numpy(features)
        self.species_id = torch.from_numpy(species_id).long()
        self.use_species = use_species

        if targets is not None:
            self.targets = torch.from_numpy(targets)
        else:
            self.targets = None

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "features": self.features[idx],
            "species_id": self.species_id[idx],
        }

        if self.targets is not None:
            item["targets"] = self.targets[idx]

        return item


def get_dataloaders(batch_size: int = 32, num_workers: int = 4,
                    use_species: bool = True) -> Tuple:
    """
    Create train and validation dataloaders.

    Note: You'll want to implement proper train/val split based on
    labeled vs unlabeled samples for semi-supervised learning.
    """
    train_df = load_train_data()
    features, targets, metadata = split_features_targets(train_df)

    # Simple split for now - you may want stratified split by species
    n_train = int(0.8 * len(features))

    train_dataset = MaldiDataset(
        features[:n_train], targets[:n_train],
        metadata[:n_train, 1], use_species
    )

    val_dataset = MaldiDataset(
        features[n_train:], targets[n_train:],
        metadata[n_train:, 1], use_species
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader
