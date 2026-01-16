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


# =============================================================================
# VALIDATION SPLIT (Phase 5.5)
# =============================================================================

# Test species distribution (from EDA)
TEST_SPECIES_DISTRIBUTION = {
    0: 0.269,  # E.coli
    1: 0.508,  # K.pneumoniae (MAJORITY)
    2: 0.193,  # P.mirabilis
    3: 0.030   # P.aeruginosa
}

SPECIES_NAMES = {0: "E.coli", 1: "K.pneumoniae", 2: "P.mirabilis", 3: "P.aeruginosa"}

# Default path for saving validation split
DEFAULT_VAL_SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "val_split.npz"


def create_test_distribution_split(
    X: np.ndarray,
    y: np.ndarray,
    species: np.ndarray,
    val_size: float = 0.2,
    random_state: int = 42,
    save_path: Optional[Path] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create train/val split where val set matches test species distribution.

    This is critical for Phase 5.5 because the train set has a very different
    species distribution than test (43% P.aeruginosa in train vs 3% in test).
    Simple stratified split would preserve train distribution, not match test.

    CORRECT formula: val_samples[species] = total_val_size * target_ratio[species]
    NOT: n_species * val_size * target_ratio (dimensionally wrong!)

    Args:
        X: Features (n_samples, n_features)
        y: Targets (n_samples, n_targets)
        species: Species IDs (n_samples,)
        val_size: Fraction of data to use for validation (default: 0.2)
        random_state: Random seed for reproducibility
        save_path: Path to save the split (default: data/processed/val_split.npz)

    Returns:
        X_train, X_val, y_train, y_val, species_train, species_val

    Example:
        >>> X_train, X_val, y_train, y_val, species_train, species_val = \\
        >>>     create_test_distribution_split(X, y, species, val_size=0.2)
        >>> # Val distribution should match test:
        >>> # E.coli: 26.9%, K.pneumoniae: 50.8%, P.mirabilis: 19.3%, P.aeruginosa: 3.0%
    """
    rng = np.random.default_rng(random_state)
    n_total = len(X)
    total_val_size = int(n_total * val_size)

    print(f"\n[create_test_distribution_split]")
    print(f"  Total samples: {n_total}")
    print(f"  Target val size: {total_val_size} ({val_size*100:.1f}%)")

    # Group samples by species
    species_groups = {}
    for species_id in range(4):
        mask = (species == species_id)
        species_groups[species_id] = {
            'indices': np.where(mask)[0],
            'count': mask.sum()
        }

    # Calculate target val samples for each species
    val_samples_per_species = {}
    for species_id, target_pct in TEST_SPECIES_DISTRIBUTION.items():
        target_count = int(total_val_size * target_pct)
        val_samples_per_species[species_id] = target_count

    # Handle rounding: distribute remainder to species with most samples
    current_total = sum(val_samples_per_species.values())
    remainder = total_val_size - current_total

    if remainder != 0:
        # Sort species by available count (descending)
        species_by_count = sorted(
            val_samples_per_species.keys(),
            key=lambda s: species_groups[s]['count'],
            reverse=True
        )
        for i in range(remainder):
            species_id = species_by_count[i % len(species_by_count)]
            val_samples_per_species[species_id] += 1

    print(f"\n  Target val distribution:")
    for species_id, target_count in val_samples_per_species.items():
        available = species_groups[species_id]['count']
        pct = target_count / total_val_size
        print(f"    {SPECIES_NAMES[species_id]:15} {target_count:4d} ({pct*100:5.1f}%) - available: {available}")

    # Sample from each species group
    val_indices = []
    train_indices = []

    for species_id in range(4):
        group_indices = species_groups[species_id]['indices']
        n_val = val_samples_per_species[species_id]

        # Check if we have enough samples
        if n_val > len(group_indices):
            import warnings
            warnings.warn(
                f"Species {SPECIES_NAMES[species_id]} has only {len(group_indices)} "
                f"samples but need {n_val} for validation. Using all available."
            )
            n_val = len(group_indices)

        # Shuffle and split
        rng.shuffle(group_indices)
        val_indices.extend(group_indices[:n_val])
        train_indices.extend(group_indices[n_val:])

    # Convert to numpy arrays and shuffle (remove species clustering)
    val_indices = np.array(val_indices)
    train_indices = np.array(train_indices)

    rng.shuffle(val_indices)
    rng.shuffle(train_indices)

    # Split data
    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]
    species_train, species_val = species[train_indices], species[val_indices]

    # Verify the split
    print(f"\n  Actual val distribution:")
    val_species_dist = (pd.Series(species_val).value_counts() / len(species_val)).sort_index()
    for species_id in range(4):
        actual_pct = val_species_dist.get(species_id, 0)
        target_pct = TEST_SPECIES_DISTRIBUTION[species_id]
        diff = actual_pct - target_pct
        status = "✓" if abs(diff) < 0.01 else "⚠"
        print(f"    {SPECIES_NAMES[species_id]:15} {actual_pct*100:5.1f}% "
              f"(target: {target_pct*100:5.1f}%, diff: {diff*100:+5.1f}%) {status}")

    # Verify no overlap
    assert len(set(train_indices) & set(val_indices)) == 0, "Data leakage! Train and val overlap."
    print(f"\n  ✓ No overlap between train and val")

    # Verify total samples
    assert len(X_train) + len(X_val) == n_total, "Sample count mismatch!"
    print(f"  ✓ Train: {len(X_train)}, Val: {len(X_val)}, Total: {len(X_train) + len(X_val)}")

    # Save to disk if path provided
    if save_path is None:
        save_path = DEFAULT_VAL_SPLIT_PATH

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        species_train=species_train,
        species_val=species_val,
        train_indices=train_indices,
        val_indices=val_indices,
        random_state=random_state,
        val_size=val_size
    )
    print(f"\n  ✓ Validation split saved to: {save_path}")

    return X_train, X_val, y_train, y_val, species_train, species_val


def load_validation_split(
    split_path: Optional[Path] = None,
    recreate: bool = False,
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    species: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the validation split from disk.

    If the split doesn't exist and recreate=True, it will be created.
    If recreate=False and split doesn't exist, raises FileNotFoundError.

    Args:
        split_path: Path to the saved split (default: data/processed/val_split.npz)
        recreate: If True and split doesn't exist, create it from provided data
        X, y, species: Data for recreating the split (required if recreate=True)

    Returns:
        X_train, X_val, y_train, y_val, species_train, species_val

    Raises:
        FileNotFoundError: If split doesn't exist and recreate=False
        ValueError: If recreate=True but X, y, species not provided
    """
    if split_path is None:
        split_path = DEFAULT_VAL_SPLIT_PATH

    split_path = Path(split_path)

    if not split_path.exists():
        if recreate:
            if X is None or y is None or species is None:
                raise ValueError(
                    "Must provide X, y, species when recreate=True and split doesn't exist"
                )
            print(f"[load_validation_split] Split not found, creating new split...")
            return create_test_distribution_split(X, y, species, save_path=split_path)
        else:
            raise FileNotFoundError(
                f"Validation split not found at {split_path}. "
                f"Set recreate=True to create it, or run create_test_distribution_split() first."
            )

    # Load from disk
    data = np.load(split_path)
    X_train = data['X_train']
    X_val = data['X_val']
    y_train = data['y_train']
    y_val = data['y_val']
    species_train = data['species_train']
    species_val = data['species_val']

    print(f"[load_validation_split] Loaded from: {split_path}")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")

    return X_train, X_val, y_train, y_val, species_train, species_val
