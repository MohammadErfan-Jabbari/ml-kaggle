"""Inference script for generating Kaggle submissions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data.dataset import load_test_data, split_features_targets, MaldiDataset, ANTIBIOTICS, load_sample_submission
from models.baseline import create_model


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def predict(model: nn.Module, dataloader, device: str) -> np.ndarray:
    """Generate predictions for test set."""
    model.eval()

    all_probs = []
    all_sample_ids = []

    for batch in dataloader:
        features = batch["features"].to(device)
        species_id = batch["species_id"].to(device)

        logits = model(features, species_id)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.append(probs)

    return np.vstack(all_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", default="outputs/models/best.pt", help="Path to model checkpoint")
    parser.add_argument("--output", default="outputs/submissions/submission.csv", help="Output submission path")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Device
    device = torch.device(config["experiment"]["device"]
                          if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test data
    print("Loading test data...")
    test_df = load_test_data()
    sample_sub = load_sample_submission()

    features, _, metadata = split_features_targets(test_df)
    test_dataset = MaldiDataset(
        features,
        targets=None,
        species_id=metadata[:, 1],
        use_species=config["model"]["use_species"],
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
    )

    print(f"Test samples: {len(test_dataset)}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = create_model(config["model"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # Generate predictions
    print("Generating predictions...")
    predictions = predict(model, test_loader, device)

    # Create submission DataFrame
    submission = pd.DataFrame({
        "sample_id": sample_sub["sample_id"].values,
        **{antibiotic: predictions[:, i] for i, antibiotic in enumerate(ANTIBIOTICS)}
    })

    # Save submission
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Submission saved to {args.output}")
    print(f"Submission shape: {submission.shape}")
    print(f"\nSample predictions:")
    print(submission.head())


if __name__ == "__main__":
    main()
