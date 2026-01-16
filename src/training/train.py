"""Training script for AMR prediction."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset import load_train_data, split_features_targets, MaldiDataset, ANTIBIOTICS
from models.baseline import create_model
from utils.metrics import mean_auc, print_metrics
from utils.loss import MaskedBCEWithLogitsLoss


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer,
                criterion: nn.Module, device: str) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        features = batch["features"].to(device)
        species_id = batch["species_id"].to(device)
        targets = batch["targets"].to(device)

        # Forward pass
        logits = model(features, species_id)
        loss = criterion(logits, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model: nn.Module, dataloader: DataLoader, device: str) -> dict:
    """Validate and compute metrics."""
    model.eval()

    all_logits = []
    all_targets = []

    for batch in dataloader:
        features = batch["features"].to(device)
        species_id = batch["species_id"].to(device)
        targets = batch["targets"]

        logits = model(features, species_id)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_logits.append(probs)
        all_targets.append(targets.numpy())

    y_pred = np.vstack(all_logits)
    y_true = np.vstack(all_targets)

    metrics = mean_auc(y_true, y_pred, ANTIBIOTICS)
    return metrics


def main():
    # Load config
    config = load_config("configs/baseline.yaml")

    # Set seed
    torch.manual_seed(config["experiment"]["seed"])

    # Device
    device = torch.device(config["experiment"]["device"]
                          if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    train_df = load_train_data()
    features, targets, metadata = split_features_targets(train_df)

    # Simple train/val split (you'll want to improve this)
    n_val = int(0.2 * len(features))
    val_features, train_features = features[:n_val], features[n_val:]
    val_targets, train_targets = targets[:n_val], targets[n_val:]
    val_species, train_species = metadata[:n_val, 1], metadata[n_val:, 1]

    train_dataset = MaldiDataset(train_features, train_targets, train_species,
                                  use_species=config["model"]["use_species"])
    val_dataset = MaldiDataset(val_features, val_targets, val_species,
                                use_species=config["model"]["use_species"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create model
    model = create_model(config["model"]).to(device)

    # Loss and optimizer (handle NaN targets with masked loss)
    criterion = MaskedBCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Training loop
    best_auc = 0.0
    patience_counter = 0

    for epoch in range(config["training"]["epochs"]):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_metrics = validate(model, val_loader, device)
        val_auc = val_metrics["mean_auc"]

        print(f"Epoch {epoch+1}/{config['training']['epochs']} - "
              f"Loss: {train_loss:.4f} - Val AUC: {val_auc:.4f}")

        # Early stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), "outputs/models/best.pt")
            print(f"  New best AUC: {best_auc:.4f} - saved checkpoint")
        else:
            patience_counter += 1
            if patience_counter >= config["training"]["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nTraining complete. Best Val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
