"""Baseline model architectures for AMR prediction."""

import torch
import torch.nn as nn
from typing import Optional


class MLPBaseline(nn.Module):
    """Simple MLP baseline for multi-label AMR prediction."""

    def __init__(
        self,
        input_dim: int = 6000,
        hidden_dims: list = [512, 256, 128],
        output_dim: int = 8,
        dropout: float = 0.3,
        num_species: int = 4,
        use_species: bool = True,
    ):
        """
        Args:
            input_dim: Number of MALDI features (6000)
            hidden_dims: List of hidden layer sizes
            output_dim: Number of antibiotics (8)
            dropout: Dropout probability
            num_species: Number of bacterial species (4)
            use_species: Whether to include species embedding
        """
        super().__init__()

        self.use_species = use_species
        self.num_species = num_species

        # Species embedding
        if use_species:
            self.species_embedding = nn.Embedding(num_species, 32)
            input_dim += 32

        # Build MLP layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor, species_id: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch_size, 6000) MALDI features
            species_id: (batch_size,) species identifiers

        Returns:
            logits: (batch_size, 8) - one logit per antibiotic
        """
        x = features

        # Add species embedding
        if self.use_species:
            species_emb = self.species_embedding(species_id)  # (B, 32)
            x = torch.cat([x, species_emb], dim=1)

        logits = self.mlp(x)
        return logits


def create_model(config: dict) -> nn.Module:
    """Factory function to create model from config."""
    model_type = config.get("type", "mlp")

    if model_type == "mlp":
        return MLPBaseline(
            input_dim=config["input_dim"],
            hidden_dims=config.get("hidden_dims", [512, 256, 128]),
            output_dim=config["output_dim"],
            dropout=config.get("dropout", 0.3),
            num_species=config.get("num_species", 4),
            use_species=config.get("use_species", True),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
