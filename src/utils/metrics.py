"""Evaluation metrics for multi-label AMR prediction."""

import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Dict, List


def mean_auc(y_true: np.ndarray, y_pred: np.ndarray, antibiotic_names: List[str]) -> Dict[str, float]:
    """
    Calculate mean AUC across all antibiotics.

    Args:
        y_true: Shape (n_samples, n_antibiotics) - True labels (0 or 1, may contain NaN)
        y_pred: Shape (n_samples, n_antibiotics) - Predicted probabilities
        antibiotic_names: List of antibiotic column names

    Returns:
        Dictionary with mean_auc and individual antibiotic AUCs
    """
    aucs = {}
    valid_mask = ~np.isnan(y_true)

    for i, name in enumerate(antibiotic_names):
        mask = valid_mask[:, i]
        if mask.sum() > 1:
            # Need at least 2 samples with both classes for AUC
            try:
                auc = roc_auc_score(y_true[mask, i], y_pred[mask, i])
                aucs[name] = auc
            except ValueError:
                aucs[name] = 0.0
        else:
            aucs[name] = 0.0

    mean_auc_value = np.mean(list(aucs.values()))
    aucs['mean_auc'] = mean_auc_value

    return aucs


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """Print metrics in a formatted way."""
    print(f"{prefix}Mean AUC: {metrics.get('mean_auc', 0):.4f}")
    for name, value in metrics.items():
        if name != 'mean_auc':
            print(f"{prefix}  {name}: {value:.4f}")
