import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedBCEWithLogitsLoss(nn.Module):
    """
    Binary Cross Entropy with Logits Loss that ignores NaN target values.

    This is critical for the AMR dataset where labels are frequently missing
    (e.g., Amox/Clav is missing in 42.8% of samples).
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Tensor of shape (batch_size, n_classes) - raw logits
            targets: Tensor of shape (batch_size, n_classes) - labels (0/1/NaN)

        Returns:
            loss: Scalar or tensor depending on reduction
        """
        # Create mask for valid targets (not NaN)
        mask = ~torch.isnan(targets)

        # Filter predictions and targets
        # We need to flatten to apply the mask efficiently
        preds_flat = predictions[mask]
        targets_flat = targets[mask].float()

        if len(targets_flat) == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # Compute loss only on valid samples
        loss = F.binary_cross_entropy_with_logits(
            preds_flat,
            targets_flat,
            reduction=self.reduction
        )

        return loss
