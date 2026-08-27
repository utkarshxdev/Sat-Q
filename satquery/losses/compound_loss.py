"""
satquery/losses/compound_loss.py
──────────────────────────────────
Combined FocalLoss + DiceLoss for binary change detection.

Change pixels account for < 5% of total image pixels (severe class imbalance).
FocalLoss down-weights easy negatives (unchanged pixels) so the model focuses
on the rare changed regions. DiceLoss directly optimises the overlap metric,
which is robust to class imbalance.

Combined:
    L = α_focal * FocalLoss + α_dice * DiceLoss

Default: equal weighting (0.5 / 0.5).

Also provides BCEWithLogitsLoss wrapper for the fusion multi-label classifier.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary Focal Loss for change detection.

    L_focal = -α_t * (1 - p_t)^γ * log(p_t)

    where:
        p_t = σ(logits) for positive class, 1 - σ(logits) for negative.
        α_t = alpha for positive class, (1 - alpha) for negative.
        γ   = focusing parameter (default 2).

    Args:
        gamma: Focusing exponent. Higher → more focus on hard examples.
        alpha: Balance factor for positive class (change=1). 0.75 upweights
               change pixels relative to no-change pixels.
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.75,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, 1, H, W) — raw model output (before sigmoid).
            targets : (B, 1, H, W) — binary ground truth {0, 1} float.

        Returns:
            Scalar loss value.
        """
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )  # (B, 1, H, W)

        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.

    L_dice = 1 - (2 * ∑(p * g) + ε) / (∑p + ∑g + ε)

    Works on probabilities (after sigmoid) rather than logits.

    Args:
        smooth : Laplace smoothing to prevent division by zero.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, 1, H, W) raw output.
            targets : (B, 1, H, W) binary {0, 1} float.

        Returns:
            Scalar Dice loss in [0, 1].
        """
        probs = torch.sigmoid(logits)
        # Flatten spatial dims for computation
        probs_flat   = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_score.mean()


class CompoundChangeLoss(nn.Module):
    """
    FocalLoss + DiceLoss combined for change detection training.

    L = focal_weight * FocalLoss + dice_weight * DiceLoss

    Args:
        focal_weight : Weight for focal loss component (default 0.5).
        dice_weight  : Weight for dice loss component (default 0.5).
        focal_gamma  : Focal loss gamma parameter.
        focal_alpha  : Focal loss alpha (minority class weight).
    """

    def __init__(
        self,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.75,
    ) -> None:
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight  = dice_weight
        self.focal_loss   = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.dice_loss    = DiceLoss()

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            logits  : (B, 1, H, W)
            targets : (B, 1, H, W)  float {0, 1}

        Returns:
            total_loss : Scalar combined loss.
            components : dict with individual loss values for logging.
        """
        fl = self.focal_loss(logits, targets)
        dl = self.dice_loss(logits, targets)
        total = self.focal_weight * fl + self.dice_weight * dl
        return total, {"focal": fl.item(), "dice": dl.item(), "total": total.item()}
