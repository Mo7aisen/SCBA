"""
Loss functions for segmentation tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice loss for segmentation.

    Dice coefficient: 2 * |X ∩ Y| / (|X| + |Y|)
    """

    def __init__(self, smooth=1.0, ignore_index=-100):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) logits
            target: (B, H, W) or (B, 1, H, W) ground truth
        """
        # Convert to probabilities
        pred = F.softmax(pred, dim=1)

        # Handle target shape
        if target.dim() == 4:
            target = target.squeeze(1)

        # One-hot encode target
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target.long(), num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        # Flatten
        pred = pred.view(pred.size(0), pred.size(1), -1)
        target_one_hot = target_one_hot.view(target_one_hot.size(0), target_one_hot.size(1), -1)

        # Compute Dice
        intersection = (pred * target_one_hot).sum(dim=2)
        union = pred.sum(dim=2) + target_one_hot.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Average over classes (skip background if needed)
        return 1.0 - dice[:, 1:].mean()


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss."""

    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss(smooth=smooth)
        self.bce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        dice_loss = self.dice(pred, target)
        bce_loss = self.bce(pred, target.long())
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) logits
            target: (B, H, W) ground truth
        """
        ce_loss = F.cross_entropy(pred, target.long(), reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky loss - generalization of Dice loss.

    Salehi et al., "Tversky loss function for image segmentation", MICCAI 2017.
    """

    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)

        if target.dim() == 4:
            target = target.squeeze(1)

        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target.long(), num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        # Flatten
        pred = pred.view(pred.size(0), pred.size(1), -1)
        target_one_hot = target_one_hot.view(target_one_hot.size(0), target_one_hot.size(1), -1)

        # True positives, false positives, false negatives
        tp = (pred * target_one_hot).sum(dim=2)
        fp = (pred * (1 - target_one_hot)).sum(dim=2)
        fn = ((1 - pred) * target_one_hot).sum(dim=2)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        return 1.0 - tversky[:, 1:].mean()


def get_loss(loss_name="dice_bce", **kwargs):
    """
    Factory function for loss functions.

    Args:
        loss_name: One of 'dice', 'bce', 'dice_bce', 'focal', 'tversky'
        **kwargs: Additional arguments for the loss function
    """
    if loss_name == "dice":
        return DiceLoss(**kwargs)
    elif loss_name == "bce":
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_name == "dice_bce":
        return DiceBCELoss(**kwargs)
    elif loss_name == "focal":
        return FocalLoss(**kwargs)
    elif loss_name == "tversky":
        return TverskyLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")
