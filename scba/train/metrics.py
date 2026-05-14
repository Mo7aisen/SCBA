"""
Evaluation metrics for segmentation tasks.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix


def dice_coefficient(pred, target, smooth=1.0):
    """
    Compute Dice coefficient.

    Args:
        pred: (B, C, H, W) predictions (logits or probabilities)
        target: (B, H, W) ground truth
        smooth: Smoothing factor
    """
    if pred.dim() == 4 and pred.shape[1] > 1:
        pred = torch.argmax(pred, dim=1)
    elif pred.dim() == 4:
        pred = pred.squeeze(1)

    if target.dim() == 4:
        target = target.squeeze(1)

    pred = pred.float()
    target = target.float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    return (2.0 * intersection + smooth) / (union + smooth)


def iou_score(pred, target, smooth=1e-6):
    """
    Compute Intersection over Union (IoU).

    Args:
        pred: (B, C, H, W) predictions
        target: (B, H, W) ground truth
        smooth: Smoothing factor
    """
    if pred.dim() == 4 and pred.shape[1] > 1:
        pred = torch.argmax(pred, dim=1)
    elif pred.dim() == 4:
        pred = pred.squeeze(1)

    if target.dim() == 4:
        target = target.squeeze(1)

    pred = pred.float()
    target = target.float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    return (intersection + smooth) / (union + smooth)


def boundary_f_score(pred, target, theta=2.0):
    """
    Compute Boundary F-score (BF-score).

    Csurka et al., "What is a good evaluation measure for semantic segmentation?", BMVC 2013.

    Args:
        pred: (H, W) binary prediction
        target: (H, W) binary ground truth
        theta: Distance threshold in pixels
    """
    from scipy.ndimage import distance_transform_edt

    # Convert to numpy if needed
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    # Ensure binary
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)

    # Extract contours
    pred_contour = np.logical_xor(pred, np.roll(pred, 1, axis=0)) | np.logical_xor(
        pred, np.roll(pred, 1, axis=1)
    )
    target_contour = np.logical_xor(target, np.roll(target, 1, axis=0)) | np.logical_xor(
        target, np.roll(target, 1, axis=1)
    )

    if not pred_contour.any() or not target_contour.any():
        return 0.0

    # Distance transforms
    dist_pred = distance_transform_edt(~pred_contour)
    dist_target = distance_transform_edt(~target_contour)

    # Precision and recall
    precision = (dist_target[pred_contour] <= theta).sum() / pred_contour.sum()
    recall = (dist_pred[target_contour] <= theta).sum() / target_contour.sum()

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def pixel_accuracy(pred, target):
    """Compute pixel-wise accuracy."""
    if pred.dim() == 4 and pred.shape[1] > 1:
        pred = torch.argmax(pred, dim=1)
    elif pred.dim() == 4:
        pred = pred.squeeze(1)

    if target.dim() == 4:
        target = target.squeeze(1)

    correct = (pred == target).sum().item()
    total = target.numel()

    return correct / total


def sensitivity_specificity(pred, target):
    """
    Compute sensitivity (recall) and specificity.

    Returns:
        dict with 'sensitivity' and 'specificity'
    """
    if pred.dim() == 4 and pred.shape[1] > 1:
        pred = torch.argmax(pred, dim=1)
    elif pred.dim() == 4:
        pred = pred.squeeze(1)

    if target.dim() == 4:
        target = target.squeeze(1)

    pred = pred.cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()

    tn, fp, fn, tp = confusion_matrix(target, pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {"sensitivity": sensitivity, "specificity": specificity}


class MetricTracker:
    """Track metrics during training."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {}
        self.counts = {}

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0

            self.metrics[key] += value
            self.counts[key] += 1

    def compute(self):
        return {key: val / self.counts[key] for key, val in self.metrics.items()}

    def __repr__(self):
        metrics = self.compute()
        return " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
