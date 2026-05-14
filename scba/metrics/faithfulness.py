"""
Faithfulness metrics for segmentation explanations.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


def _score_model(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    target_class: int = 1,
    target_mask: Optional[torch.Tensor] = None,
) -> float:
    """Return a scalar score for the target class."""
    output = model(image)
    if output.shape[1] > 1:
        probs = torch.softmax(output, dim=1)[:, target_class]
    else:
        probs = torch.sigmoid(output)[:, 0]

    if target_mask is not None:
        mask = target_mask.to(image.device)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        if mask.shape[-2:] != probs.shape[-2:]:
            mask = F.interpolate(mask.float(), size=probs.shape[-2:], mode="nearest")
        probs = probs * mask.squeeze(1)

    return float(probs.mean().item())


def deletion_insertion_auc(
    model: torch.nn.Module,
    image: torch.Tensor,
    saliency: np.ndarray,
    *,
    target_class: int = 1,
    target_mask: Optional[torch.Tensor] = None,
    pixel_mask: Optional[np.ndarray] = None,
    steps: int = 20,
    baseline: str = "mean",
) -> Dict[str, float]:
    """Compute deletion/insertion AUC for a saliency map."""
    if image.ndim != 4:
        raise ValueError("image tensor must be shaped (B, C, H, W)")

    device = image.device
    saliency_flat = saliency.reshape(-1)
    if pixel_mask is not None:
        pixel_mask = pixel_mask.astype(bool)
        if pixel_mask.shape != saliency.shape:
            raise ValueError("pixel_mask must match saliency spatial shape")
        mask_flat = pixel_mask.reshape(-1)
        masked_indices = np.flatnonzero(mask_flat)
        if masked_indices.size == 0:
            return {
                "deletion_auc": float("nan"),
                "insertion_auc": float("nan"),
                "deletion_score_0": float("nan"),
                "insertion_score_0": float("nan"),
            }
        saliency_masked = saliency_flat[masked_indices]
        order_local = np.argsort(saliency_masked)[::-1]
        order = masked_indices[order_local]
        n_pixels = masked_indices.size
    else:
        order = np.argsort(saliency_flat)[::-1].copy()
        n_pixels = saliency_flat.size

    if baseline == "mean":
        baseline_value = float(image.mean().item())
    elif baseline == "zero":
        baseline_value = 0.0
    else:
        baseline_value = float(baseline)

    image_del = image.clone()
    image_ins = torch.full_like(image, baseline_value)

    deletion_scores = []
    insertion_scores = []
    prev_k = 0
    for step in range(steps + 1):
        k = int((step / steps) * n_pixels)
        if k > prev_k:
            idx = order[prev_k:k]
            idx_t = torch.from_numpy(idx).to(device)
            image_del.view(-1)[idx_t] = baseline_value
            image_ins.view(-1)[idx_t] = image.view(-1)[idx_t]
            prev_k = k

        deletion_scores.append(
            _score_model(model, image_del, target_class=target_class, target_mask=target_mask)
        )
        insertion_scores.append(
            _score_model(model, image_ins, target_class=target_class, target_mask=target_mask)
        )

    fractions = np.linspace(0, 1, steps + 1)
    deletion_auc = float(np.trapz(deletion_scores, fractions))
    insertion_auc = float(np.trapz(insertion_scores, fractions))

    return {
        "deletion_auc": deletion_auc,
        "insertion_auc": insertion_auc,
        "deletion_score_0": float(deletion_scores[0]),
        "insertion_score_0": float(insertion_scores[0]),
    }
