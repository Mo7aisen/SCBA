"""
Robustness metrics for explanation maps.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from scipy.stats import spearmanr

from scba.xai.common import explain


def saliency_robustness(
    model: torch.nn.Module,
    image: torch.Tensor,
    saliency: np.ndarray,
    *,
    method: str,
    target_class: int = 1,
    target_mask: Optional[torch.Tensor] = None,
    device: str = "cuda",
    n_samples: int = 3,
    noise_sigma: float = 0.01,
    seed: int = 42,
    explain_kwargs: Optional[Dict[str, object]] = None,
    pixel_mask: Optional[np.ndarray] = None,
    topk_fraction: Optional[float] = None,
    correlation: str = "pearson",
) -> Dict[str, float]:
    """
    Estimate robustness as correlation under Gaussian-noise perturbations.

    Scientific justification (M4): computing correlation over the full image can
    overstate robustness for smooth saliency maps. Optional masking and top-k
    restriction allow robustness to be evaluated within clinically relevant
    regions (e.g., lung mask, boundary band, ROI band).
    """
    if image.ndim != 4:
        raise ValueError("image tensor must be shaped (B, C, H, W)")

    rng = np.random.default_rng(seed)

    if pixel_mask is not None:
        if pixel_mask.shape != saliency.shape:
            raise ValueError("pixel_mask must have the same shape as saliency")
        mask = pixel_mask.astype(bool)
    else:
        mask = np.ones_like(saliency, dtype=bool)

    if topk_fraction is not None:
        if not (0.0 < float(topk_fraction) <= 1.0):
            raise ValueError("topk_fraction must be in (0, 1]")
        flat_ref = saliency[mask].reshape(-1)
        if flat_ref.size == 0:
            # Scientific justification: correlation is undefined with empty mask
            return {"robustness_corr_mean": float("nan"), "robustness_corr_std": float("nan"), "robustness_corr_mask_n": 0}
        k = max(1, int(np.ceil(flat_ref.size * float(topk_fraction))))
        topk_idx = np.argpartition(flat_ref, -k)[-k:]
        mask_positions = np.flatnonzero(mask.reshape(-1))
        topk_positions = mask_positions[topk_idx]
        topk_mask = np.zeros(mask.size, dtype=bool)
        topk_mask[topk_positions] = True
        mask = topk_mask.reshape(mask.shape)

    reference = saliency[mask].reshape(-1)
    if reference.size < 2:
        # Scientific justification: correlation requires at least 2 values
        return {"robustness_corr_mean": float("nan"), "robustness_corr_std": float("nan"), "robustness_corr_mask_n": int(reference.size)}
    correlations = []

    for _ in range(n_samples):
        noise = torch.from_numpy(
            rng.normal(0.0, noise_sigma, size=image.shape).astype(np.float32)
        ).to(image.device)
        perturbed = torch.clamp(image + noise, 0.0, 1.0)
        perturbed_saliency = explain(
            perturbed,
            model,
            method=method,
            target_class=target_class,
            target_mask=target_mask,
            device=device,
            **(explain_kwargs or {}),
        )
        candidate = perturbed_saliency.map[mask].reshape(-1)
        if correlation == "spearman":
            corr = float(spearmanr(reference, candidate).correlation)
        elif correlation == "pearson":
            corr = float(np.corrcoef(reference, candidate)[0, 1])
        else:
            raise ValueError("correlation must be 'pearson' or 'spearman'")
        if not np.isfinite(corr):
            corr = 0.0
        correlations.append(float(corr))

    return {
        "robustness_corr_mean": float(np.mean(correlations)),
        "robustness_corr_std": float(np.std(correlations)),
        "robustness_corr_mask_n": int(reference.size),
    }
