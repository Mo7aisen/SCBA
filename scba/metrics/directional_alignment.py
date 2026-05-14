"""
Auxiliary directional alignment metrics for SCBA++.

This module is intentionally additive: it does not replace the existing binary
Directional Consistency (DC) metric. It provides:

1) `directional_alignment_score(...)`:
   A cosine-based alignment score between the CoA shift vector and the expected
   direction toward the ROI centroid.

2) `directional_alignment_proxy_from_distances(...)`:
   A low-risk proxy that can be computed from existing stored metrics without
   recomputing saliency maps. It uses the relationship between radial movement
   toward the ROI (distance decrease) and total shift magnitude.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def _to_xy(point: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(point), dtype=float)
    if arr.shape != (2,):
        raise ValueError("Expected a 2D point (y, x).")
    return arr


def directional_alignment_score(
    coa_original: Iterable[float],
    coa_perturbed: Iterable[float],
    roi_centroid: Iterable[float],
    *,
    eps: float = 1e-6,
) -> float:
    """
    Continuous directional alignment using cosine similarity.

    Returns:
        float in [0, 1] where:
        - 1.0 = perfect alignment with expected direction
        - 0.5 = perpendicular movement
        - 0.0 = opposite direction
    """
    coa_orig = _to_xy(coa_original)
    coa_pert = _to_xy(coa_perturbed)
    roi = _to_xy(roi_centroid)

    shift = coa_pert - coa_orig
    expected = roi - coa_orig

    shift_norm = float(np.linalg.norm(shift))
    expected_norm = float(np.linalg.norm(expected))
    if shift_norm < eps or expected_norm < eps:
        return 0.0

    cos_sim = float(np.dot(shift, expected) / (shift_norm * expected_norm))
    cos_sim = float(np.clip(cos_sim, -1.0, 1.0))

    # Map [-1, 1] -> [0, 1] so that 0.5 corresponds to perpendicular movement.
    return float(0.5 * (cos_sim + 1.0))


def directional_alignment_proxy_from_distances(
    *,
    shift_distance: float,
    distance_to_roi: float,
    eps: float = 1e-6,
) -> float:
    """
    Proxy directional alignment computed from stored SCBA metrics.

    Let `v` be the CoA shift vector (original -> perturbed) and `r` be the radial
    distance from the CoA to the ROI centroid. The stored metric `distance_to_roi`
    equals r_orig - r_pert (positive when moving closer to the ROI centroid).

    For small shifts relative to r_orig, we have the first-order approximation:
        r_orig - r_pert ≈ ||v|| * cos(theta)
    where theta is the angle between v and the expected direction.

    This proxy estimates cos(theta) as:
        cos_hat = (r_orig - r_pert) / ||v|| = distance_to_roi / shift_distance
    and maps it to [0, 1] using the same convention as `directional_alignment_score`.

    Returns:
        float in [0, 1] where:
        - 1.0 = movement toward ROI
        - 0.5 = approximately perpendicular (no radial progress)
        - 0.0 = movement away from ROI
    """
    denom = float(shift_distance) + eps
    cos_hat = float(distance_to_roi) / denom
    cos_hat = float(np.clip(cos_hat, -1.0, 1.0))
    return float(0.5 * (cos_hat + 1.0))


def directional_alignment_proxy_batch(
    shift_distances: np.ndarray,
    distances_to_roi: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Vectorized wrapper around `directional_alignment_proxy_from_distances`."""
    shift_distances = np.asarray(shift_distances, dtype=float)
    distances_to_roi = np.asarray(distances_to_roi, dtype=float)
    cos_hat = distances_to_roi / (shift_distances + eps)
    cos_hat = np.clip(cos_hat, -1.0, 1.0)
    return 0.5 * (cos_hat + 1.0)

