"""
Counterfactual consistency metrics for SCBA.

Novel metrics to evaluate whether XAI explanations follow causal edits:
- AM-ROI: Attribution Mass in ROI
- ΔAM-ROI: Change in Attribution Mass
- CoA-Δ: Center of Attribution shift
- Directional Consistency
"""

import numpy as np
from typing import Dict, Optional, Tuple


def attribution_mass_roi(saliency: np.ndarray, roi: np.ndarray) -> float:
    """
    Compute attribution mass inside ROI.

    AM-ROI = ∑_{p∈ROI} S(p) where S is normalized saliency

    Args:
        saliency: (H, W) saliency map (should be normalized so ∑S = 1)
        roi: (H, W) binary ROI mask

    Returns:
        Fraction of attribution mass in ROI [0, 1]
    """
    # Normalize saliency
    saliency_norm = saliency / (saliency.sum() + 1e-10)

    # Compute mass in ROI
    am_roi = (saliency_norm * roi).sum()

    return float(am_roi)


def delta_attribution_mass_roi(
    saliency_original: np.ndarray,
    saliency_perturbed: np.ndarray,
    roi: np.ndarray,
) -> float:
    """
    Compute change in attribution mass in ROI.

    ΔAM-ROI = AM-ROI(perturbed) - AM-ROI(original)

    Positive value means attribution increased in ROI after perturbation.

    Args:
        saliency_original: Original saliency map
        saliency_perturbed: Perturbed saliency map
        roi: Binary ROI mask

    Returns:
        Change in attribution mass [-1, 1]
    """
    am_orig = attribution_mass_roi(saliency_original, roi)
    am_pert = attribution_mass_roi(saliency_perturbed, roi)

    return am_pert - am_orig


def center_of_attribution(saliency: np.ndarray) -> Tuple[float, float]:
    """
    Compute center of attribution (centroid of saliency map).

    CoA = (∑ y·S(y,x) / ∑S, ∑ x·S(y,x) / ∑S)

    Args:
        saliency: (H, W) saliency map

    Returns:
        (y_center, x_center) coordinates
    """
    # Normalize
    saliency_norm = saliency / (saliency.sum() + 1e-10)

    # Coordinate grids
    H, W = saliency.shape
    y_coords, x_coords = np.mgrid[0:H, 0:W]

    # Weighted centroid
    y_center = (saliency_norm * y_coords).sum()
    x_center = (saliency_norm * x_coords).sum()

    return float(y_center), float(x_center)


def coa_shift(
    saliency_original: np.ndarray,
    saliency_perturbed: np.ndarray,
    roi_center: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    """
    Compute Center of Attribution shift.

    Args:
        saliency_original: Original saliency
        saliency_perturbed: Perturbed saliency
        roi_center: Optional ROI center. If provided, also compute distance to ROI

    Returns:
        Dictionary with:
            - shift_distance: Euclidean distance CoA moved
            - shift_y, shift_x: Components of shift
            - distance_to_roi: Distance moved toward ROI (if roi_center provided)
    """
    coa_orig = center_of_attribution(saliency_original)
    coa_pert = center_of_attribution(saliency_perturbed)

    shift_y = coa_pert[0] - coa_orig[0]
    shift_x = coa_pert[1] - coa_orig[1]

    shift_distance = np.sqrt(shift_y ** 2 + shift_x ** 2)

    result = {
        "shift_distance": float(shift_distance),
        "shift_y": float(shift_y),
        "shift_x": float(shift_x),
    }

    # Distance to ROI center
    if roi_center is not None:
        dist_orig = np.sqrt(
            (coa_orig[0] - roi_center[0]) ** 2 + (coa_orig[1] - roi_center[1]) ** 2
        )
        dist_pert = np.sqrt(
            (coa_pert[0] - roi_center[0]) ** 2 + (coa_pert[1] - roi_center[1]) ** 2
        )
        # Positive if moved closer to ROI
        result["distance_to_roi"] = float(dist_orig - dist_pert)

    return result


def directional_consistency(
    saliency_original: np.ndarray,
    saliency_perturbed: np.ndarray,
    saliency_repaired: np.ndarray,
    roi: np.ndarray,
    threshold: float = 0.05,
) -> Dict[str, float]:
    """
    Compute directional consistency metric.

    Checks if:
    1. CoA moved toward ROI after perturbation
    2. CoA returned toward original after repair

    Args:
        saliency_original: Original saliency
        saliency_perturbed: Perturbed saliency
        saliency_repaired: Repaired saliency
        roi: Binary ROI mask
        threshold: Minimum movement to consider significant

    Returns:
        Dictionary with consistency score and components
    """
    # Get ROI center
    roi_coords = np.argwhere(roi > 0)
    if len(roi_coords) == 0:
        return {"directional_consistency": 0.0, "forward_correct": False, "backward_correct": False}

    roi_center = roi_coords.mean(axis=0)  # (y, x)

    # CoA shifts
    coa_orig = center_of_attribution(saliency_original)
    coa_pert = center_of_attribution(saliency_perturbed)
    coa_repair = center_of_attribution(saliency_repaired)

    # Distance to ROI
    dist_orig = np.sqrt(
        (coa_orig[0] - roi_center[0]) ** 2 + (coa_orig[1] - roi_center[1]) ** 2
    )
    dist_pert = np.sqrt(
        (coa_pert[0] - roi_center[0]) ** 2 + (coa_pert[1] - roi_center[1]) ** 2
    )
    dist_repair = np.sqrt(
        (coa_repair[0] - roi_center[0]) ** 2 + (coa_repair[1] - roi_center[1]) ** 2
    )

    # Forward consistency: did CoA move toward ROI?
    forward_movement = dist_orig - dist_pert
    forward_correct = forward_movement > threshold

    # Backward consistency: did CoA return toward original?
    backward_movement = dist_pert - dist_repair
    backward_correct = backward_movement > threshold

    # Overall consistency
    if forward_correct and backward_correct:
        consistency_score = 1.0
    elif forward_correct or backward_correct:
        consistency_score = 0.5
    else:
        consistency_score = 0.0

    return {
        "directional_consistency": float(consistency_score),
        "forward_correct": bool(forward_correct),
        "backward_correct": bool(backward_correct),
        "forward_movement": float(forward_movement),
        "backward_movement": float(backward_movement),
    }


def saliency_entropy(saliency: np.ndarray) -> float:
    """
    Compute entropy of saliency map (compactness measure).

    Lower entropy = more compact/focused explanation.

    Args:
        saliency: (H, W) saliency map

    Returns:
        Entropy in nats
    """
    # Normalize to probability distribution
    saliency_norm = saliency / (saliency.sum() + 1e-10)

    # Remove zeros
    saliency_nonzero = saliency_norm[saliency_norm > 0]

    # Entropy
    entropy = -(saliency_nonzero * np.log(saliency_nonzero + 1e-10)).sum()

    return float(entropy)


def compute_cf_metrics(
    saliency_original: np.ndarray,
    saliency_perturbed: np.ndarray,
    saliency_repaired: np.ndarray,
    roi: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all counterfactual consistency metrics.

    Args:
        saliency_original: Original saliency map
        saliency_perturbed: Perturbed saliency map
        saliency_repaired: Repaired saliency map
        roi: Binary ROI mask

    Returns:
        Dictionary with all CF metrics
    """
    # AM-ROI metrics
    am_roi_orig = attribution_mass_roi(saliency_original, roi)
    am_roi_pert = attribution_mass_roi(saliency_perturbed, roi)
    am_roi_repair = attribution_mass_roi(saliency_repaired, roi)

    delta_am_roi = delta_attribution_mass_roi(saliency_original, saliency_perturbed, roi)

    # CoA metrics
    roi_center = np.argwhere(roi > 0).mean(axis=0) if roi.sum() > 0 else None
    coa_shift_metrics = coa_shift(saliency_original, saliency_perturbed, roi_center)

    # Directional consistency
    dc_metrics = directional_consistency(
        saliency_original, saliency_perturbed, saliency_repaired, roi
    )

    # Entropy
    entropy_orig = saliency_entropy(saliency_original)
    entropy_pert = saliency_entropy(saliency_perturbed)

    return {
        "am_roi_original": am_roi_orig,
        "am_roi_perturbed": am_roi_pert,
        "am_roi_repaired": am_roi_repair,
        "delta_am_roi": delta_am_roi,
        **coa_shift_metrics,
        **dc_metrics,
        "entropy_original": entropy_orig,
        "entropy_perturbed": entropy_pert,
    }
