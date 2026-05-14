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
    roi: np.ndarray,
    threshold_px: float = 0.0,
) -> Dict[str, float]:
    """
    Compute directional consistency metric.

    Scientific definition (manuscript-aligned):
    DC is binary and forward-only. It measures whether the explanation's Center of
    Attribution (CoA) moves closer to the ROI centroid after the counterfactual
    perturbation.

    DC = 1 iff dist(CoA(S'), ROI_c) < dist(CoA(S), ROI_c) by at least threshold_px
    DC = 0 otherwise.

    Note: DC must not depend on any "repair" image/explanation. Repair is a
    separate diagnostic and including it changes the metric's meaning and can
    inflate consistency.

    Args:
        saliency_original: Original saliency
        saliency_perturbed: Perturbed saliency
        roi: Binary ROI mask
        threshold_px: Minimum reduction in distance-to-ROI (in pixels) required
            to count as consistent. Default 0.0 matches the strict inequality in
            the manuscript while allowing explicit robustness sweeps.

    Returns:
        Dictionary with binary DC and diagnostic distances (pixels).
    """
    # Scientific justification: ROI centroid matches the manuscript's ROI_c.
    roi_coords = np.argwhere(roi > 0)
    if len(roi_coords) == 0:
        return {
            "directional_consistency": 0.0,
            "forward_movement_px": 0.0,
            "dist_to_roi_original_px": float("nan"),
            "dist_to_roi_perturbed_px": float("nan"),
        }

    roi_center = roi_coords.mean(axis=0)  # (y, x)

    # CoA locations
    coa_orig = center_of_attribution(saliency_original)
    coa_pert = center_of_attribution(saliency_perturbed)

    # Distances to ROI centroid (pixels)
    dist_orig = np.sqrt(
        (coa_orig[0] - roi_center[0]) ** 2 + (coa_orig[1] - roi_center[1]) ** 2
    )
    dist_pert = np.sqrt(
        (coa_pert[0] - roi_center[0]) ** 2 + (coa_pert[1] - roi_center[1]) ** 2
    )

    # Forward-only movement: positive means CoA moved closer to ROI.
    forward_movement_px = float(dist_orig - dist_pert)
    consistency_score = 1.0 if forward_movement_px > float(threshold_px) else 0.0

    return {
        "directional_consistency": float(consistency_score),
        "forward_movement_px": forward_movement_px,
        "dist_to_roi_original_px": float(dist_orig),
        "dist_to_roi_perturbed_px": float(dist_pert),
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
    *,
    mask_original: Optional[np.ndarray] = None,
    mask_perturbed: Optional[np.ndarray] = None,
    operation: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute all counterfactual consistency metrics.

    Args:
        saliency_original: Original saliency map
        saliency_perturbed: Perturbed saliency map
        saliency_repaired: Repaired saliency map
        roi: Binary ROI mask (legacy: symmetric band around edit)
        mask_original: Optional binary mask used to derive directional ROIs.
        mask_perturbed: Optional binary mask used to derive directional ROIs.
        operation: Optional operation label ("dilate" or "erode") used to select
            the expected ROI (added vs removed pixels) and define aligned sign
            conventions.

    Returns:
        Dictionary with all CF metrics.

    Notes on sign conventions (directional metrics):
        - For DILATE, the expected ROI is the set of added pixels:
              R_add = {p : mask_cf(p)=1 and mask_orig(p)=0}
          Correct behavior implies ΔAM(R_add) > 0.
        - For ERODE, the expected ROI is the set of removed pixels:
              R_rem = {p : mask_orig(p)=1 and mask_cf(p)=0}
          Correct behavior implies ΔAM(R_rem) < 0 (attribution evacuates removed region).
        - We therefore provide an aligned metric that is positive for correct behavior:
              ΔAM_aligned = ΔAM(R_add)                     (dilate)
              ΔAM_aligned = -ΔAM(R_rem) = AM_orig - AM_cf   (erode)
    """
    # --- Legacy ROI band metrics (symmetric band) ---
    am_roi_orig = attribution_mass_roi(saliency_original, roi)
    am_roi_pert = attribution_mass_roi(saliency_perturbed, roi)
    am_roi_repair = attribution_mass_roi(saliency_repaired, roi)

    delta_am_roi = delta_attribution_mass_roi(saliency_original, saliency_perturbed, roi)

    roi_center = np.argwhere(roi > 0).mean(axis=0) if roi.sum() > 0 else None
    coa_shift_metrics = coa_shift(saliency_original, saliency_perturbed, roi_center)
    dc_metrics = directional_consistency(saliency_original, saliency_perturbed, roi)

    # --- Directional ROI metrics (operation-aware) ---
    directional = {
        "has_directional_roi": 0.0,
        "operation": operation,
    }
    if mask_original is not None and mask_perturbed is not None:
        mo = (np.asarray(mask_original) > 0).astype(np.uint8)
        mp = (np.asarray(mask_perturbed) > 0).astype(np.uint8)
        roi_added = ((mp > 0) & (mo == 0)).astype(np.uint8)
        roi_removed = ((mo > 0) & (mp == 0)).astype(np.uint8)
        directional.update(
            {
                "has_directional_roi": 1.0,
                "roi_added_pixels": float(int(roi_added.sum())),
                "roi_removed_pixels": float(int(roi_removed.sum())),
            }
        )

        # Compute raw ΔAM in both directional ROIs (may be 0 if ROI empty)
        am_add_orig = attribution_mass_roi(saliency_original, roi_added) if roi_added.sum() else 0.0
        am_add_pert = attribution_mass_roi(saliency_perturbed, roi_added) if roi_added.sum() else 0.0
        delta_add = float(am_add_pert - am_add_orig)

        am_rem_orig = attribution_mass_roi(saliency_original, roi_removed) if roi_removed.sum() else 0.0
        am_rem_pert = attribution_mass_roi(saliency_perturbed, roi_removed) if roi_removed.sum() else 0.0
        delta_rem = float(am_rem_pert - am_rem_orig)

        directional.update(
            {
                "am_roi_added_original": float(am_add_orig),
                "am_roi_added_perturbed": float(am_add_pert),
                "delta_am_roi_added": float(delta_add),
                "am_roi_removed_original": float(am_rem_orig),
                "am_roi_removed_perturbed": float(am_rem_pert),
                "delta_am_roi_removed": float(delta_rem),
            }
        )

        # Expected ROI for the given operation
        op = str(operation or "").lower().strip()
        if op in {"dilate", "dilation"}:
            roi_expected = roi_added
            delta_expected = delta_add
            delta_aligned = delta_expected  # should be > 0 for correct behavior
            expected_type = "added"
        elif op in {"erode", "erosion"}:
            roi_expected = roi_removed
            delta_expected = delta_rem
            delta_aligned = -delta_expected  # should be > 0 for correct behavior
            expected_type = "removed"
        else:
            roi_expected = None
            delta_expected = float("nan")
            delta_aligned = float("nan")
            expected_type = None

        directional.update(
            {
                "roi_expected_type": expected_type,
                "delta_am_roi_expected": float(delta_expected) if np.isfinite(delta_expected) else float("nan"),
                "delta_am_roi_aligned": float(delta_aligned) if np.isfinite(delta_aligned) else float("nan"),
            }
        )

        if roi_expected is not None and roi_expected.sum() > 0:
            roi_expected_center = np.argwhere(roi_expected > 0).mean(axis=0)
            coa_expected = coa_shift(saliency_original, saliency_perturbed, roi_expected_center)
            dc_expected = directional_consistency(saliency_original, saliency_perturbed, roi_expected)
            directional.update(
                {
                    "shift_distance_expected": float(coa_expected.get("shift_distance", float("nan"))),
                    "distance_to_roi_expected": float(coa_expected.get("distance_to_roi", float("nan"))),
                    "directional_consistency_expected": float(dc_expected.get("directional_consistency", float("nan"))),
                    "forward_movement_expected_px": float(dc_expected.get("forward_movement_px", float("nan"))),
                    "dist_to_roi_expected_original_px": float(dc_expected.get("dist_to_roi_original_px", float("nan"))),
                    "dist_to_roi_expected_perturbed_px": float(dc_expected.get("dist_to_roi_perturbed_px", float("nan"))),
                }
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
        **directional,
    }
