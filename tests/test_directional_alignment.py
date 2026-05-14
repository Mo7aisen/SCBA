import numpy as np

from scba.metrics.directional_alignment import (
    directional_alignment_proxy_from_distances,
    directional_alignment_score,
)


def test_directional_alignment_perfect_alignment_near_one() -> None:
    coa_orig = np.array([100.0, 100.0])
    coa_pert = np.array([110.0, 110.0])
    roi_center = np.array([120.0, 120.0])
    assert directional_alignment_score(coa_orig, coa_pert, roi_center) > 0.95


def test_directional_alignment_opposite_near_zero() -> None:
    coa_orig = np.array([100.0, 100.0])
    coa_pert = np.array([90.0, 90.0])
    roi_center = np.array([120.0, 120.0])
    assert directional_alignment_score(coa_orig, coa_pert, roi_center) < 0.05


def test_directional_alignment_perpendicular_near_half() -> None:
    coa_orig = np.array([100.0, 100.0])
    roi_center = np.array([120.0, 120.0])
    # Expected direction is (20, 20); a perpendicular shift is proportional to (1, -1).
    coa_pert = np.array([110.0, 90.0])
    result = directional_alignment_score(coa_orig, coa_pert, roi_center)
    assert 0.45 < result < 0.55


def test_directional_alignment_proxy_bounds_and_sanity() -> None:
    # Toward ROI: distance_to_roi ~= shift_distance -> score ~= 1.0
    assert directional_alignment_proxy_from_distances(shift_distance=10.0, distance_to_roi=10.0) > 0.95
    # Away from ROI: distance_to_roi ~= -shift_distance -> score ~= 0.0
    assert directional_alignment_proxy_from_distances(shift_distance=10.0, distance_to_roi=-10.0) < 0.05
    # Perpendicular / no radial change: distance_to_roi ~= 0 -> score ~= 0.5
    result = directional_alignment_proxy_from_distances(shift_distance=10.0, distance_to_roi=0.0)
    assert 0.45 < result < 0.55
