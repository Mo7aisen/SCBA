import numpy as np

from scba.metrics.cf_consistency import directional_consistency


def _one_hot_map(h: int, w: int, y: int, x: int) -> np.ndarray:
    sal = np.zeros((h, w), dtype=float)
    sal[y, x] = 1.0
    return sal


def test_directional_consistency_binary_forward_only_toward_roi() -> None:
    # Scientific justification: manuscript DC is binary and forward-only:
    # DC=1 iff the perturbed CoA moves closer to the ROI centroid.
    roi = np.zeros((11, 11), dtype=np.uint8)
    roi[0, 0] = 1

    sal_orig = _one_hot_map(11, 11, y=10, x=10)
    sal_pert = _one_hot_map(11, 11, y=5, x=5)

    out = directional_consistency(sal_orig, sal_pert, roi)
    assert out["directional_consistency"] == 1.0
    assert out["directional_consistency"] in (0.0, 1.0)
    assert out["forward_movement_px"] > 0


def test_directional_consistency_binary_forward_only_away_from_roi() -> None:
    roi = np.zeros((11, 11), dtype=np.uint8)
    roi[0, 0] = 1

    sal_orig = _one_hot_map(11, 11, y=2, x=2)
    sal_pert = _one_hot_map(11, 11, y=4, x=4)

    out = directional_consistency(sal_orig, sal_pert, roi)
    assert out["directional_consistency"] == 0.0
    assert out["directional_consistency"] in (0.0, 1.0)
    assert out["forward_movement_px"] < 0


def test_directional_consistency_respects_threshold_px() -> None:
    roi = np.zeros((11, 11), dtype=np.uint8)
    roi[0, 0] = 1

    sal_orig = _one_hot_map(11, 11, y=4, x=0)
    sal_pert = _one_hot_map(11, 11, y=3, x=0)

    out = directional_consistency(sal_orig, sal_pert, roi, threshold_px=1.5)
    assert np.isclose(out["forward_movement_px"], 1.0)
    assert out["directional_consistency"] == 0.0
