"""
Boundary mask utilities for lung-specific faithfulness metrics.
"""

from __future__ import annotations

import numpy as np
from skimage import morphology


def border_band_from_mask(mask: np.ndarray, band_px: int = 6) -> np.ndarray:
    """Return a binary band around the lung boundary."""
    if band_px < 1:
        raise ValueError("band_px must be >= 1")

    mask_uint8 = (mask > 0).astype(np.uint8)
    if mask_uint8.sum() == 0:
        return np.zeros_like(mask_uint8)

    selem = morphology.disk(band_px)
    dilated = morphology.binary_dilation(mask_uint8, selem)
    eroded = morphology.binary_erosion(mask_uint8, selem)
    band = np.logical_xor(dilated, eroded).astype(np.uint8)
    return band
