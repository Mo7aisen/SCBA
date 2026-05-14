"""
Border edit generators for counterfactual segmentation audits.

Implements morphological operations, TPS warping, and Poisson blending for
realistic border perturbations.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage
from scipy.interpolate import Rbf
from skimage import measure, morphology


@dataclass
class BorderEditConfig:
    """Configuration for border edit operations."""

    radius_px: int = 4  # Morphology radius
    operation: str = "dilate"  # 'dilate', 'erode', 'open', 'close'
    band_px: int = 12  # Width of ROI band around contour
    area_budget: float = 0.50  # Max |Δarea| as fraction (increased from 0.10 to 0.50 for realistic edits)
    seed: int = 42  # Random seed
    blend_method: str = "poisson"  # 'poisson' or 'alpha'
    warp_mode: str = "tps"  # 'tps' or 'none'
    warp_points: int = 160  # Number of contour points for TPS fitting
    warp_smooth: float = 1.0  # TPS smoothing factor
    warp_margin_px: int = 20  # Padding around ROI band for warping
    min_warp_points: int = 25  # Minimum points needed to attempt warp
    min_ssim: float = 0.60  # Minimum SSIM to accept counterfactual image
    max_intensity_delta: float = 0.20  # Max mean intensity change inside ROI


class BorderEditor:
    """
    Generate controlled border perturbations for counterfactual analysis.

    Creates realistic border edits using morphological operations, optional
    TPS warping, and seamless Poisson blending.
    """

    def __init__(self, config: BorderEditConfig = None):
        self.config = config or BorderEditConfig()

    def apply_border_edit(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        return_metadata: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply border edit to image and mask.

        Args:
            image: (H, W) grayscale image, float32 in [0, 1]
            mask: (H, W) binary mask, uint8

        Returns:
            Tuple of:
                - perturbed_image: (H, W) edited image
                - perturbed_mask: (H, W) edited mask
                - roi_band: (H, W) binary ROI band for evaluation
            If return_metadata=True, also returns a metadata dict.
        """
        np.random.seed(self.config.seed)

        # Ensure proper input formats
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        # Binarize mask if needed
        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8)

        # 1. Morphological perturbation
        perturbed_mask = self._morph_edit(mask)

        # 2. Validate area budget
        area_orig = int(mask.sum())
        area_pert = int(perturbed_mask.sum())
        delta_area = abs(area_pert - area_orig) / float(max(area_orig, 1))

        if not self._check_area_budget(mask, perturbed_mask):
            print(f"Warning: Area budget ({self.config.area_budget:.2f}) exceeded (Δ={delta_area:.3f})")
            print(f"  Original area: {area_orig}, Perturbed area: {area_pert}")
            print(f"  Using smaller radius or increase area_budget")
            perturbed_mask = mask.copy()

        # 3. Extract ROI band (symmetric contour band)
        roi_band = self._extract_roi_band(mask, perturbed_mask)

        # 4. Warp image to match new mask (TPS)
        warped_image, warp_meta = self._warp_image_to_mask(
            image, mask, perturbed_mask, roi_band
        )

        # 5. Seamless blending
        if self.config.blend_method == "poisson":
            perturbed_image, blend_meta = self._poisson_blend(warped_image, image, roi_band)
        else:
            perturbed_image = self._alpha_blend(warped_image, image, roi_band)
            blend_meta = {"requested": "alpha", "used": "alpha", "poisson_ok": None}

        validation = self._validate_counterfactual(image, perturbed_image, roi_band)
        if validation["ssim"] < self.config.min_ssim or abs(validation["mean_delta"]) > self.config.max_intensity_delta:
            perturbed_image = self._alpha_blend(perturbed_image, image, roi_band)
            validation["fallback"] = "alpha_blend"
            # Scientific justification (M3): track when realism constraints force
            # a fallback blend so reviewers can audit failure rates.
            blend_meta = {**blend_meta, "used": "alpha", "validation_fallback": True}

        if return_metadata:
            metadata = {"warp": warp_meta, "validation": validation, "blend": blend_meta}
            return perturbed_image, perturbed_mask, roi_band, metadata

        return perturbed_image, perturbed_mask, roi_band

    def _morph_edit(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operation to mask."""
        selem = morphology.disk(self.config.radius_px)

        if self.config.operation == "dilate":
            perturbed = morphology.binary_dilation(mask, selem)
        elif self.config.operation == "erode":
            perturbed = morphology.binary_erosion(mask, selem)
        elif self.config.operation == "open":
            perturbed = morphology.binary_opening(mask, selem)
        elif self.config.operation == "close":
            perturbed = morphology.binary_closing(mask, selem)
        else:
            raise ValueError(f"Unknown operation: {self.config.operation}")

        return perturbed.astype(np.uint8)

    def _check_area_budget(self, mask_orig: np.ndarray, mask_pert: np.ndarray) -> bool:
        """Check if area change is within budget."""
        # Use int() to avoid overflow with large numpy ints
        area_orig = int(mask_orig.sum())
        area_pert = int(mask_pert.sum())

        if area_orig == 0:
            return True

        delta_area = abs(area_pert - area_orig) / float(area_orig)
        return delta_area <= self.config.area_budget

    def _extract_roi_band(
        self, mask_orig: np.ndarray, mask_pert: np.ndarray
    ) -> np.ndarray:
        """
        Extract symmetric contour band around the edit.

        ROI band = dilated symmetric difference of original and perturbed masks.
        """
        # Symmetric difference: regions that changed
        diff = np.logical_xor(mask_orig, mask_pert).astype(np.uint8)

        # Dilate to get a band
        selem = morphology.disk(self.config.band_px // 2)
        roi_band = morphology.binary_dilation(diff, selem).astype(np.uint8)

        return roi_band

    def _largest_contour(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Return the longest contour as an ordered (N, 2) array in (y, x) coords."""
        contours = measure.find_contours(mask.astype(np.float32), 0.5)
        if not contours:
            return None
        contour = max(contours, key=lambda c: c.shape[0])
        return contour

    def _resample_contour(self, contour: np.ndarray, n_points: int) -> np.ndarray:
        """Resample an ordered contour to a fixed number of points."""
        if contour.shape[0] < 2:
            return contour

        deltas = np.diff(contour, axis=0, append=contour[:1])
        dists = np.sqrt((deltas ** 2).sum(axis=1))
        cumulative = np.cumsum(dists)
        if cumulative[-1] <= 0:
            return contour
        cumulative = np.insert(cumulative[:-1], 0, 0.0)
        target = np.linspace(0, cumulative[-1], n_points, endpoint=False)
        y = np.interp(target, cumulative, contour[:, 0])
        x = np.interp(target, cumulative, contour[:, 1])
        return np.stack([y, x], axis=1)

    def _bbox_from_mask(self, mask: np.ndarray, margin: int) -> Tuple[int, int, int, int]:
        """Compute a padded bounding box from a binary mask."""
        coords = np.argwhere(mask > 0)
        if coords.size == 0:
            return 0, mask.shape[0], 0, mask.shape[1]
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        y0 = max(0, int(y0) - margin)
        x0 = max(0, int(x0) - margin)
        y1 = min(mask.shape[0], int(y1) + margin + 1)
        x1 = min(mask.shape[1], int(x1) + margin + 1)
        return y0, y1, x0, x1

    def _validate_counterfactual(
        self, image_orig: np.ndarray, image_cf: np.ndarray, roi_band: np.ndarray
    ) -> dict:
        """Validate counterfactual realism using SSIM and intensity shifts."""
        from skimage.metrics import structural_similarity as ssim

        if roi_band.sum() == 0:
            return {"ssim": 1.0, "mean_delta": 0.0, "std_delta": 0.0}

        y0, y1, x0, x1 = self._bbox_from_mask(roi_band, margin=4)
        orig_patch = image_orig[y0:y1, x0:x1]
        cf_patch = image_cf[y0:y1, x0:x1]
        if min(orig_patch.shape) < 7:
            ssim_score = 1.0
        else:
            ssim_score = float(ssim(orig_patch, cf_patch, data_range=1.0))

        delta = image_cf[roi_band > 0] - image_orig[roi_band > 0]
        mean_delta = float(np.mean(delta)) if delta.size else 0.0
        std_delta = float(np.std(delta)) if delta.size else 0.0

        return {"ssim": ssim_score, "mean_delta": mean_delta, "std_delta": std_delta}

    def _warp_image_to_mask(
        self,
        image: np.ndarray,
        mask_orig: np.ndarray,
        mask_pert: np.ndarray,
        roi_band: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Warp image using Thin-Plate Spline to match new mask contour.

        Simplified approach: sample correspondence points from contours.
        """
        if self.config.warp_mode == "none":
            return image.copy(), {"warp_applied": False, "reason": "disabled"}

        roi_band = roi_band if roi_band is not None else self._extract_roi_band(mask_orig, mask_pert)
        if roi_band.sum() == 0:
            return image.copy(), {"warp_applied": False, "reason": "empty_roi"}

        contour_orig = self._largest_contour(mask_orig)
        contour_pert = self._largest_contour(mask_pert)

        if contour_orig is None or contour_pert is None:
            return image.copy(), {"warp_applied": False, "reason": "missing_contour"}

        n_points = min(self.config.warp_points, contour_orig.shape[0], contour_pert.shape[0])
        if n_points < self.config.min_warp_points:
            return image.copy(), {"warp_applied": False, "reason": "too_few_points"}

        src = self._resample_contour(contour_orig, n_points)
        dst = self._resample_contour(contour_pert, n_points)

        y0, y1, x0, x1 = self._bbox_from_mask(roi_band, margin=self.config.warp_margin_px)
        grid_y, grid_x = np.mgrid[y0:y1, x0:x1]

        try:
            rbf_x = Rbf(
                src[:, 1], src[:, 0], dst[:, 1], function="thin_plate", smooth=self.config.warp_smooth
            )
            rbf_y = Rbf(
                src[:, 1], src[:, 0], dst[:, 0], function="thin_plate", smooth=self.config.warp_smooth
            )
            map_x = rbf_x(grid_x, grid_y)
            map_y = rbf_y(grid_x, grid_y)
        except Exception as exc:
            return image.copy(), {"warp_applied": False, "reason": f"tps_failed: {exc}"}

        map_x = np.clip(map_x, 0, image.shape[1] - 1).astype(np.float32)
        map_y = np.clip(map_y, 0, image.shape[0] - 1).astype(np.float32)
        warped_patch = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        warped = image.copy()
        warped[y0:y1, x0:x1] = warped_patch

        return warped, {"warp_applied": True, "n_points": int(n_points), "bbox": [y0, y1, x0, x1]}

    def _poisson_blend(
        self, source: np.ndarray, target: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """
        Seamless Poisson blending.

        Args:
            source: Warped image
            target: Original image
            mask: ROI band to blend

        Returns:
            (blended_image, blend_metadata)
        """
        # Convert to uint8 for OpenCV
        source_uint8 = (source * 255).astype(np.uint8)
        target_uint8 = (target * 255).astype(np.uint8)

        if mask.sum() == 0:
            return target, {"requested": "poisson", "used": "alpha", "poisson_ok": False, "reason": "empty_mask"}

        # Find center of mask for Poisson
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return target, {"requested": "poisson", "used": "alpha", "poisson_ok": False, "reason": "no_coords"}

        center = coords.mean(axis=0).astype(int)
        center = (int(center[1]), int(center[0]))  # (x, y) for OpenCV

        # Convert to 3-channel for OpenCV
        if source_uint8.ndim == 2:
            source_uint8 = cv2.cvtColor(source_uint8, cv2.COLOR_GRAY2BGR)
            target_uint8 = cv2.cvtColor(target_uint8, cv2.COLOR_GRAY2BGR)

        mask_uint8 = (mask * 255).astype(np.uint8)

        # Poisson blending
        try:
            blended = cv2.seamlessClone(
                source_uint8, target_uint8, mask_uint8, center, cv2.NORMAL_CLONE
            )
            # Convert back to grayscale and [0, 1]
            if blended.ndim == 3:
                blended = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)
            return blended.astype(np.float32) / 255.0, {"requested": "poisson", "used": "poisson", "poisson_ok": True}
        except Exception as e:
            # Scientific justification (M3): do not silently fall back; record
            # Poisson failures to quantify how often realism depends on alpha blending.
            return (
                self._alpha_blend(source, target, mask),
                {"requested": "poisson", "used": "alpha", "poisson_ok": False, "poisson_error": str(e)},
            )

    def _alpha_blend(
        self, source: np.ndarray, target: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """
        Simple alpha blending.

        Args:
            source: Warped image
            target: Original image
            mask: ROI band

        Returns:
            Blended image
        """
        # Smooth mask for gradual blending
        mask_smooth = ndimage.gaussian_filter(mask.astype(float), sigma=2.0)
        mask_smooth = np.clip(mask_smooth, 0, 1)

        blended = source * mask_smooth + target * (1 - mask_smooth)
        return blended


def apply_border_edit(
    image: np.ndarray,
    mask: np.ndarray,
    radius_px: int = 4,
    operation: str = "dilate",
    band_px: int = 12,
    area_budget: float = 0.50,
    seed: int = 42,
    warp_mode: str = "tps",
    warp_points: int = 160,
    warp_smooth: float = 1.0,
    warp_margin_px: int = 20,
    min_warp_points: int = 25,
    min_ssim: float = 0.60,
    max_intensity_delta: float = 0.20,
    blend_method: str = "poisson",
    return_metadata: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function for border editing.

    Args:
        image: (H, W) float32 image in [0, 1]
        mask: (H, W) uint8 binary mask
        radius_px: Morphology radius (default=4, try 2-3 for smaller changes)
        operation: 'dilate', 'erode', 'open', 'close'
        band_px: ROI band width
        area_budget: Max area change as fraction (default=0.50 for 50% change)
        seed: Random seed
        warp_mode: 'tps' or 'none'
        warp_points: Number of contour points for TPS fitting
        warp_smooth: TPS smoothing factor
        warp_margin_px: Padding around ROI band for warping
        min_warp_points: Minimum points required for TPS warp
        min_ssim: Minimum SSIM to accept counterfactual image
        max_intensity_delta: Max mean intensity delta allowed in ROI
        blend_method: 'poisson' or 'alpha'
        return_metadata: If True, return metadata dict with warp + validation + blend info

    Returns:
        (perturbed_image, perturbed_mask, roi_band)
    """
    config = BorderEditConfig(
        radius_px=radius_px,
        operation=operation,
        band_px=band_px,
        area_budget=area_budget,
        seed=seed,
        warp_mode=warp_mode,
        warp_points=warp_points,
        warp_smooth=warp_smooth,
        warp_margin_px=warp_margin_px,
        min_warp_points=min_warp_points,
        min_ssim=min_ssim,
        max_intensity_delta=max_intensity_delta,
        blend_method=blend_method,
    )
    editor = BorderEditor(config)
    return editor.apply_border_edit(image, mask, return_metadata=return_metadata)
