"""
Gaussian nodule surrogate generator for counterfactual lesion insertion.

Creates realistic synthetic lesions with controlled intensity and placement.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage


@dataclass
class NoduleConfig:
    """Configuration for nodule generation."""

    sigma_x: float = 6.0  # Gaussian width in x
    sigma_y: float = 6.0  # Gaussian width in y
    rotation_deg: float = 0.0  # Rotation angle
    delta_intensity: float = 0.3  # Intensity contrast relative to background
    margin_px: int = 20  # Margin from lung border
    use_border_adjacent: bool = False  # Allow placement near borders
    seed: int = 42  # Random seed


class GaussianNoduleGenerator:
    """
    Generate synthetic Gaussian nodule surrogates.

    Creates controlled lesion-like perturbations for counterfactual analysis.
    """

    def __init__(self, config: NoduleConfig = None):
        self.config = config or NoduleConfig()

    def insert_nodule(
        self, image: np.ndarray, lung_mask: np.ndarray, center: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """
        Insert Gaussian nodule into image.

        Args:
            image: (H, W) grayscale image, float32 in [0, 1]
            lung_mask: (H, W) binary mask of lung parenchyma
            center: Optional (y, x) center. If None, random placement

        Returns:
            Tuple of:
                - image_with_nodule: (H, W) perturbed image
                - nodule_mask: (H, W) binary mask of nodule
                - center: (y, x) nodule center
        """
        np.random.seed(self.config.seed)

        H, W = image.shape

        # 1. Sample placement location
        if center is None:
            center = self._sample_location(lung_mask)

        # 2. Generate Gaussian blob
        nodule_intensity, nodule_mask = self._generate_gaussian_blob(H, W, center)

        # 3. Blend nodule into image
        # Match local intensity statistics
        local_mean = self._get_local_intensity(image, lung_mask, center)
        nodule_intensity = nodule_intensity * self.config.delta_intensity + local_mean

        # Composite
        image_with_nodule = image.copy()
        image_with_nodule = (
            image_with_nodule * (1 - nodule_mask) + nodule_intensity * nodule_mask
        )

        # Add subtle noise to match grain
        noise = np.random.normal(0, 0.01, image.shape)
        image_with_nodule = np.clip(image_with_nodule + noise * nodule_mask, 0, 1)

        # Convert nodule_mask to binary
        nodule_mask_binary = (nodule_mask > 0.1).astype(np.uint8)

        return image_with_nodule, nodule_mask_binary, center

    def _sample_location(self, lung_mask: np.ndarray) -> Tuple[int, int]:
        """
        Sample random location within lung parenchyma.

        Args:
            lung_mask: (H, W) binary lung mask

        Returns:
            (y, x) center coordinates
        """
        # Erode mask to avoid borders
        if not self.config.use_border_adjacent:
            from skimage import morphology

            selem = morphology.disk(self.config.margin_px)
            valid_region = morphology.binary_erosion(lung_mask, selem)
        else:
            valid_region = lung_mask

        # Find valid coordinates
        coords = np.argwhere(valid_region > 0)

        if len(coords) == 0:
            # Fallback: use full lung mask
            coords = np.argwhere(lung_mask > 0)

        if len(coords) == 0:
            # Fallback: center of image
            H, W = lung_mask.shape
            return (H // 2, W // 2)

        # Random sample
        idx = np.random.randint(0, len(coords))
        return tuple(coords[idx])

    def _generate_gaussian_blob(
        self, H: int, W: int, center: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate 2D anisotropic Gaussian blob.

        Args:
            H, W: Image dimensions
            center: (y, x) center

        Returns:
            Tuple of:
                - intensity: (H, W) Gaussian intensity map
                - mask: (H, W) soft mask [0, 1]
        """
        y, x = np.ogrid[:H, :W]
        cy, cx = center

        # Rotation matrix
        theta = np.deg2rad(self.config.rotation_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # Rotated coordinates
        y_rot = (y - cy) * cos_t - (x - cx) * sin_t
        x_rot = (y - cy) * sin_t + (x - cx) * cos_t

        # Anisotropic Gaussian
        exponent = (x_rot ** 2) / (2 * self.config.sigma_x ** 2) + (y_rot ** 2) / (
            2 * self.config.sigma_y ** 2
        )
        gaussian = np.exp(-exponent)

        # Normalize
        gaussian = gaussian / (gaussian.max() + 1e-10)

        return gaussian.astype(np.float32), gaussian.astype(np.float32)

    def _get_local_intensity(
        self, image: np.ndarray, lung_mask: np.ndarray, center: Tuple[int, int], radius: int = 30
    ) -> float:
        """
        Get local mean intensity around center.

        Args:
            image: (H, W) image
            lung_mask: (H, W) mask
            center: (y, x) center
            radius: Sampling radius

        Returns:
            Mean intensity
        """
        cy, cx = center
        H, W = image.shape

        # Extract local region
        y_min = max(0, cy - radius)
        y_max = min(H, cy + radius)
        x_min = max(0, cx - radius)
        x_max = min(W, cx + radius)

        local_img = image[y_min:y_max, x_min:x_max]
        local_mask = lung_mask[y_min:y_max, x_min:x_max]

        if local_mask.sum() == 0:
            return image[lung_mask > 0].mean()

        return local_img[local_mask > 0].mean()


def insert_gaussian_nodule(
    image: np.ndarray,
    lung_mask: np.ndarray,
    sigma: float = 6.0,
    delta_intensity: float = 0.3,
    center: Optional[Tuple[int, int]] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Convenience function for nodule insertion.

    Args:
        image: (H, W) float32 image
        lung_mask: (H, W) uint8 binary mask
        sigma: Gaussian width (isotropic)
        delta_intensity: Intensity contrast
        center: Optional (y, x) center
        seed: Random seed

    Returns:
        (image_with_nodule, nodule_mask, center)
    """
    config = NoduleConfig(
        sigma_x=sigma,
        sigma_y=sigma,
        delta_intensity=delta_intensity,
        seed=seed,
    )
    generator = GaussianNoduleGenerator(config)
    return generator.insert_nodule(image, lung_mask, center)
