"""
Repair operations: inpainting to reverse counterfactual edits.
"""

import cv2
import numpy as np


def inpaint_region(
    image: np.ndarray, mask: np.ndarray, method: str = "telea", radius: int = 3
) -> np.ndarray:
    """
    Inpaint region specified by mask.

    Args:
        image: (H, W) float32 image in [0, 1]
        mask: (H, W) uint8 binary mask (1 = region to inpaint)
        method: 'telea' or 'ns' (Navier-Stokes)
        radius: Inpainting radius

    Returns:
        Inpainted image
    """
    # Convert to uint8 for OpenCV
    image_uint8 = (image * 255).astype(np.uint8)

    # Ensure mask is uint8
    mask_uint8 = (mask * 255).astype(np.uint8)

    # Choose method
    if method == "telea":
        flags = cv2.INPAINT_TELEA
    elif method == "ns":
        flags = cv2.INPAINT_NS
    else:
        raise ValueError(f"Unknown inpainting method: {method}")

    # Inpaint
    inpainted = cv2.inpaint(image_uint8, mask_uint8, radius, flags)

    # Convert back to float32
    return inpainted.astype(np.float32) / 255.0


def repair_border_edit(
    image_perturbed: np.ndarray,
    image_original: np.ndarray,
    roi_band: np.ndarray,
    method: str = "direct",
) -> np.ndarray:
    """
    Repair border edit by restoring original or inpainting.

    Args:
        image_perturbed: (H, W) perturbed image
        image_original: (H, W) original image
        roi_band: (H, W) ROI band to repair
        method: 'direct' (restore) or 'inpaint'

    Returns:
        Repaired image
    """
    if method == "direct":
        # Simply restore original in ROI band
        repaired = image_perturbed.copy()
        repaired[roi_band > 0] = image_original[roi_band > 0]
        return repaired
    elif method == "inpaint":
        # Inpaint the ROI band
        return inpaint_region(image_perturbed, roi_band, method="telea")
    else:
        raise ValueError(f"Unknown repair method: {method}")


def repair_nodule(
    image_with_nodule: np.ndarray, nodule_mask: np.ndarray, method: str = "telea"
) -> np.ndarray:
    """
    Remove nodule by inpainting.

    Args:
        image_with_nodule: (H, W) image with nodule
        nodule_mask: (H, W) binary nodule mask
        method: Inpainting method

    Returns:
        Repaired image (nodule removed)
    """
    return inpaint_region(image_with_nodule, nodule_mask, method=method, radius=5)


def compute_repair_quality(
    image_repaired: np.ndarray, image_original: np.ndarray
) -> dict:
    """
    Compute metrics to assess repair quality.

    Args:
        image_repaired: Repaired image
        image_original: Original image (ground truth)

    Returns:
        Dictionary with SSIM and MSE
    """
    from skimage.metrics import structural_similarity as ssim

    # SSIM
    ssim_score = ssim(image_original, image_repaired, data_range=1.0)

    # MSE
    mse = np.mean((image_original - image_repaired) ** 2)

    # PSNR
    if mse > 0:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    else:
        psnr = float("inf")

    return {"ssim": ssim_score, "mse": mse, "psnr": psnr}
