"""
RISE (Random Input Sampling for Explanation) implementation.

Based on:
Petsiuk et al., "RISE: Randomized Input Sampling for Explanation of
Black-box Models", BMVC 2018.
"""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from scba.xai.common import ExplainerBase, SaliencyMap


class RISE(ExplainerBase):
    """
    RISE: Black-box explanation method using random masked inputs.

    Generates importance map by measuring output change when masking
    different regions of the input.
    """

    def __init__(self, model, device="cuda"):
        super().__init__(model, device)

    def explain(
        self,
        image: torch.Tensor,
        target_class: int = 1,
        target_mask: torch.Tensor = None,
        n_masks: int = 2000,
        mask_probability: float = 0.5,
        cell_size: int = 8,
        batch_size: int = 128,
        normalize: bool = True,
        seed: int = None,
        **kwargs,
    ) -> SaliencyMap:
        """
        Generate RISE explanation.

        Args:
            image: (1, C, H, W) or (C, H, W) input
            target_class: Class to explain
            target_mask: Not used for RISE (black-box method)
            n_masks: Number of random masks to generate
            mask_probability: Probability of keeping each cell
            cell_size: Size of mask cells (grid resolution)
            batch_size: Batch size for processing masks
            normalize: Whether to normalize output
            seed: Random seed for mask generation

        Returns:
            SaliencyMap object
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        image = self._prepare_image(image)
        _, C, H, W = image.shape

        # Generate random masks
        masks = self._generate_random_masks(
            H, W, n_masks, mask_probability, cell_size
        )

        # Compute importance scores
        importance_map = np.zeros((H, W), dtype=np.float32)

        n_batches = (n_masks + batch_size - 1) // batch_size

        for i in tqdm(range(n_batches), desc="RISE"):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_masks)
            batch_masks = masks[start:end]  # (B, H, W)

            # Convert to tensor
            batch_masks_t = torch.from_numpy(batch_masks).float().to(self.device)
            batch_masks_t = batch_masks_t.unsqueeze(1)  # (B, 1, H, W)

            # Apply masks to image
            masked_images = image * batch_masks_t  # (B, C, H, W)

            # Get model predictions
            with torch.no_grad():
                outputs = self.model(masked_images)  # (B, C, H, W)
                scores = outputs[:, target_class].mean(dim=(1, 2))  # (B,)

            # Accumulate weighted masks
            for j, score in enumerate(scores):
                importance_map += batch_masks[j] * score.cpu().item()

        # Normalize by number of masks
        importance_map /= n_masks

        raw_map = importance_map.copy()

        if normalize:
            importance_map = self._normalize_saliency(importance_map)

        metadata = {
            "target_class": target_class,
            "n_masks": n_masks,
            "mask_probability": mask_probability,
            "cell_size": cell_size,
        }

        return SaliencyMap(
            map=importance_map, raw_map=raw_map, method="rise", metadata=metadata
        )

    def _generate_random_masks(
        self, H: int, W: int, n_masks: int, p: float, cell_size: int
    ) -> np.ndarray:
        """
        Generate random binary masks.

        Args:
            H, W: Image height and width
            n_masks: Number of masks
            p: Probability of keeping each cell
            cell_size: Size of mask cells

        Returns:
            (n_masks, H, W) array of binary masks
        """
        # Grid size
        grid_h = H // cell_size + 1
        grid_w = W // cell_size + 1

        # Generate random masks on grid
        masks_grid = np.random.rand(n_masks, grid_h, grid_w) < p

        # Upsample to image size
        masks = np.zeros((n_masks, H, W), dtype=np.float32)

        for i in range(n_masks):
            # Repeat each grid cell
            mask = np.repeat(np.repeat(masks_grid[i], cell_size, axis=0), cell_size, axis=1)
            masks[i] = mask[:H, :W]

        return masks
