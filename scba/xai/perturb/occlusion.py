"""
Occlusion-based sensitivity analysis.

Sliding window approach to measure importance of different regions.
"""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from scba.xai.common import ExplainerBase, SaliencyMap


class Occlusion(ExplainerBase):
    """
    Occlusion-based explanation.

    Systematically occludes patches and measures output change.
    """

    def __init__(self, model, device="cuda"):
        super().__init__(model, device)

    def explain(
        self,
        image: torch.Tensor,
        target_class: int = 1,
        target_mask: torch.Tensor = None,
        patch_size: int = 16,
        stride: int = 8,
        occlusion_value: float = 0.0,
        normalize: bool = True,
        **kwargs,
    ) -> SaliencyMap:
        """
        Generate occlusion-based explanation.

        Args:
            image: (1, C, H, W) or (C, H, W) input
            target_class: Class to explain
            target_mask: Not used for occlusion
            patch_size: Size of occlusion patch
            stride: Stride for sliding window
            occlusion_value: Value to fill occluded regions
            normalize: Whether to normalize output

        Returns:
            SaliencyMap object
        """
        image = self._prepare_image(image)
        _, C, H, W = image.shape

        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(image)
            baseline_score = baseline_output[:, target_class].mean().item()

        # Initialize importance map
        importance_map = np.zeros((H, W), dtype=np.float32)
        counts = np.zeros((H, W), dtype=np.float32)

        # Slide window
        positions = []
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                positions.append((y, x))

        # Process in batches for efficiency (auto-adjust for large images)
        batch_size = 2 if H * W > 512 * 512 else 8
        print(f"  Occlusion batch_size: {batch_size} for {H}x{W} image ({len(positions)} positions)")
        for i in tqdm(range(0, len(positions), batch_size), desc="Occlusion"):
            batch_positions = positions[i : i + batch_size]
            batch_images = []

            for y, x in batch_positions:
                # Create occluded image
                occluded = image.clone()
                occluded[:, :, y : y + patch_size, x : x + patch_size] = occlusion_value
                batch_images.append(occluded)

            # Stack and process batch
            batch_images = torch.cat(batch_images, dim=0)

            with torch.no_grad():
                outputs = self.model(batch_images)
                scores = outputs[:, target_class].mean(dim=(1, 2))

            # Update importance map
            for j, (y, x) in enumerate(batch_positions):
                # Importance = baseline - occluded (higher = more important)
                importance = baseline_score - scores[j].item()
                importance_map[y : y + patch_size, x : x + patch_size] += importance
                counts[y : y + patch_size, x : x + patch_size] += 1

            # Clear GPU memory after each batch
            del batch_images, outputs, scores
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # Average overlapping regions
        importance_map = importance_map / (counts + 1e-10)

        raw_map = importance_map.copy()

        if normalize:
            importance_map = self._normalize_saliency(importance_map)

        metadata = {
            "target_class": target_class,
            "patch_size": patch_size,
            "stride": stride,
            "baseline_score": baseline_score,
        }

        return SaliencyMap(
            map=importance_map, raw_map=raw_map, method="occlusion", metadata=metadata
        )
