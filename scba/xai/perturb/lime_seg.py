"""
LIME (Locally Interpretable Model-agnostic Explanations) for Segmentation.

Adapted for dense prediction tasks like medical image segmentation.
Uses superpixel-based perturbations and linear model approximation.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from skimage.segmentation import slic, quickshift
from tqdm import tqdm

from scba.xai.common import ExplainerBase, SaliencyMap


class LIMESegmentation(ExplainerBase):
    """
    LIME for segmentation tasks.

    Generates importance map by perturbing superpixels and fitting
    a linear model to approximate the segmentation model locally.
    """

    def __init__(self, model, device="cuda"):
        super().__init__(model, device)

    def explain(
        self,
        image: torch.Tensor,
        target_class: int = 1,
        target_mask: torch.Tensor = None,
        n_samples: int = 1000,
        n_segments: int = 100,
        segmentation_method: str = "slic",
        kernel_width: float = 0.25,
        batch_size: int = 32,
        normalize: bool = True,
        seed: int = None,
        **kwargs,
    ) -> SaliencyMap:
        """
        Generate LIME explanation for segmentation.

        Args:
            image: (1, C, H, W) or (C, H, W) input
            target_class: Class to explain
            target_mask: Not used for LIME (black-box method)
            n_samples: Number of perturbed samples
            n_segments: Number of superpixels
            segmentation_method: "slic" or "quickshift"
            kernel_width: Width of exponential kernel for weighting
            batch_size: Batch size for processing perturbed samples
            normalize: Whether to normalize output
            seed: Random seed

        Returns:
            SaliencyMap object
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        image = self._prepare_image(image)
        _, C, H, W = image.shape

        # Convert to numpy for superpixel segmentation
        image_np = image.squeeze(0).cpu().numpy()  # (C, H, W)
        if C == 1:
            image_for_seg = image_np[0]  # (H, W)
        else:
            image_for_seg = np.transpose(image_np, (1, 2, 0))  # (H, W, C)

        # Normalize for superpixel segmentation
        img_min, img_max = image_for_seg.min(), image_for_seg.max()
        if img_max > img_min:
            image_for_seg = (image_for_seg - img_min) / (img_max - img_min)

        # Generate superpixels
        print(f"  Generating {n_segments} superpixels using {segmentation_method}...")
        if segmentation_method == "slic":
            segments = slic(image_for_seg, n_segments=n_segments, compactness=10, sigma=1)
        elif segmentation_method == "quickshift":
            segments = quickshift(image_for_seg, kernel_size=4, max_dist=200, ratio=0.2)
        else:
            raise ValueError(f"Unknown segmentation method: {segmentation_method}")

        n_features = len(np.unique(segments))
        print(f"  Generated {n_features} unique superpixels")

        # Get original prediction (mean probability in target class)
        with torch.no_grad():
            original_output = self.model(image)
            original_pred = torch.softmax(original_output, dim=1)[:, target_class].mean().item()

        # Generate perturbations
        print(f"  Generating {n_samples} perturbations...")
        data = np.random.randint(0, 2, (n_samples, n_features))
        predictions = []

        # Adjust batch size for large images
        if H * W > 512 * 512:
            batch_size = min(batch_size, 16)
            print(f"  Reduced batch_size to {batch_size} for {H}x{W} image")

        # Process perturbations in batches
        for i in tqdm(range(0, n_samples, batch_size), desc="LIME perturbations"):
            batch_data = data[i:i + batch_size]
            batch_images = []

            for sample in batch_data:
                # Create perturbed image
                perturbed = image_np.copy()
                for j, val in enumerate(sample):
                    if val == 0:
                        # Zero out superpixel
                        perturbed[:, segments == j] = 0

                batch_images.append(torch.from_numpy(perturbed).float())

            batch_tensor = torch.stack(batch_images).to(self.device)

            with torch.no_grad():
                outputs = self.model(batch_tensor)
                preds = torch.softmax(outputs, dim=1)[:, target_class].mean(dim=(1, 2))
                predictions.extend(preds.cpu().numpy())

            # Clear GPU memory
            del batch_tensor, outputs, preds
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        predictions = np.array(predictions)

        # Calculate distances (cosine similarity between binary vectors)
        distances = np.sqrt(np.sum((data - 1) ** 2, axis=1))

        # Calculate weights using exponential kernel
        weights = np.exp(-(distances ** 2) / (kernel_width ** 2))

        # Fit weighted linear model
        print(f"  Fitting weighted linear model...")
        model = Ridge(alpha=1.0)
        model.fit(data, predictions, sample_weight=weights)

        # Get feature importance (coefficients)
        feature_importance = model.coef_

        # Map superpixel importance back to pixel space
        importance_map = np.zeros((H, W), dtype=np.float32)
        for i, importance in enumerate(feature_importance):
            importance_map[segments == i] = importance

        raw_map = importance_map.copy()

        if normalize:
            importance_map = self._normalize_saliency(importance_map)

        metadata = {
            "target_class": target_class,
            "n_samples": n_samples,
            "n_segments": n_segments,
            "n_features": n_features,
            "segmentation_method": segmentation_method,
            "r2_score": float(model.score(data, predictions, sample_weight=weights)),
            "original_pred": float(original_pred),
        }

        return SaliencyMap(
            map=importance_map, raw_map=raw_map, method="lime", metadata=metadata
        )
