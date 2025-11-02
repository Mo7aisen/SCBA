"""
SHAP (SHapley Additive exPlanations) - KernelSHAP for Segmentation.

Implements Shapley value approximation for dense prediction tasks.
Based on Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions", NeurIPS 2017.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from skimage.segmentation import slic, quickshift
from tqdm import tqdm
from itertools import combinations

from scba.xai.common import ExplainerBase, SaliencyMap


class KernelSHAPSegmentation(ExplainerBase):
    """
    KernelSHAP for segmentation tasks.

    Uses superpixel-based coalitions and weighted linear regression
    to approximate Shapley values efficiently.
    """

    def __init__(self, model, device="cuda"):
        super().__init__(model, device)

    def explain(
        self,
        image: torch.Tensor,
        target_class: int = 1,
        target_mask: torch.Tensor = None,
        n_samples: int = 2000,
        n_segments: int = 50,
        segmentation_method: str = "slic",
        batch_size: int = 32,
        normalize: bool = True,
        seed: int = None,
        **kwargs,
    ) -> SaliencyMap:
        """
        Generate KernelSHAP explanation for segmentation.

        Args:
            image: (1, C, H, W) or (C, H, W) input
            target_class: Class to explain
            target_mask: Not used for SHAP (black-box method)
            n_samples: Number of coalition samples
            n_segments: Number of superpixels (features)
            segmentation_method: "slic" or "quickshift"
            batch_size: Batch size for processing coalitions
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

        # Adjust samples and batch size for computational efficiency
        if n_features > 30:
            n_samples = min(n_samples, 1000)
            print(f"  Reduced n_samples to {n_samples} for {n_features} features")

        if H * W > 512 * 512:
            batch_size = min(batch_size, 16)
            print(f"  Reduced batch_size to {batch_size} for {H}x{W} image")

        # Compute background (all superpixels masked)
        background_image = np.zeros_like(image_np)
        background_tensor = torch.from_numpy(background_image).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            background_output = self.model(background_tensor)
            background_pred = torch.softmax(background_output, dim=1)[:, target_class].mean().item()

        # Get original prediction
        with torch.no_grad():
            original_output = self.model(image)
            original_pred = torch.softmax(original_output, dim=1)[:, target_class].mean().item()

        print(f"  Background pred: {background_pred:.4f}, Original pred: {original_pred:.4f}")

        # Generate coalitions using weighted sampling
        print(f"  Generating {n_samples} coalition samples...")

        # SHAP kernel weights: more weight to coalitions with intermediate sizes
        coalition_data = []
        coalition_weights = []

        for _ in range(n_samples):
            # Sample coalition size with SHAP kernel weighting
            size = np.random.randint(0, n_features + 1)

            # Create coalition (random subset of features)
            coalition = np.zeros(n_features, dtype=int)
            if size > 0:
                active_features = np.random.choice(n_features, size=size, replace=False)
                coalition[active_features] = 1

            coalition_data.append(coalition)

            # SHAP kernel weight
            if size == 0 or size == n_features:
                weight = 10000  # Very high weight for empty and full coalitions
            else:
                weight = (n_features - 1) / (size * (n_features - size))

            coalition_weights.append(weight)

        coalition_data = np.array(coalition_data)
        coalition_weights = np.array(coalition_weights)

        # Evaluate coalitions
        print(f"  Evaluating coalitions...")
        predictions = []

        for i in tqdm(range(0, n_samples, batch_size), desc="SHAP coalitions"):
            batch_data = coalition_data[i:i + batch_size]
            batch_images = []

            for coalition in batch_data:
                # Create masked image (only active superpixels)
                masked = background_image.copy()
                for j, val in enumerate(coalition):
                    if val == 1:
                        masked[:, segments == j] = image_np[:, segments == j]

                batch_images.append(torch.from_numpy(masked).float())

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

        # Fit weighted linear model to estimate Shapley values
        print(f"  Fitting weighted linear model for Shapley values...")
        model = Ridge(alpha=1.0)
        model.fit(coalition_data, predictions, sample_weight=coalition_weights)

        # Get Shapley values (coefficients represent feature importance)
        shapley_values = model.coef_

        # Map superpixel Shapley values back to pixel space
        importance_map = np.zeros((H, W), dtype=np.float32)
        for i, shap_value in enumerate(shapley_values):
            importance_map[segments == i] = shap_value

        raw_map = importance_map.copy()

        if normalize:
            importance_map = self._normalize_saliency(importance_map)

        metadata = {
            "target_class": target_class,
            "n_samples": n_samples,
            "n_segments": n_segments,
            "n_features": n_features,
            "segmentation_method": segmentation_method,
            "r2_score": float(model.score(coalition_data, predictions, sample_weight=coalition_weights)),
            "original_pred": float(original_pred),
            "background_pred": float(background_pred),
        }

        return SaliencyMap(
            map=importance_map, raw_map=raw_map, method="shap", metadata=metadata
        )
