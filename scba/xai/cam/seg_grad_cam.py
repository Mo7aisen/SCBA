"""
Seg-Grad-CAM implementation for segmentation models.

Based on:
Vinogradova et al., "Towards Interpretable Semantic Segmentation via Gradient-Weighted
Class Activation Mapping", AAAI 2020.
"""

import numpy as np
import torch
import torch.nn.functional as F

from scba.xai.common import ExplainerBase, FeatureExtractor, SaliencyMap


class SegGradCAM(ExplainerBase):
    """
    Seg-Grad-CAM for segmentation models.

    Computes pixel-wise class activation maps using gradients of the target class
    with respect to feature maps, weighted by spatial importance.
    """

    def __init__(self, model, device="cuda", target_layer=None):
        super().__init__(model, device)
        self.target_layer = target_layer or self._get_last_conv_layer()

    def _get_last_conv_layer(self):
        """Find the last convolutional layer in the model."""
        last_conv = None
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
                last_conv_name = name

        if last_conv is None:
            raise ValueError("No Conv2d layers found in model")

        print(f"Using layer: {last_conv_name}")
        return last_conv

    def explain(
        self,
        image: torch.Tensor,
        target_class: int = 1,
        target_mask: torch.Tensor = None,
        normalize: bool = True,
        **kwargs,
    ) -> SaliencyMap:
        """
        Generate Seg-Grad-CAM explanation.

        Args:
            image: (1, C, H, W) or (C, H, W) input image
            target_class: Class to explain (1 for foreground in binary segmentation)
            target_mask: Optional (H, W) mask to guide explanation
            normalize: Whether to normalize output to [0, 1]

        Returns:
            SaliencyMap object
        """
        image = self._prepare_image(image)
        image.requires_grad = True

        # Set up feature extractor
        feature_extractor = FeatureExtractor(self.model, self.target_layer)

        # Forward pass
        self.model.zero_grad()
        output = self.model(image)  # (1, C, H, W)

        # Get target scores
        if target_mask is not None:
            # Mask-guided: use mask to weight the target class scores
            target_mask = target_mask.to(self.device)
            if target_mask.ndim == 2:
                target_mask = target_mask.unsqueeze(0).unsqueeze(0)

            # Upsample mask to output size if needed
            if target_mask.shape[-2:] != output.shape[-2:]:
                target_mask = F.interpolate(
                    target_mask.float(),
                    size=output.shape[-2:],
                    mode="nearest",
                )

            # Weight target class by mask
            score = (output[:, target_class] * target_mask.squeeze()).sum()
        else:
            # Standard: sum of target class scores
            score = output[:, target_class].sum()

        # Backward pass
        score.backward()

        # Get gradients and features
        gradients = feature_extractor.gradients  # (1, C, H, W)
        features = feature_extractor.features  # (1, C, H, W)

        # Compute weights (global average pooling of gradients)
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of feature maps
        cam = (weights * features).sum(dim=1, keepdim=True)  # (1, 1, H, W)

        # ReLU to keep only positive contributions
        cam = F.relu(cam)

        # Upsample to input size
        cam = F.interpolate(
            cam,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # Convert to numpy
        cam_np = cam.squeeze().cpu().detach().numpy()

        # Store raw cam
        raw_cam = cam_np.copy()

        # Normalize if requested
        if normalize:
            cam_np = self._normalize_saliency(cam_np)

        metadata = {
            "target_class": target_class,
            "target_layer": str(self.target_layer),
            "has_target_mask": target_mask is not None,
        }

        return SaliencyMap(
            map=cam_np,
            raw_map=raw_cam,
            method="seg_grad_cam",
            metadata=metadata,
        )


class SegXResCAM(ExplainerBase):
    """
    Seg-XRes-CAM: Spatially weighted variant of Seg-Grad-CAM.

    Better for region-within-mask explanation.
    """

    def __init__(self, model, device="cuda", target_layer=None):
        super().__init__(model, device)
        self.target_layer = target_layer or self._get_last_conv_layer()

    def _get_last_conv_layer(self):
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        return last_conv

    def explain(
        self,
        image: torch.Tensor,
        target_class: int = 1,
        target_mask: torch.Tensor = None,
        normalize: bool = True,
        **kwargs,
    ) -> SaliencyMap:
        """Generate Seg-XRes-CAM explanation."""
        image = self._prepare_image(image)
        image.requires_grad = True

        feature_extractor = FeatureExtractor(self.model, self.target_layer)

        self.model.zero_grad()
        output = self.model(image)

        if target_mask is not None:
            target_mask = target_mask.to(self.device)
            if target_mask.ndim == 2:
                target_mask = target_mask.unsqueeze(0).unsqueeze(0)
            if target_mask.shape[-2:] != output.shape[-2:]:
                target_mask = F.interpolate(
                    target_mask.float(), size=output.shape[-2:], mode="nearest"
                )
            score = (output[:, target_class] * target_mask.squeeze()).sum()
        else:
            score = output[:, target_class].sum()

        score.backward()

        gradients = feature_extractor.gradients
        features = feature_extractor.features

        # Spatial weighting: use gradient magnitude at each location
        weights = gradients  # (1, C, H, W) - preserve spatial dimension

        # Weighted combination
        cam = (weights * features).sum(dim=1, keepdim=True)

        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)

        cam_np = cam.squeeze().cpu().detach().numpy()
        raw_cam = cam_np.copy()

        if normalize:
            cam_np = self._normalize_saliency(cam_np)

        metadata = {
            "target_class": target_class,
            "target_layer": str(self.target_layer),
            "spatial_weighting": True,
        }

        return SaliencyMap(
            map=cam_np, raw_map=raw_cam, method="seg_xres_cam", metadata=metadata
        )
