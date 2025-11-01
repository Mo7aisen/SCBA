"""
HiResCAM implementation - high-resolution class activation mapping.

Based on:
Draelos et al., "Machine-learning-based multiple abnormality prediction with
large-scale chest computed tomography volumes", Medical Image Analysis, 2020.
"""

import numpy as np
import torch
import torch.nn.functional as F

from scba.xai.common import ExplainerBase, FeatureExtractor, SaliencyMap


class HiResCAM(ExplainerBase):
    """
    HiResCAM: Element-wise product of gradients and activations.

    Produces higher resolution saliency maps compared to Grad-CAM.
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
        """
        Generate HiResCAM explanation.

        HiResCAM uses element-wise multiplication instead of channel-wise
        weighting, preserving spatial resolution.
        """
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

        gradients = feature_extractor.gradients  # (1, C, H, W)
        features = feature_extractor.features  # (1, C, H, W)

        # Element-wise product (this is the key difference from Grad-CAM)
        cam = (gradients * features).sum(dim=1, keepdim=True)  # (1, 1, H, W)

        # ReLU
        cam = F.relu(cam)

        # Upsample to input size
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)

        cam_np = cam.squeeze().cpu().detach().numpy()
        raw_cam = cam_np.copy()

        if normalize:
            cam_np = self._normalize_saliency(cam_np)

        metadata = {
            "target_class": target_class,
            "target_layer": str(self.target_layer),
            "high_resolution": True,
        }

        return SaliencyMap(map=cam_np, raw_map=raw_cam, method="hires_cam", metadata=metadata)
