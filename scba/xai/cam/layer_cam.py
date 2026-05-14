"""
LayerCAM implementations for segmentation models.

Provides single-layer LayerCAM and a multi-layer aggregation wrapper
that targets decoder conv blocks by default.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from scba.xai.common import (
    ExplainerBase,
    FeatureExtractor,
    SaliencyMap,
    get_decoder_target_layers,
    get_last_conv_layer_with_name,
    resolve_target_layers,
)


class LayerCAM(ExplainerBase):
    """
    LayerCAM for segmentation models.

    Computes pixel-wise attributions using positive gradients per spatial location,
    preserving fine-grained structure compared with global pooling CAMs.
    """

    def __init__(self, model, device: str = "cuda", target_layer: Optional[torch.nn.Module] = None):
        super().__init__(model, device)
        if target_layer is None:
            _, target_layer = get_last_conv_layer_with_name(model)
        self.target_layer = target_layer

    def explain(
        self,
        image: torch.Tensor,
        target_class: int = 1,
        target_mask: torch.Tensor | None = None,
        normalize: bool = True,
        **kwargs,
    ) -> SaliencyMap:
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
        feature_extractor.close()

        weights = F.relu(gradients)
        cam = (weights * features).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)

        cam_np = cam.squeeze().detach().cpu().numpy()
        raw_cam = cam_np.copy()

        if normalize:
            cam_np = self._normalize_saliency(cam_np)

        metadata = {
            "target_class": target_class,
            "target_layer": str(self.target_layer),
            "method": "layer_cam",
        }

        return SaliencyMap(map=cam_np, raw_map=raw_cam, method="layer_cam", metadata=metadata)


class MultiLayerCAM(ExplainerBase):
    """
    Aggregate LayerCAM maps across multiple decoder layers.

    Defaults to the last two decoder blocks (up3, up4) in UNet-like models.
    """

    def __init__(
        self,
        model,
        device: str = "cuda",
        target_layers: Optional[Sequence[object]] = None,
        aggregate: str = "mean",
    ):
        super().__init__(model, device)
        default_layers = get_decoder_target_layers(model, n_layers=2)
        self.target_layers = resolve_target_layers(model, target_layers, default_layers=default_layers)
        self.aggregate = aggregate

    def _aggregate(self, cams: List[np.ndarray]) -> np.ndarray:
        if not cams:
            raise ValueError("No CAM maps to aggregate.")
        if self.aggregate == "mean":
            return np.mean(cams, axis=0)
        if self.aggregate == "max":
            return np.max(cams, axis=0)
        raise ValueError(f"Unknown aggregate mode: {self.aggregate}")

    def explain(
        self,
        image: torch.Tensor,
        target_class: int = 1,
        target_mask: torch.Tensor | None = None,
        normalize: bool = True,
        **kwargs,
    ) -> SaliencyMap:
        cams = []
        layer_names = []

        for name, layer in self.target_layers:
            layer_names.append(name)
            explainer = LayerCAM(self.model, device=self.device, target_layer=layer)
            cam = explainer.explain(
                image,
                target_class=target_class,
                target_mask=target_mask,
                normalize=True,
            )
            cams.append(cam.raw_map)

        aggregated = self._aggregate(cams)
        raw_map = aggregated.copy()
        if normalize:
            aggregated = self._normalize_saliency(aggregated)

        metadata = {
            "target_class": target_class,
            "target_layers": layer_names,
            "aggregate": self.aggregate,
            "method": "multi_layer_cam",
        }

        return SaliencyMap(
            map=aggregated, raw_map=raw_map, method="multi_layer_cam", metadata=metadata
        )
