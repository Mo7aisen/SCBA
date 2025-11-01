"""
Common utilities and base classes for XAI methods.

Provides unified API for all explanation methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SaliencyMap:
    """
    Container for saliency/attribution map.

    Attributes:
        map: (H, W) numpy array, normalized to [0, 1]
        raw_map: (H, W) numpy array, unnormalized scores
        method: Name of XAI method used
        metadata: Additional method-specific information
    """

    map: np.ndarray
    raw_map: np.ndarray
    method: str
    metadata: Optional[Dict] = None

    def __post_init__(self):
        assert self.map.ndim == 2, "Saliency map must be 2D"
        assert 0 <= self.map.min() and self.map.max() <= 1, "Map must be normalized to [0, 1]"


class ExplainerBase(ABC):
    """
    Base class for all XAI explainers.

    Enforces common interface and provides utility methods.
    """

    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    @abstractmethod
    def explain(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None,
        target_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SaliencyMap:
        """
        Generate explanation for an image.

        Args:
            image: (1, C, H, W) or (C, H, W) input image
            target_class: Class to explain (for classification). For segmentation, use 1 (foreground)
            target_mask: Optional ground truth mask for guided explanation
            **kwargs: Method-specific parameters

        Returns:
            SaliencyMap object
        """
        pass

    def _normalize_saliency(self, saliency: np.ndarray) -> np.ndarray:
        """Normalize saliency map to [0, 1]."""
        if saliency.max() == saliency.min():
            return np.zeros_like(saliency)

        normalized = (saliency - saliency.min()) / (saliency.max() - saliency.min())
        return normalized.astype(np.float32)

    def _prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        """Ensure image has batch dimension."""
        if image.ndim == 3:
            image = image.unsqueeze(0)
        return image.to(self.device)


def normalize_saliency_map(saliency: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize saliency map.

    Args:
        saliency: (H, W) or (B, H, W) saliency map
        method: 'minmax', 'absmax', or 'percentile'

    Returns:
        Normalized saliency map in [0, 1]
    """
    if method == "minmax":
        s_min, s_max = saliency.min(), saliency.max()
        if s_max == s_min:
            return np.zeros_like(saliency)
        return (saliency - s_min) / (s_max - s_min)

    elif method == "absmax":
        s_max = np.abs(saliency).max()
        if s_max == 0:
            return np.zeros_like(saliency)
        normalized = saliency / s_max
        return (normalized + 1) / 2  # Shift to [0, 1]

    elif method == "percentile":
        p_low, p_high = np.percentile(saliency, [2, 98])
        saliency_clipped = np.clip(saliency, p_low, p_high)
        return normalize_saliency_map(saliency_clipped, method="minmax")

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def get_target_layer(model: nn.Module, layer_name: Optional[str] = None) -> nn.Module:
    """
    Get target layer for CAM-based methods.

    Args:
        model: PyTorch model
        layer_name: Name of layer. If None, returns last conv layer

    Returns:
        Target layer module
    """
    if layer_name is not None:
        # Get specific layer by name
        for name, module in model.named_modules():
            if name == layer_name:
                return module
        raise ValueError(f"Layer {layer_name} not found in model")

    # Find last convolutional layer
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module

    if last_conv is None:
        raise ValueError("No convolutional layers found in model")

    return last_conv


class FeatureExtractor:
    """
    Hook-based feature extractor for intermediate layers.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.features = None
        self.gradients = None

        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self._forward_hook_fn)
        self.backward_hook = target_layer.register_full_backward_hook(self._backward_hook_fn)

    def _forward_hook_fn(self, module, input, output):
        self.features = output.detach()

    def _backward_hook_fn(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __del__(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def explain(
    image: torch.Tensor,
    model: nn.Module,
    target_mask: Optional[torch.Tensor] = None,
    target_class: Optional[int] = None,
    method: str = "seg_grad_cam",
    device: str = "cuda",
    **kwargs,
) -> SaliencyMap:
    """
    Unified explain function - factory for all XAI methods.

    Args:
        image: (1, C, H, W) or (C, H, W) input
        model: Segmentation model
        target_mask: Optional GT mask for guided explanation
        target_class: Class to explain (default: 1 for foreground)
        method: XAI method name
        device: 'cuda' or 'cpu'
        **kwargs: Method-specific arguments

    Returns:
        SaliencyMap object

    Supported methods:
        - seg_grad_cam: Seg-Grad-CAM
        - seg_xres_cam: Seg-XRes-CAM
        - hires_cam: HiResCAM
        - grad_cam_pp: Grad-CAM++
        - guided_grad_cam: Guided Grad-CAM
        - integrated_gradients: Integrated Gradients
        - lrp: Layer-wise Relevance Propagation
        - rise: RISE
        - lime: LIME
        - shap: SHAP
        - occlusion: Occlusion
    """
    # Import here to avoid circular dependencies
    from scba.xai.cam.seg_grad_cam import SegGradCAM

    method_map = {
        "seg_grad_cam": SegGradCAM,
        # More methods will be added as we implement them
    }

    if method not in method_map:
        raise ValueError(
            f"Unknown method: {method}. Available: {list(method_map.keys())}"
        )

    explainer_class = method_map[method]
    explainer = explainer_class(model, device=device)

    return explainer.explain(
        image, target_class=target_class, target_mask=target_mask, **kwargs
    )
