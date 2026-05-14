"""
Common utilities and base classes for XAI methods.

Provides unified API for all explanation methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import inspect
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
        # Handle case when CUDA is not available
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
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
        return image.to(self.device).clone().detach()


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

def get_last_conv_layer_with_name(
    model: nn.Module,
    *,
    min_kernel: int = 2,
    exclude_names: Optional[Sequence[str]] = None,
) -> Tuple[str, nn.Conv2d]:
    """Return the last Conv2d layer, skipping 1x1 heads by default."""
    last_name = None
    last_conv = None
    exclude_names = exclude_names or ("outc",)

    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if min_kernel and (
            module.kernel_size[0] < min_kernel or module.kernel_size[1] < min_kernel
        ):
            continue
        if exclude_names and any(name.startswith(prefix) for prefix in exclude_names):
            continue
        last_name = name
        last_conv = module

    if last_conv is None or last_name is None:
        raise ValueError("No suitable Conv2d layers found in model")

    return last_name, last_conv


def get_last_conv_layer(
    model: nn.Module,
    *,
    min_kernel: int = 2,
    exclude_names: Optional[Sequence[str]] = None,
) -> nn.Conv2d:
    """Return the last Conv2d layer with kernel size >= min_kernel."""
    _, layer = get_last_conv_layer_with_name(
        model, min_kernel=min_kernel, exclude_names=exclude_names
    )
    return layer


def get_decoder_target_layers(
    model: nn.Module,
    *,
    n_layers: int = 2,
    min_kernel: int = 2,
) -> List[Tuple[str, nn.Conv2d]]:
    """Return decoder Conv2d layers from UNet-style up blocks (latest n_layers)."""
    candidates: Dict[str, Tuple[str, nn.Conv2d]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if min_kernel and (
            module.kernel_size[0] < min_kernel or module.kernel_size[1] < min_kernel
        ):
            continue
        if not name.startswith("up"):
            continue
        if ".conv." not in name:
            continue
        block = name.split(".")[0]
        candidates[block] = (name, module)

    if not candidates:
        return [get_last_conv_layer_with_name(model, min_kernel=min_kernel)]

    ordered_blocks = sorted(candidates.keys())
    selected = [candidates[block] for block in ordered_blocks[-n_layers:]]
    return selected


def resolve_target_layers(
    model: nn.Module,
    target_layers: Optional[Iterable[object]],
    *,
    default_layers: Optional[List[Tuple[str, nn.Conv2d]]] = None,
) -> List[Tuple[str, nn.Conv2d]]:
    """Normalize target layer specification to a list of (name, module)."""
    if target_layers is None:
        return default_layers or [get_last_conv_layer_with_name(model)]

    resolved: List[Tuple[str, nn.Conv2d]] = []
    for layer in target_layers:
        if isinstance(layer, str):
            found = None
            for name, module in model.named_modules():
                if name == layer:
                    if not isinstance(module, nn.Conv2d):
                        raise ValueError(f"Layer {layer} is not Conv2d.")
                    found = (name, module)
                    break
            if found is None:
                raise ValueError(f"Layer {layer} not found in model.")
            resolved.append(found)
        elif isinstance(layer, nn.Conv2d):
            name = None
            for module_name, module in model.named_modules():
                if module is layer:
                    name = module_name
                    break
            resolved.append((name or layer.__class__.__name__, layer))
        else:
            raise TypeError("target_layers must be Conv2d modules or layer name strings.")

    return resolved


def get_target_layer(
    model: nn.Module,
    layer_name: Optional[str] = None,
    *,
    min_kernel: int = 2,
    exclude_names: Optional[Sequence[str]] = None,
) -> nn.Module:
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

    return get_last_conv_layer(
        model, min_kernel=min_kernel, exclude_names=exclude_names
    )


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

    def close(self) -> None:
        """Remove hooks to avoid dangling references."""
        self.forward_hook.remove()
        self.backward_hook.remove()

    def __del__(self):
        self.close()


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
        - layer_cam: LayerCAM
        - multi_layer_cam: Multi-layer LayerCAM (decoder aggregation)
        - guided_grad_cam: Guided Grad-CAM
        - integrated_gradients: Integrated Gradients
        - gradient: Plain input gradients (baseline)
        - input_x_gradient: Input×Gradient (baseline)
        - random_map: Random map (negative control)
        - lrp: Layer-wise Relevance Propagation
        - rise: RISE
        - lime: LIME
        - shap: SHAP
        - occlusion: Occlusion
    """
    # Import here to avoid circular dependencies
    from scba.xai.cam.seg_grad_cam import SegGradCAM, SegXResCAM
    from scba.xai.cam.hires_cam import HiResCAM
    from scba.xai.cam.grad_cam_pp import GradCAMPlusPlus
    from scba.xai.cam.layer_cam import LayerCAM, MultiLayerCAM
    from scba.xai.perturb.rise import RISE
    from scba.xai.perturb.occlusion import Occlusion
    from scba.xai.perturb.lime_seg import LIMESegmentation
    from scba.xai.perturb.shap_seg import KernelSHAPSegmentation
    from scba.xai.gradient.integrated_gradients import IntegratedGradients
    from scba.xai.gradient.plain_gradients import GradientSaliency
    from scba.xai.baselines.random_map import RandomMap

    method_map = {
        # Gradient-based CAM methods
        "seg_grad_cam": SegGradCAM,
        "seg_xres_cam": SegXResCAM,
        "hires_cam": HiResCAM,
        "grad_cam_pp": GradCAMPlusPlus,
        "layer_cam": LayerCAM,
        "multi_layer_cam": MultiLayerCAM,
        # Gradient-based attribution
        "integrated_gradients": IntegratedGradients,
        "gradient": GradientSaliency,
        "input_x_gradient": GradientSaliency,
        # Negative controls / baselines
        "random_map": RandomMap,
        # Perturbation-based methods
        "rise": RISE,
        "occlusion": Occlusion,
        "lime": LIMESegmentation,
        "shap": KernelSHAPSegmentation,
    }

    if method not in method_map:
        raise ValueError(
            f"Unknown method: {method}. Available: {list(method_map.keys())}"
        )

    if target_class is None:
        # Scientific justification: SCBA uses binary segmentation with logits for
        # background (0) and foreground lung (1). Defaulting to the foreground
        # class makes the XAI factory behavior match the experimental protocol
        # and prevents silent "background explanation" artifacts.
        target_class = 1

    explainer_class = method_map[method]
    init_kwargs = {}
    init_params = inspect.signature(explainer_class.__init__).parameters
    for key in ("target_layer", "target_layers", "aggregate", "base_cam"):
        if key in kwargs and key in init_params:
            init_kwargs[key] = kwargs.pop(key)
    explainer = explainer_class(model, device=device, **init_kwargs)

    if method == "input_x_gradient":
        kwargs = {**kwargs, "use_input_x_gradient": True}

    return explainer.explain(image, target_class=target_class, target_mask=target_mask, **kwargs)
