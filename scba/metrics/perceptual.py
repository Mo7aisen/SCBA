"""
Perceptual distance metrics for radiologist-proxy realism checks.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import certifi

try:
    from torchvision.models import resnet18
    from torchvision.models import ResNet18_Weights
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("torchvision is required for perceptual distance metrics") from exc


class PerceptualDistance:
    """Compute perceptual distance using a pretrained ResNet backbone."""

    def __init__(
        self,
        device: str = "cuda",
        *,
        layers: Optional[Iterable[str]] = None,
        input_size: int = 224,
    ) -> None:
        self.device = torch.device(device)
        self.input_size = input_size
        self.layers = list(layers) if layers is not None else ["layer2", "layer3", "layer4"]

        # Scientific justification (M5): use a standard CA bundle for HTTPS
        # downloads instead of disabling certificate verification. Some sandboxed
        # environments ship without system CAs, so we point urllib/torch hub to
        # certifi's curated bundle.
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

        # Scientific justification (M5): rely on torchvision's built-in weight
        # loading (TLS verified, cached under TORCH_HOME) rather than custom
        # SSL-bypassing download logic.
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.eval()
        self.model.to(self.device)

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def _prepare(self, image: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            tensor = torch.from_numpy(image.astype(np.float32))
        else:
            tensor = image.float()

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1)
        if tensor.max() > 1.0:
            tensor = tensor / 255.0

        tensor = tensor.to(self.device)
        if tensor.shape[-1] != self.input_size or tensor.shape[-2] != self.input_size:
            tensor = F.interpolate(tensor, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)

        tensor = (tensor - self.mean) / self.std
        return tensor

    def _extract_features(self, image: torch.Tensor) -> List[torch.Tensor]:
        x = self.model.conv1(image)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        features = []
        x = self.model.layer1(x)
        if "layer1" in self.layers:
            features.append(x)
        x = self.model.layer2(x)
        if "layer2" in self.layers:
            features.append(x)
        x = self.model.layer3(x)
        if "layer3" in self.layers:
            features.append(x)
        x = self.model.layer4(x)
        if "layer4" in self.layers:
            features.append(x)

        return features

    def compute(self, image_a: np.ndarray | torch.Tensor, image_b: np.ndarray | torch.Tensor) -> float:
        """Return mean feature MSE across selected layers."""
        with torch.no_grad():
            tensor_a = self._prepare(image_a)
            tensor_b = self._prepare(image_b)

            feats_a = self._extract_features(tensor_a)
            feats_b = self._extract_features(tensor_b)

            if not feats_a or not feats_b:
                raise ValueError("No features extracted for perceptual distance.")

            distances = [F.mse_loss(fa, fb).item() for fa, fb in zip(feats_a, feats_b)]
            return float(np.mean(distances))
