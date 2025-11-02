"""
Integrated Gradients for segmentation tasks.

Based on Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017.
Computes path integral of gradients from baseline to input.
"""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from scba.xai.common import ExplainerBase, SaliencyMap


class IntegratedGradients(ExplainerBase):
    """
    Integrated Gradients for segmentation.

    Computes the path integral of gradients along a straight line
    from a baseline (e.g., black image) to the input image.
    """

    def __init__(self, model, device="cuda"):
        super().__init__(model, device)

    def explain(
        self,
        image: torch.Tensor,
        target_class: int = 1,
        target_mask: torch.Tensor = None,
        n_steps: int = 50,
        baseline: str = "black",
        batch_size: int = 8,
        normalize: bool = True,
        **kwargs,
    ) -> SaliencyMap:
        """
        Generate Integrated Gradients explanation.

        Args:
            image: (1, C, H, W) or (C, H, W) input
            target_class: Class to explain
            target_mask: Optional target mask for focused attribution
            n_steps: Number of integration steps
            baseline: "black" (zeros) or "gaussian" (noise)
            batch_size: Batch size for gradient computation
            normalize: Whether to normalize output

        Returns:
            SaliencyMap object
        """
        image = self._prepare_image(image)
        _, C, H, W = image.shape

        # Create baseline
        if baseline == "black":
            baseline_tensor = torch.zeros_like(image)
        elif baseline == "gaussian":
            baseline_tensor = torch.randn_like(image) * 0.1
        elif baseline == "blur":
            # Blurred version of input
            baseline_tensor = F.avg_pool2d(image, kernel_size=64, stride=1, padding=32)
        else:
            raise ValueError(f"Unknown baseline type: {baseline}")

        baseline_tensor = baseline_tensor.to(self.device)

        # Generate interpolated images along path
        alphas = torch.linspace(0, 1, n_steps + 1).to(self.device)
        interpolated_images = []

        for alpha in alphas:
            interpolated = baseline_tensor + alpha * (image - baseline_tensor)
            interpolated_images.append(interpolated)

        # Compute gradients for each interpolated image in batches
        print(f"  Computing gradients for {n_steps + 1} interpolated images...")

        # Adjust batch size for large images
        if H * W > 512 * 512:
            batch_size = min(batch_size, 4)
            print(f"  Reduced batch_size to {batch_size} for {H}x{W} image")

        all_gradients = []

        for i in tqdm(range(0, len(interpolated_images), batch_size), desc="IntGrad"):
            batch_images = interpolated_images[i:i + batch_size]
            batch_tensor = torch.cat(batch_images, dim=0)  # (B, C, H, W)
            batch_tensor.requires_grad = True

            # Forward pass
            outputs = self.model(batch_tensor)  # (B, n_classes, H, W)

            # Get target class outputs
            target_outputs = outputs[:, target_class]  # (B, H, W)

            # If target mask provided, focus on specific region
            if target_mask is not None:
                target_mask_resized = F.interpolate(
                    target_mask.unsqueeze(0).float(),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False
                ).to(self.device)
                target_outputs = target_outputs * target_mask_resized.squeeze(0)

            # Compute loss (mean of target class predictions)
            loss = target_outputs.mean()

            # Backward pass
            loss.backward()

            # Get gradients
            gradients = batch_tensor.grad  # (B, C, H, W)
            all_gradients.append(gradients.detach().cpu())

            # Clear GPU memory
            del batch_tensor, outputs, target_outputs, loss
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # Concatenate all gradients
        all_gradients = torch.cat(all_gradients, dim=0)  # (n_steps+1, C, H, W)

        # Approximate integral using trapezoidal rule
        print(f"  Computing path integral...")
        path_gradients = all_gradients.mean(dim=0)  # Average over steps (C, H, W)

        # Multiply by (input - baseline)
        diff = (image - baseline_tensor).cpu()
        integrated_gradients = path_gradients * diff  # (C, H, W)

        # Aggregate across channels (for multi-channel inputs)
        if C > 1:
            attribution = integrated_gradients.abs().sum(dim=0).numpy()  # (H, W)
        else:
            attribution = integrated_gradients.squeeze(0).numpy()  # (H, W)

        raw_map = attribution.copy()

        if normalize:
            attribution = self._normalize_saliency(attribution)

        # Compute completeness (how well IG approximates the difference)
        with torch.no_grad():
            baseline_output = self.model(baseline_tensor)
            input_output = self.model(image)

            baseline_pred = torch.softmax(baseline_output, dim=1)[:, target_class].mean().item()
            input_pred = torch.softmax(input_output, dim=1)[:, target_class].mean().item()

            # Sum of attributions should approximately equal the prediction difference
            completeness_error = abs(integrated_gradients.sum().item() - (input_pred - baseline_pred))

        metadata = {
            "target_class": target_class,
            "n_steps": n_steps,
            "baseline": baseline,
            "baseline_pred": float(baseline_pred),
            "input_pred": float(input_pred),
            "completeness_error": float(completeness_error),
        }

        return SaliencyMap(
            map=attribution, raw_map=raw_map, method="integrated_gradients", metadata=metadata
        )
