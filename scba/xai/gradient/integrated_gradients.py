"""
Integrated Gradients for segmentation tasks.

Based on Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017.
Computes path integral of gradients from baseline to input.
"""

import os
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
        seed: int = 42,
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
            # Scientific justification (M7): baselines must be deterministic for
            # reproducible attributions and completeness diagnostics.
            gen = torch.Generator(device=self.device).manual_seed(int(seed))
            baseline_tensor = torch.randn_like(image, generator=gen) * 0.1
        elif baseline == "blur":
            # Blurred version of input
            # Scientific justification: preserve spatial resolution so that the
            # interpolation path is well-defined and comparisons are consistent.
            baseline_tensor = F.avg_pool2d(image, kernel_size=63, stride=1, padding=31)
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

        disable_tqdm = os.getenv("SCBA_DISABLE_TQDM", "").lower() in {"1", "true", "yes"}
        for i in tqdm(
            range(0, len(interpolated_images), batch_size),
            desc="IntGrad",
            disable=disable_tqdm,
        ):
            batch_images = interpolated_images[i:i + batch_size]
            batch_tensor = torch.cat(batch_images, dim=0)  # (B, C, H, W)
            batch_tensor.requires_grad = True

            # Forward pass
            self.model.zero_grad(set_to_none=True)
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

            # Scientific justification (M7): compute a per-sample scalar score
            # and sum across the batch so each interpolated sample receives its
            # own gradient. Using a single `.mean()` over (B,H,W) scales
            # gradients by 1/B and makes IG depend on the chosen batch_size.
            loss = target_outputs.mean(dim=(1, 2)).sum()

            # Backward pass
            loss.backward()

            # Get gradients
            gradients = batch_tensor.grad  # (B, C, H, W)
            all_gradients.append(gradients.detach().cpu())

            # Clear GPU memory
            del batch_tensor, outputs, target_outputs, loss
            if str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()

        # Concatenate all gradients
        all_gradients = torch.cat(all_gradients, dim=0)  # (n_steps+1, C, H, W)

        # Approximate integral using trapezoidal rule
        print(f"  Computing path integral...")
        # Scientific justification (M7): use the trapezoidal rule over the
        # discretized path (Sundararajan et al.). Using a simple mean over
        # endpoints is not the trapezoidal rule.
        avg_gradients = (all_gradients[:-1] + all_gradients[1:]) / 2.0  # (n_steps, C, H, W)
        path_gradients = avg_gradients.mean(dim=0)  # (C, H, W) = sum/n_steps

        # Multiply by (input - baseline)
        # Scientific justification (M7): IG is defined per-input dimension; we
        # operate on (C, H, W) (no batch dimension) to produce a 2D saliency map.
        diff = (image - baseline_tensor).squeeze(0).cpu()  # (C, H, W)
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

            # Scientific justification (M7): completeness must be defined on the
            # same scalar function used for gradients. Here gradients are taken
            # w.r.t. the mean target-class logit, so we compare logit means.
            baseline_pred = baseline_output[:, target_class].mean().item()
            input_pred = input_output[:, target_class].mean().item()

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
