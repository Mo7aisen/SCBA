"""
Plain gradient saliency baselines for segmentation.

These baselines are intentionally simple and fast:
- gradient: |d score / d input|
- input_x_gradient: |input * d score / d input|
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from scba.xai.common import ExplainerBase, SaliencyMap


class GradientSaliency(ExplainerBase):
    """Plain gradient saliency for segmentation logits."""

    def __init__(self, model, device: str = "cuda"):
        super().__init__(model, device)

    def explain(
        self,
        image: torch.Tensor,
        target_class: int = 1,
        target_mask: torch.Tensor | None = None,
        normalize: bool = True,
        use_input_x_gradient: bool = False,
        **kwargs,
    ) -> SaliencyMap:
        image = self._prepare_image(image).to(self.device)
        image = image.requires_grad_(True)
        _, C, H, W = image.shape

        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)  # (1, n_classes, H, W)
        score_map = logits[:, target_class]  # (1, H, W)

        if target_mask is not None:
            target_mask_resized = F.interpolate(
                target_mask.unsqueeze(0).float(),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).to(self.device)
            score_map = score_map * target_mask_resized.squeeze(0)

        # Scalar per-sample score; sum keeps gradients independent of any batch scaling.
        score = score_map.mean(dim=(1, 2)).sum()
        score.backward()

        grads = image.grad.detach()  # (1, C, H, W)
        if use_input_x_gradient:
            grads = grads * image.detach()

        # Aggregate channels
        if C > 1:
            saliency = grads.abs().sum(dim=1).squeeze(0).detach().cpu().numpy()
        else:
            saliency = grads.abs().squeeze(0).squeeze(0).detach().cpu().numpy()

        raw_map = saliency.copy()
        if normalize:
            saliency = self._normalize_saliency(saliency)

        method_name = "input_x_gradient" if use_input_x_gradient else "gradient"
        return SaliencyMap(
            map=saliency,
            raw_map=raw_map,
            method=method_name,
            metadata={
                "target_class": int(target_class),
                "use_input_x_gradient": bool(use_input_x_gradient),
            },
        )

