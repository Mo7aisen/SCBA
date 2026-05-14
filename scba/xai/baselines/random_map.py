"""
Random-map negative control for saliency pipelines.

This is not an explanation method. It is a deterministic control used to verify
that SCBA metrics do not produce "meaningful" changes under attribution maps
that contain no model information.
"""

from __future__ import annotations

import torch

from scba.xai.common import ExplainerBase, SaliencyMap


class RandomMap(ExplainerBase):
    """Generate a random saliency map (negative control)."""

    def __init__(self, model, device: str = "cuda"):
        super().__init__(model, device)

    def explain(
        self,
        image: torch.Tensor,
        target_class: int = 1,
        seed: int = 42,
        distribution: str = "uniform",
        normalize: bool = True,
        **kwargs,
    ) -> SaliencyMap:
        image = self._prepare_image(image)
        _, _, H, W = image.shape

        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        if distribution == "uniform":
            saliency = torch.rand((H, W), generator=gen).numpy()
        elif distribution == "gaussian":
            saliency = torch.randn((H, W), generator=gen).abs().numpy()
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        raw_map = saliency.copy()
        if normalize:
            saliency = self._normalize_saliency(saliency)

        return SaliencyMap(
            map=saliency,
            raw_map=raw_map,
            method="random_map",
            metadata={
                "target_class": int(target_class),
                "seed": int(seed),
                "distribution": str(distribution),
            },
        )

