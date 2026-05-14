import torch

from scba.xai.gradient.integrated_gradients import IntegratedGradients


class _LinearSegModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, kernel_size=1, bias=False)
        with torch.no_grad():
            self.conv.weight.zero_()
            # target_class=1 logit = 2 * x (per-pixel)
            self.conv.weight[1, 0, 0, 0] = 2.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def test_integrated_gradients_completeness_uses_logit_scalar() -> None:
    # Scientific justification (M7): completeness must compare the same scalar
    # function used for gradients. For this linear model, IG should satisfy
    # completeness nearly exactly when defined on mean logits.
    model = _LinearSegModel().eval()
    ig = IntegratedGradients(model, device="cpu")

    image = torch.full((1, 1, 4, 4), 0.5, dtype=torch.float32)
    sal = ig.explain(
        image,
        target_class=1,
        n_steps=10,
        baseline="black",
        batch_size=8,
        normalize=False,
    )

    assert sal.metadata is not None
    assert float(sal.metadata["completeness_error"]) < 1e-4

