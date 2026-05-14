import torch

from scba.models import get_model


def test_get_model_unet_builds_and_outputs_two_logits() -> None:
    # Scientific justification: SCBA segmentation is binary with two logits
    # (background vs lung). A broken model factory silently produces invalid
    # architectures and invalidates all reported metrics.
    model = get_model("unet", in_channels=1, out_channels=2)
    model.eval()

    x = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    with torch.no_grad():
        y = model(x)

    assert tuple(y.shape) == (1, 2, 32, 32)

