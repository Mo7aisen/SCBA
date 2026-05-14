import numpy as np
import torch

from scba.models.unet import UNet
from scba.xai.common import explain


def test_explain_defaults_to_foreground_target_class_for_segmentation() -> None:
    # Scientific justification: SCBA explanations target the lung foreground
    # class (1). Defaulting to background or None changes the phenomenon being
    # measured and can create spurious “consistency” artifacts.
    torch.manual_seed(42)

    model = UNet(n_channels=1, n_classes=2, bilinear=True)
    model.eval()

    image = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    sal = explain(image, model, method="seg_grad_cam", device="cpu")

    assert sal.map.shape == (32, 32)
    assert np.isfinite(sal.map).all()
    assert 0.0 <= float(sal.map.min()) <= float(sal.map.max()) <= 1.0
    assert sal.metadata is not None
    assert sal.metadata.get("target_class") == 1

