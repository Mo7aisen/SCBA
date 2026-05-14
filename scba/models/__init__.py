"""Segmentation model architectures."""

from scba.models.unet import UNet, get_unet


def get_model(arch: str, in_channels: int = 1, out_channels: int = 2, **kwargs):
    """
    Get model by architecture name.

    Args:
        arch: Architecture name ('unet', etc.)
        in_channels: Number of input channels
        out_channels: Number of output channels
        **kwargs: Additional model-specific arguments

    Returns:
        Model instance
    """
    if arch.lower() == "unet":
        # Scientific justification: SCBA uses a 2-logit segmentation head
        # (background vs lung). `get_unet` is defined with `n_channels` and
        # `n_classes`; passing mismatched kwarg names silently breaks model
        # construction and invalidates all downstream experiments.
        return get_unet(n_channels=in_channels, n_classes=out_channels, **kwargs)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


__all__ = ["UNet", "get_unet", "get_model"]
