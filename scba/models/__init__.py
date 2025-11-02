"""Segmentation model architectures."""

from scba.models.unet import UNet, get_unet


def get_model(arch: str, in_channels: int = 1, out_channels: int = 1, **kwargs):
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
        return get_unet(in_channels=in_channels, out_channels=out_channels, **kwargs)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


__all__ = ["UNet", "get_unet", "get_model"]
