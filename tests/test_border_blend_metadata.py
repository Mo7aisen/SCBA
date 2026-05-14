import numpy as np

from scba.cf.borders import apply_border_edit


def _simple_circle_mask(h: int, w: int, r: int) -> np.ndarray:
    yy, xx = np.mgrid[:h, :w]
    cy, cx = h // 2, w // 2
    return (((yy - cy) ** 2 + (xx - cx) ** 2) <= (r**2)).astype(np.uint8)


def test_apply_border_edit_returns_blend_metadata_with_alpha() -> None:
    # Scientific justification (M3): the pipeline must record whether Poisson
    # blending succeeded or fell back, to quantify realism failure rates.
    image = np.zeros((64, 64), dtype=np.float32)
    mask = _simple_circle_mask(64, 64, r=18)

    out = apply_border_edit(
        image,
        mask,
        radius_px=2,
        operation="dilate",
        warp_mode="none",
        blend_method="alpha",
        return_metadata=True,
    )
    image_cf, mask_cf, roi, meta = out

    assert image_cf.shape == image.shape
    assert mask_cf.shape == mask.shape
    assert roi.shape == mask.shape
    assert "blend" in meta
    assert meta["blend"]["requested"] == "alpha"
    assert meta["blend"]["used"] == "alpha"


def test_apply_border_edit_returns_blend_metadata_with_poisson() -> None:
    image = np.zeros((64, 64), dtype=np.float32)
    mask = _simple_circle_mask(64, 64, r=18)

    image_cf, mask_cf, roi, meta = apply_border_edit(
        image,
        mask,
        radius_px=2,
        operation="dilate",
        warp_mode="none",
        blend_method="poisson",
        return_metadata=True,
    )

    assert "blend" in meta
    assert meta["blend"]["requested"] == "poisson"
    assert meta["blend"]["used"] in ("poisson", "alpha")
    assert isinstance(meta["blend"].get("poisson_ok"), (bool, type(None)))

