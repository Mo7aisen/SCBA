"""Test fixed border editing."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from scba.cf.borders import apply_border_edit
from scba.data.loaders.jsrt import JSRTDataset
from scba.data.transforms.standard import get_composed_transform, get_val_transforms


def main():
    print("=" * 70)
    print("TESTING FIXED BORDER EDITING")
    print("=" * 70)

    # Load a test sample
    data_root = "/home/mohaisen_mohammed/Datasets/JSRT"
    target_size = (1024, 1024)
    val_transform = get_composed_transform(get_val_transforms(target_size))
    dataset = JSRTDataset(data_root, split="test", transform=val_transform)
    sample = dataset[5]

    image_np = sample["image"].squeeze().numpy()  # (H, W)
    mask_np = sample["mask"].numpy()  # (H, W)

    print(f"\nInput:")
    print(f"  Image shape: {image_np.shape}, dtype: {image_np.dtype}")
    print(f"  Image range: [{image_np.min():.3f}, {image_np.max():.3f}]")
    print(f"  Mask shape: {mask_np.shape}, dtype: {mask_np.dtype}")
    print(f"  Mask values: {np.unique(mask_np)}")
    print(f"  Mask area: {mask_np.sum()} pixels")

    # Test with different radii
    test_configs = [
        {"radius_px": 2, "operation": "dilate"},
        {"radius_px": 3, "operation": "dilate"},
        {"radius_px": 4, "operation": "dilate"},
        {"radius_px": 2, "operation": "erode"},
    ]

    print("\n" + "=" * 70)
    print("TESTING DIFFERENT CONFIGURATIONS")
    print("=" * 70)

    for i, config in enumerate(test_configs):
        print(f"\n[Test {i+1}] radius={config['radius_px']}px, operation='{config['operation']}'")

        # Ensure mask is uint8 and binary (0 or 1, not 0 or 255)
        mask_uint8 = mask_np.astype(np.uint8)

        try:
            image_cf, mask_cf, roi_band = apply_border_edit(
                image_np,
                mask_uint8,
                radius_px=config["radius_px"],
                operation=config["operation"],
                band_px=12,
                area_budget=0.50,
                seed=42,
            )

            # Compute statistics
            area_orig = mask_uint8.sum()
            area_cf = mask_cf.sum()
            roi_pixels = roi_band.sum()
            delta_area = abs(area_cf - area_orig) / max(area_orig, 1)

            print(f"  ✓ SUCCESS")
            print(f"    Original mask area: {area_orig} pixels")
            print(f"    CF mask area: {area_cf} pixels")
            print(f"    Area change: {delta_area:.3f} ({delta_area*100:.1f}%)")
            print(f"    ROI band pixels: {roi_pixels}")
            print(f"    CF image range: [{image_cf.min():.3f}, {image_cf.max():.3f}]")

            if roi_pixels == 0:
                print(f"  ⚠️  WARNING: ROI band is empty!")
            else:
                print(f"  ✓ ROI band looks good ({roi_pixels} pixels)")

        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("✅ BORDER EDITING TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
