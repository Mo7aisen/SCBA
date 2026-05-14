"""
Quick end-to-end test of the SCBA XAI pipeline.
Tests: Model loading → XAI explanation → Counterfactual generation → Metric computation
"""

import sys
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scba.cf.borders import apply_border_edit
from scba.cf.inpaint import repair_border_edit
from scba.data.loaders.jsrt import JSRTDataset
from scba.data.transforms.standard import get_composed_transform, get_val_transforms
from scba.metrics.cf_consistency import compute_cf_metrics
from scba.models.unet import UNet
from scba.xai.common import explain


def main():
    print("=" * 70)
    print("SCBA XAI PIPELINE - END-TO-END TEST")
    print("=" * 70)

    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "runs/jsrt_unet_baseline_20251101_203253.pt"
    # Scientific justification (R1): avoid hard-coded absolute dataset paths.
    data_root = os.environ.get("SCBA_JSRT_ROOT")
    if not data_root:
        raise ValueError("Set SCBA_JSRT_ROOT to run this end-to-end test reproducibly.")

    print(f"\n✓ Device: {device}")

    # Step 1: Load model
    print("\n[Step 1] Loading trained model...")
    model = UNet(n_channels=1, n_classes=2)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")

    # Step 2: Load test sample
    print("\n[Step 2] Loading test sample...")
    target_size = (1024, 1024)
    val_transform = get_composed_transform(get_val_transforms(target_size))
    dataset = JSRTDataset(
        data_root,
        split="test",
        transform=val_transform,
        # Scientific justification (R2): write deterministic splits outside the dataset root.
        splits_path=str(Path("experiments/results/scba_publication") / "jsrt_splits.csv"),
    )
    sample = dataset[5]  # Pick a test sample
    image_tensor = sample["image"].unsqueeze(0).to(device)  # (1, 1, H, W)
    mask_np = sample["mask"].numpy()
    image_np = sample["image"].numpy()
    print(f"✓ Loaded sample: shape={image_np.shape}")

    # Step 3: Generate explanations (currently only seg_grad_cam is implemented)
    print("\n[Step 3] Generating XAI explanations...")
    print(f"  - Testing seg_grad_cam...")
    saliency_orig = explain(
        image_tensor,
        model,
        method="seg_grad_cam",
        target_class=1,  # Foreground (lung)
        device=device,
    )
    print(
        f"    ✓ Saliency shape: {saliency_orig.map.shape}, "
        f"range: [{saliency_orig.map.min():.3f}, {saliency_orig.map.max():.3f}]"
    )

    # Step 4: Create counterfactual (border edit)
    print("\n[Step 4] Creating counterfactual (border dilation)...")
    # Convert to proper format (squeeze and ensure uint8 mask)
    image_for_cf = image_np.squeeze()  # (H, W)
    mask_for_cf = (mask_np * 255).astype(np.uint8)  # (H, W) uint8

    image_cf, mask_cf, roi_band = apply_border_edit(
        image_for_cf, mask_for_cf, radius_px=4, operation="dilate", band_px=12, seed=42
    )
    print(f"✓ CF image created: ROI band has {roi_band.sum()} pixels")

    # Step 5: Generate CF explanations
    print("\n[Step 5] Generating explanations on counterfactual...")
    image_cf_tensor = torch.from_numpy(image_cf).unsqueeze(0).unsqueeze(0).to(device)
    saliency_cf = explain(
        image_cf_tensor, model, method="seg_grad_cam", target_class=1, device=device
    )
    print(f"✓ CF saliency shape: {saliency_cf.map.shape}")

    # Step 6: Repair and re-explain
    print("\n[Step 6] Repairing counterfactual and re-explaining...")
    image_repair = repair_border_edit(image_cf, image_for_cf, roi_band)
    image_repair_tensor = (
        torch.from_numpy(image_repair).unsqueeze(0).unsqueeze(0).to(device)
    )
    saliency_repair = explain(
        image_repair_tensor, model, method="seg_grad_cam", target_class=1, device=device
    )
    print(f"✓ Repaired saliency shape: {saliency_repair.map.shape}")

    # Step 7: Compute CF consistency metrics
    print("\n[Step 7] Computing counterfactual consistency metrics...")
    metrics = compute_cf_metrics(
        saliency_orig.map, saliency_cf.map, saliency_repair.map, roi_band
    )

    print("\n" + "=" * 70)
    print("COUNTERFACTUAL CONSISTENCY RESULTS")
    print("=" * 70)
    print(f"AM-ROI (original):     {metrics['am_roi_original']:.4f}")
    print(f"AM-ROI (perturbed):    {metrics['am_roi_perturbed']:.4f}")
    print(f"AM-ROI (repaired):     {metrics['am_roi_repaired']:.4f}")
    print(f"ΔAM-ROI:               {metrics['delta_am_roi']:.4f}")
    print(
        f"CoA shift distance:    {metrics['shift_distance']:.2f} pixels"
    )
    print(
        f"Distance to ROI:       {metrics['distance_to_roi']:.2f} pixels"
    )
    print(
        f"Directional Consistency: {metrics['directional_consistency']:.2f}"
    )
    # Scientific justification: DC is binary and forward-only (manuscript-aligned),
    # so we report the forward distance reduction rather than a backward/repair term.
    print(f"Forward movement (px):  {metrics['forward_movement_px']:.2f}")
    print(f"Dist to ROI (orig px):  {metrics['dist_to_roi_original_px']:.2f}")
    print(f"Dist to ROI (pert px):  {metrics['dist_to_roi_perturbed_px']:.2f}")
    print("=" * 70)

    # Interpretation
    print("\n[Interpretation]")
    if metrics["delta_am_roi"] > 0.1:
        print("✅ GOOD: Saliency increased in edited region (ΔAM-ROI > 0.1)")
    else:
        print("⚠️  LOW: Saliency did not strongly follow the edit")

    if metrics["directional_consistency"] == 1.0:
        print(
            "✅ GOOD: Explanation moved toward the edited ROI (DC=1)"
        )
    else:
        print("⚠️  LOW: Explanation did not move toward the edited ROI (DC=0)")

    if metrics["forward_movement_px"] > 0:
        print("✅ EXCELLENT: Forward distance-to-ROI decreased (moved closer)")
    else:
        print("⚠️  WARNING: Forward distance-to-ROI did not decrease (moved away or unchanged)")

    print("\n" + "=" * 70)
    print("✅ END-TO-END TEST SUCCESSFUL!")
    print("=" * 70)
    print("\nAll pipeline components are working:")
    print("  ✓ Model loading and inference")
    print("  ✓ XAI explanation generation (seg_grad_cam)")
    print("  ✓ Counterfactual border editing")
    print("  ✓ Image repair/inpainting")
    print("  ✓ CF consistency metrics computation")
    print("\n⚠️  NOTE: Currently only seg_grad_cam is implemented.")
    print("   Other XAI methods (hires_cam, rise, etc.) need to be added to:")
    print("   scba/xai/common.py (method_map)")
    print("\nCore pipeline is functional and ready for experiments!")


if __name__ == "__main__":
    main()
