"""
Comprehensive test of all XAI methods on GPU.
"""

import sys
from pathlib import Path
import time

import torch

sys.path.insert(0, str(Path(__file__).parent))

from scba.data.loaders.jsrt import JSRTDataset
from scba.data.transforms.standard import get_composed_transform, get_val_transforms
from scba.models.unet import UNet
from scba.xai.common import explain


def main():
    print("=" * 70)
    print("COMPREHENSIVE XAI METHODS TEST - GPU VERIFICATION")
    print("=" * 70)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Device: {device}")

    if device.type != "cuda":
        print("⚠️  WARNING: No GPU available! Tests will run on CPU (slower)")

    # Load model
    print("\n[1/3] Loading model...")
    model_path = "runs/jsrt_unet_baseline_20251101_203253.pt"
    model = UNet(n_channels=1, n_classes=2)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")

    # Load test sample
    print("\n[2/3] Loading test sample...")
    data_root = "/home/mohaisen_mohammed/Datasets/JSRT"
    target_size = (1024, 1024)
    val_transform = get_composed_transform(get_val_transforms(target_size))
    dataset = JSRTDataset(data_root, split="test", transform=val_transform)
    sample = dataset[5]
    image_tensor = sample["image"].unsqueeze(0).to(device)
    print(f"✓ Sample loaded: {image_tensor.shape}")

    # Test all methods
    print("\n[3/3] Testing XAI methods...")
    print("=" * 70)

    methods_to_test = {
        "seg_grad_cam": {"desc": "Seg-Grad-CAM (segmentation-aware)"},
        "seg_xres_cam": {"desc": "Seg-XRes-CAM (spatially weighted)"},
        "hires_cam": {"desc": "HiResCAM (high-resolution)"},
        "grad_cam_pp": {"desc": "Grad-CAM++ (pixel-wise weighting)"},
        "rise": {"desc": "RISE (random masking)", "n_masks": 500},  # Reduce for faster testing
        "occlusion": {"desc": "Occlusion (sliding window)", "patch_size": 32, "stride": 16},
    }

    results = {}

    for method_name, config in methods_to_test.items():
        print(f"\n[Method: {method_name}]")
        print(f"  Description: {config['desc']}")

        try:
            # Clear GPU cache
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # Start timing
            start_time = time.time()

            # Generate explanation
            kwargs = {k: v for k, v in config.items() if k != "desc"}
            saliency = explain(
                image_tensor,
                model,
                method=method_name,
                target_class=1,
                device=device,
                **kwargs
            )

            # End timing
            elapsed = time.time() - start_time

            # Validate output
            assert saliency.map.shape == (1024, 1024), f"Unexpected shape: {saliency.map.shape}"
            assert saliency.map.min() >= 0 and saliency.map.max() <= 1, "Not normalized"

            # Store results
            results[method_name] = {
                "success": True,
                "time": elapsed,
                "mean_saliency": saliency.map.mean(),
                "max_saliency": saliency.map.max(),
            }

            print(f"  ✅ SUCCESS")
            print(f"     Time: {elapsed:.2f}s")
            print(f"     Saliency range: [{saliency.map.min():.3f}, {saliency.map.max():.3f}]")
            print(f"     Mean attribution: {saliency.map.mean():.3f}")

            # GPU memory status
            if device.type == "cuda":
                mem_allocated = torch.cuda.memory_allocated(0) / 1e9
                mem_reserved = torch.cuda.memory_reserved(0) / 1e9
                print(f"     GPU memory: {mem_allocated:.2f}/{mem_reserved:.2f} GB")

        except Exception as e:
            results[method_name] = {
                "success": False,
                "error": str(e),
            }
            print(f"  ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    success_count = sum(1 for r in results.values() if r["success"])
    total_count = len(results)

    print(f"\n✅ {success_count}/{total_count} methods working")

    if success_count == total_count:
        print("\n🎉 ALL XAI METHODS OPERATIONAL ON GPU!")
    else:
        print("\n⚠️  Some methods failed. Check errors above.")

    # Performance comparison
    if success_count > 0:
        print("\nPerformance Comparison:")
        print("-" * 70)
        print(f"{'Method':<20} {'Time (s)':<12} {'Mean Sal.':<12} {'Notes'}")
        print("-" * 70)
        for method, result in sorted(results.items()):
            if result["success"]:
                time_str = f"{result['time']:.2f}"
                mean_str = f"{result['mean_saliency']:.3f}"
                notes = "Fast" if result["time"] < 1.0 else "Slow" if result["time"] > 10.0 else "Medium"
                print(f"{method:<20} {time_str:<12} {mean_str:<12} {notes}")

    print("=" * 70)


if __name__ == "__main__":
    main()
