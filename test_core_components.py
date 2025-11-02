"""
Core component verification test for SCBA.
Tests each component individually without full pipeline integration.
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scba.data.loaders.jsrt import JSRTDataset
from scba.data.transforms.standard import get_composed_transform, get_val_transforms
from scba.models.unet import UNet
from scba.xai.common import explain


def test_model_inference():
    """Test model loading and inference."""
    print("\n[TEST 1] Model Inference")
    print("-" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "runs/jsrt_unet_baseline_20251101_203253.pt"

    # Load model
    model = UNet(n_channels=1, n_classes=2)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 1, 1024, 1024).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (1, 2, 1024, 1024), f"Unexpected output shape: {output.shape}"
    print(f"✓ Model inference working")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    return model, device


def test_data_loading():
    """Test data loading."""
    print("\n[TEST 2] Data Loading")
    print("-" * 60)

    data_root = "/home/mohaisen_mohammed/Datasets/JSRT"
    target_size = (1024, 1024)
    val_transform = get_composed_transform(get_val_transforms(target_size))

    dataset = JSRTDataset(data_root, split="test", transform=val_transform)
    sample = dataset[0]

    assert "image" in sample, "Missing 'image' key"
    assert "mask" in sample, "Missing 'mask' key"
    assert sample["image"].shape == (1, 1024, 1024), f"Unexpected image shape: {sample['image'].shape}"
    assert sample["mask"].shape == (1024, 1024), f"Unexpected mask shape: {sample['mask'].shape}"

    print(f"✓ Data loading working")
    print(f"  Test set size: {len(dataset)}")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Mask shape: {sample['mask'].shape}")
    print(f"  Image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    print(f"  Mask values: {np.unique(sample['mask'])}")

    return dataset


def test_xai_explanation(model, dataset, device):
    """Test XAI explanation generation."""
    print("\n[TEST 3] XAI Explanation Generation")
    print("-" * 60)

    sample = dataset[5]
    image_tensor = sample["image"].unsqueeze(0).to(device)

    # Test seg_grad_cam
    saliency = explain(
        image_tensor, model, method="seg_grad_cam", target_class=1, device=device
    )

    assert saliency.map.shape == (1024, 1024), f"Unexpected saliency shape: {saliency.map.shape}"
    assert saliency.map.min() >= 0 and saliency.map.max() <= 1, "Saliency not normalized"

    print(f"✓ XAI explanation working")
    print(f"  Method: seg_grad_cam")
    print(f"  Saliency shape: {saliency.map.shape}")
    print(f"  Saliency range: [{saliency.map.min():.3f}, {saliency.map.max():.3f}]")
    print(f"  Mean attribution: {saliency.map.mean():.3f}")


def test_evaluation_results():
    """Test that evaluation results exist and are valid."""
    print("\n[TEST 4] Evaluation Results")
    print("-" * 60)

    import json

    results_path = "experiments/results/jsrt_eval/jsrt_test_metrics.json"
    with open(results_path, "r") as f:
        metrics = json.load(f)

    required_metrics = ["dice", "iou", "accuracy", "sensitivity", "specificity"]
    for metric in required_metrics:
        assert metric in metrics, f"Missing metric: {metric}"
        assert 0 <= metrics[metric] <= 1, f"Invalid metric value: {metric}={metrics[metric]}"

    print(f"✓ Evaluation results valid")
    print(f"  Dice: {metrics['dice']:.4f}")
    print(f"  IoU: {metrics['iou']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")


def main():
    print("=" * 70)
    print("SCBA CORE COMPONENTS VERIFICATION")
    print("=" * 70)

    try:
        # Test 1: Model inference
        model, device = test_model_inference()

        # Test 2: Data loading
        dataset = test_data_loading()

        # Test 3: XAI explanation
        test_xai_explanation(model, dataset, device)

        # Test 4: Evaluation results
        test_evaluation_results()

        print("\n" + "=" * 70)
        print("✅ ALL CORE COMPONENTS VERIFIED SUCCESSFULLY!")
        print("=" * 70)
        print("\nComponent Status:")
        print("  ✓ Model loading and inference")
        print("  ✓ Data loading (JSRT dataset)")
        print("  ✓ XAI explanation generation (seg_grad_cam)")
        print("  ✓ Evaluation pipeline (metrics computed)")
        print("\nSystem is ready for research experiments!")
        print("\nNext Steps:")
        print("  1. Fix counterfactual border editing (area budget issues)")
        print("  2. Implement additional XAI methods (hires_cam, rise, etc.)")
        print("  3. Run systematic XAI evaluation experiments")
        print("  4. Generate paper figures and analysis")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
