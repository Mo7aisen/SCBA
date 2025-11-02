"""
Complete SCBA (Synthetic Counterfactual Border Audit) Pipeline

Professional implementation for publication-ready results.
Tests counterfactual consistency of multiple XAI methods.
"""

import json
import sys
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from scba.cf.borders import apply_border_edit
from scba.cf.inpaint import repair_border_edit
from scba.data.loaders.jsrt import JSRTDataset
from scba.data.transforms.standard import get_composed_transform, get_val_transforms
from scba.metrics.cf_consistency import compute_cf_metrics
from scba.models.unet import UNet
from scba.xai.common import explain


def main():
    print("=" * 80)
    print("SCBA: SYNTHETIC COUNTERFACTUAL BORDER AUDIT")
    print("Professional Pipeline for XAI Evaluation in Medical Image Segmentation")
    print("=" * 80)

    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "runs/jsrt_unet_baseline_20251101_203253.pt"
    data_root = "/home/mohaisen_mohammed/Datasets/JSRT"
    output_dir = Path("experiments/results/scba_pilot")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n✓ Device: {device}")
    print(f"✓ Output: {output_dir}")

    # Load model
    print("\n[1/6] Loading trained model...")
    model = UNet(n_channels=1, n_classes=2)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    best_score = checkpoint.get('best_score', None)
    score_str = f"{best_score:.4f}" if best_score is not None else "N/A"
    print(f"✓ Model loaded (epoch {checkpoint.get('epoch', 'unknown')}, Dice: {score_str})")

    # Load test samples
    print("\n[2/6] Loading test samples...")
    target_size = (1024, 1024)
    val_transform = get_composed_transform(get_val_transforms(target_size))
    dataset = JSRTDataset(data_root, split="test", transform=val_transform, return_patient_id=True)

    # Select diverse test samples
    n_samples = 5  # Professional: test on multiple samples
    test_indices = [0, 5, 10, 15, 20]  # Spread across test set
    print(f"✓ Loaded {len(dataset)} test samples, using {n_samples} for evaluation")

    # XAI methods to test (use fast CAM methods for now)
    xai_methods = {
        "seg_grad_cam": "Seg-Grad-CAM",
        "hires_cam": "HiResCAM",
        "grad_cam_pp": "Grad-CAM++",
        "seg_xres_cam": "Seg-XRes-CAM",
    }
    print(f"✓ Testing {len(xai_methods)} XAI methods")

    # Counterfactual configurations
    cf_configs = [
        {"radius_px": 2, "operation": "dilate", "desc": "Dilate r=2"},
        {"radius_px": 3, "operation": "dilate", "desc": "Dilate r=3"},
        {"radius_px": 2, "operation": "erode", "desc": "Erode r=2"},
    ]
    print(f"✓ Testing {len(cf_configs)} counterfactual perturbations")

    # Run experiments
    print("\n[3/6] Running SCBA experiments...")
    print("=" * 80)

    all_results = {}
    total_experiments = n_samples * len(xai_methods) * len(cf_configs)
    experiment_counter = 0

    for sample_idx in test_indices:
        sample = dataset[sample_idx]
        patient_id = sample["patient_id"]
        image_np = sample["image"].squeeze().numpy()  # (H, W)
        mask_np = sample["mask"].numpy()  # (H, W)
        image_tensor = sample["image"].unsqueeze(0).to(device)  # (1, 1, H, W)

        print(f"\n{'='*80}")
        print(f"Sample: {patient_id} ({sample_idx+1}/{n_samples})")
        print(f"{'='*80}")

        sample_results = {}

        for method_name, method_desc in xai_methods.items():
            print(f"\n  XAI Method: {method_desc}")

            # Generate original explanation
            start_time = time.time()
            saliency_orig = explain(
                image_tensor, model, method=method_name, target_class=1, device=device
            )
            explain_time = time.time() - start_time
            print(f"    ✓ Explanation generated ({explain_time:.2f}s)")

            method_results = {}

            for cf_config in cf_configs:
                experiment_counter += 1
                print(f"    [{experiment_counter}/{total_experiments}] CF: {cf_config['desc']}...")

                # Convert mask to uint8 and ensure binary
                mask_uint8 = mask_np.astype(np.uint8)

                # Apply counterfactual border edit
                try:
                    image_cf, mask_cf, roi_band = apply_border_edit(
                        image_np,
                        mask_uint8,
                        radius_px=cf_config["radius_px"],
                        operation=cf_config["operation"],
                        band_px=12,
                        area_budget=0.50,
                        seed=42,
                    )

                    if roi_band.sum() == 0:
                        print(f"      ⚠️  ROI band empty, skipping")
                        continue

                    # Generate CF explanation
                    image_cf_tensor = torch.from_numpy(image_cf).unsqueeze(0).unsqueeze(0).to(device)
                    saliency_cf = explain(
                        image_cf_tensor, model, method=method_name, target_class=1, device=device
                    )

                    # Repair and re-explain
                    image_repair = repair_border_edit(image_cf, image_np, roi_band)
                    image_repair_tensor = torch.from_numpy(image_repair).unsqueeze(0).unsqueeze(0).to(device)
                    saliency_repair = explain(
                        image_repair_tensor, model, method=method_name, target_class=1, device=device
                    )

                    # Compute CF consistency metrics
                    metrics = compute_cf_metrics(
                        saliency_orig.map,
                        saliency_cf.map,
                        saliency_repair.map,
                        roi_band,
                    )

                    # Store results
                    cf_key = f"{cf_config['operation']}_r{cf_config['radius_px']}"
                    method_results[cf_key] = {
                        "config": cf_config,
                        "metrics": metrics,
                        "roi_pixels": int(roi_band.sum()),
                        "area_change": float(abs(mask_cf.sum() - mask_uint8.sum()) / max(mask_uint8.sum(), 1)),
                    }

                    # Print key metrics
                    print(f"      ✓ ΔAM-ROI: {metrics['delta_am_roi']:.4f} | " +
                          f"CoA shift: {metrics['shift_distance']:.1f}px | " +
                          f"DC: {metrics['directional_consistency']:.2f}")

                    # Clear GPU memory
                    if device.type == "cuda":
                        del image_cf_tensor, saliency_cf, image_repair_tensor, saliency_repair
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"      ❌ Error: {e}")
                    continue

            sample_results[method_name] = method_results

        all_results[patient_id] = sample_results

    # Aggregate results
    print("\n[4/6] Analyzing results...")
    print("=" * 80)

    method_aggregates = {}
    for method_name in xai_methods.keys():
        all_delta_am = []
        all_shifts = []
        all_dc = []

        for patient_results in all_results.values():
            if method_name in patient_results:
                for cf_results in patient_results[method_name].values():
                    all_delta_am.append(cf_results["metrics"]["delta_am_roi"])
                    all_shifts.append(cf_results["metrics"]["shift_distance"])
                    all_dc.append(cf_results["metrics"]["directional_consistency"])

        if all_delta_am:
            method_aggregates[method_name] = {
                "mean_delta_am_roi": np.mean(all_delta_am),
                "std_delta_am_roi": np.std(all_delta_am),
                "mean_shift_distance": np.mean(all_shifts),
                "std_shift_distance": np.std(all_shifts),
                "mean_dc": np.mean(all_dc),
                "std_dc": np.std(all_dc),
                "n_experiments": len(all_delta_am),
            }

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Counterfactual Consistency Across Methods")
    print("=" * 80)
    print(f"{'Method':<20} {'ΔAM-ROI':<15} {'CoA Shift (px)':<18} {'DC':<12} {'N'}")
    print("-" * 80)
    for method_name, agg in sorted(method_aggregates.items()):
        delta_str = f"{agg['mean_delta_am_roi']:.4f}±{agg['std_delta_am_roi']:.4f}"
        shift_str = f"{agg['mean_shift_distance']:.1f}±{agg['std_shift_distance']:.1f}"
        dc_str = f"{agg['mean_dc']:.3f}±{agg['std_dc']:.3f}"
        print(f"{xai_methods[method_name]:<20} {delta_str:<15} {shift_str:<18} {dc_str:<12} {agg['n_experiments']}")
    print("=" * 80)

    # Save results
    print("\n[5/6] Saving results...")
    results_file = output_dir / "scba_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "summary": method_aggregates,
            "detailed_results": all_results,
            "config": {
                "model_path": str(model_path),
                "n_samples": n_samples,
                "xai_methods": list(xai_methods.keys()),
                "cf_configs": cf_configs,
            }
        }, f, indent=2)
    print(f"✓ Results saved to {results_file}")

    # Generate visualization
    print("\n[6/6] Generating publication figure...")
    fig, axes = plt.subplots(2, len(xai_methods), figsize=(16, 8))

    for i, (method_name, method_desc) in enumerate(xai_methods.items()):
        if method_name in method_aggregates:
            agg = method_aggregates[method_name]

            # ΔAM-ROI bar
            axes[0, i].bar([0], [agg["mean_delta_am_roi"]], yerr=[agg["std_delta_am_roi"]],
                          color='steelblue', capsize=5)
            axes[0, i].set_title(method_desc, fontsize=10)
            axes[0, i].set_ylabel("ΔAM-ROI", fontsize=9)
            axes[0, i].set_xlim(-0.5, 0.5)
            axes[0, i].set_xticks([])
            axes[0, i].grid(axis='y', alpha=0.3)

            # Directional Consistency bar
            axes[1, i].bar([0], [agg["mean_dc"]], yerr=[agg["std_dc"]],
                          color='coral', capsize=5)
            axes[1, i].set_ylabel("Directional Consistency", fontsize=9)
            axes[1, i].set_ylim(0, 1)
            axes[1, i].set_xlim(-0.5, 0.5)
            axes[1, i].set_xticks([])
            axes[1, i].grid(axis='y', alpha=0.3)

    plt.suptitle("SCBA: Counterfactual Consistency Evaluation", fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig_path = output_dir / "scba_summary.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved to {fig_path}")

    print("\n" + "=" * 80)
    print("✅ SCBA PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Results: {results_file}")
    print(f"📈 Figure: {fig_path}")
    print(f"\n🎯 Key Finding: Best method based on ΔAM-ROI:")
    best_method = max(method_aggregates.items(), key=lambda x: x[1]["mean_delta_am_roi"])
    print(f"   {xai_methods[best_method[0]]}: ΔAM-ROI = {best_method[1]['mean_delta_am_roi']:.4f}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
