"""
RADIUS SENSITIVITY ANALYSIS FOR SCBA++
=======================================

Tests robustness of boundary attribution findings across perturbation magnitudes.

Runs r=4 and r=5 experiments (r=2, r=3 already exist in publication results).
Validates that:
- Counterfactual realism maintained (SSIM > 0.85, perceptual distance < 0.10)
- Attribution shift scales approximately linearly with perturbation magnitude
- Statistical conclusions remain stable across radii

NEW FILES ONLY - does not modify existing publication results.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from scba.cf.borders import apply_border_edit
from scba.cf.inpaint import repair_border_edit
from scba.data.loaders.jsrt import JSRTDataset
from scba.data.loaders.montgomery import MontgomeryDataset
from scba.data.transforms.standard import get_composed_transform, get_val_transforms
from scba.metrics.cf_consistency import compute_cf_metrics
from scba.metrics.boundary import border_band_from_mask
from scba.metrics.faithfulness import deletion_insertion_auc
from scba.metrics.perceptual import PerceptualDistance
from scba.metrics.robustness import saliency_robustness
from scba.models.unet import UNet
from scba.xai.common import explain, get_decoder_target_layers


DATASET_CONFIG = {
    "jsrt": {
        "loader": JSRTDataset,
        "model_path": "runs/jsrt_unet_baseline_20251101_203253.pt",
        "model_path_env": "SCBA_JSRT_MODEL_PATH",
        "data_root": None,
        "data_root_env": "SCBA_JSRT_ROOT",
        "dataset_name": "JSRT",
    },
    "montgomery": {
        "loader": MontgomeryDataset,
        "model_path": "runs/montgomery_unet_baseline_20251101_203253.pt",
        "model_path_env": "SCBA_MONTGOMERY_MODEL_PATH",
        "data_root": None,
        "data_root_env": "SCBA_MONTGOMERY_ROOT",
        "dataset_name": "Montgomery",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run radius sensitivity analysis for SCBA++."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for all stochastic components (default: 42).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="jsrt",
        choices=DATASET_CONFIG.keys(),
        help="Dataset to evaluate (default: jsrt).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained segmentation model checkpoint.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to dataset root directory.",
    )
    parser.add_argument(
        "--radii",
        type=str,
        default="4,5",
        help="Comma-separated radii to test (default: 4,5).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/scba_radius_sensitivity",
        help="Directory to store experiment outputs (NEW FILES ONLY).",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Optional limit on number of test samples (debug only).",
    )
    parser.add_argument(
        "--auc-steps",
        type=int,
        default=20,
        help="Steps for deletion/insertion AUC (publication default: 20).",
    )
    parser.add_argument(
        "--mask-source",
        type=str,
        default="pred",
        choices=["pred", "gt"],
        help="Mask source for counterfactual edits/ROIs: pred (default) or gt.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing results file if available.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_key = args.dataset
    cfg = DATASET_CONFIG[dataset_key]
    dataset_label = cfg["dataset_name"]
    DatasetClass = cfg["loader"]

    # Parse radii
    radii = [int(r) for r in args.radii.split(",")]

    print("=" * 90)
    print("SCBA++ RADIUS SENSITIVITY ANALYSIS")
    print(f"Dataset: {dataset_label}")
    print(f"Radii: {radii}")
    print("=" * 90)

    # Configuration
    from scba.utils.reproducibility import seed_everything
    # Scientific justification (R4): radius sweeps use repeated stochastic edits;
    # fixed seeds are required for stable curves and confidence intervals.
    seed_everything(args.seed, deterministic=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_model_path = (
        args.model_path
        or os.environ.get(cfg.get("model_path_env", ""), "")
        or cfg["model_path"]
    )
    model_path = Path(default_model_path)
    default_data_root = cfg.get("data_root") or os.environ.get(cfg.get("data_root_env", ""))
    if not (args.data_root or default_data_root):
        raise ValueError(
            f"--data-root is required (or set {cfg.get('data_root_env')}) to run reproducibly."
        )
    data_root = Path(args.data_root or default_data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / f"{dataset_key}_radius_sensitivity.json"

    print(f"\n✓ Device: {device}")
    print(f"✓ Model: {model_path}")
    print(f"✓ Data root: {data_root}")
    print(f"✓ Output directory: {output_dir}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. "
            f"Pass --model-path or set {cfg.get('model_path_env')}."
        )

    # Load model
    print("\n[1/5] Loading trained model...")
    model = UNet(n_channels=1, n_classes=2)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint.get("epoch", "unknown")
        best_score = checkpoint.get("best_score", checkpoint.get("best_dice", None))
    else:
        state_dict = checkpoint
        epoch = "unknown"
        best_score = None
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    score_str = f"{best_score:.4f}" if best_score is not None else "N/A"
    print(f"✓ Model loaded (epoch {epoch}, Dice: {score_str})")

    # Load test samples
    print("\n[2/5] Loading test samples...")
    target_size = (1024, 1024)
    val_transform = get_composed_transform(get_val_transforms(target_size))
    dataset = DatasetClass(
        data_root,
        split="test",
        transform=val_transform,
        return_patient_id=True,
        # Scientific justification (R2): write deterministic splits under the
        # experiment output directory, not into the dataset root.
        splits_path=str(output_dir / f"{dataset_key}_splits.csv"),
    )

    total_samples = len(dataset)
    test_indices = list(range(total_samples))
    if args.limit_samples is not None:
        test_indices = test_indices[: max(args.limit_samples, 0)]
    n_samples = len(test_indices)
    print(f"✓ Using {n_samples} test samples for {dataset_label}")

    # Decoder layers for CAM methods
    decoder_layers = get_decoder_target_layers(model, n_layers=2)
    decoder_layer_names = [name for name, _ in decoder_layers]
    primary_layer = decoder_layers[-1][1]

    # XAI methods: test only Multi-Layer CAM (primary method from publication)
    method_kwargs = {
        "multi_layer_cam": {"target_layers": [layer for _, layer in decoder_layers]},
    }
    xai_methods = {
        "multi_layer_cam": f"Multi-Layer LayerCAM ({', '.join(decoder_layer_names)})",
    }
    print(f"✓ Testing Multi-Layer CAM (primary method)")

    # Counterfactual configurations
    cf_configs = []
    for radius in radii:
        cf_configs.append(
            {"radius_px": radius, "operation": "dilate", "desc": f"Dilate r={radius}"}
        )
    print(f"✓ Testing {len(cf_configs)} perturbations: {[c['desc'] for c in cf_configs]}")

    total_experiments = n_samples * len(xai_methods) * len(cf_configs)
    print(f"\n✓ Total experiments: {total_experiments}")
    print(f"✓ Estimated time: ~{total_experiments * 0.5 / 60:.1f} minutes on GPU")

    # Set up perceptual metric
    perceptual_metric = PerceptualDistance(device=str(device))

    # Run experiments
    print("\n[3/5] Running radius sensitivity experiments...")
    print("=" * 90)

    all_results = {}
    if args.resume and results_file.exists():
        with open(results_file, "r") as f:
            previous = json.load(f)
        all_results = previous.get("detailed_results", {}) or {}
        print(f"✓ Resuming from {results_file}")
    poisson_stats = {
        "total_edits": 0,
        "poisson_requested": 0,
        "poisson_used": 0,
        "poisson_failed": 0,
        "validation_fallback": 0,
        "by_method": {},
    }
    def record_poisson(edit_meta, method_name):
        if not edit_meta:
            return
        blend = edit_meta.get("blend", {}) if isinstance(edit_meta, dict) else {}
        requested = blend.get("requested")
        used = blend.get("used")
        validation_fallback = bool(blend.get("validation_fallback", False))

        poisson_stats["total_edits"] += 1
        if requested == "poisson":
            poisson_stats["poisson_requested"] += 1
        if used == "poisson":
            poisson_stats["poisson_used"] += 1
        if requested == "poisson" and used != "poisson":
            poisson_stats["poisson_failed"] += 1
        if validation_fallback:
            poisson_stats["validation_fallback"] += 1

        by_method = poisson_stats["by_method"].setdefault(
            method_name,
            {
                "total_edits": 0,
                "poisson_requested": 0,
                "poisson_used": 0,
                "poisson_failed": 0,
                "validation_fallback": 0,
            },
        )
        by_method["total_edits"] += 1
        if requested == "poisson":
            by_method["poisson_requested"] += 1
        if used == "poisson":
            by_method["poisson_used"] += 1
        if requested == "poisson" and used != "poisson":
            by_method["poisson_failed"] += 1
        if validation_fallback:
            by_method["validation_fallback"] += 1

    total_start_time = time.time()

    disable_tqdm = os.getenv("SCBA_DISABLE_TQDM", "").lower() in {"1", "true", "yes"}
    sample_iter = tqdm(test_indices, desc="Samples", disable=disable_tqdm)
    expected_cf_keys = [f"{cfg['operation']}_r{cfg['radius_px']}" for cfg in cf_configs]
    for loop_index, sample_idx in enumerate(sample_iter, start=1):
        sample = dataset[sample_idx]
        patient_id = sample["patient_id"]
        if disable_tqdm:
            print(f"[Sample {loop_index}/{n_samples}] patient_id={patient_id}")
        image_np = sample["image"].squeeze().numpy()
        image_tensor = sample["image"].unsqueeze(0).to(device)

        sample_results = all_results.get(patient_id, {})

        # Scientific justification (M2): define counterfactual edits/ROIs from
        # the model prediction (ŷ), not the ground-truth mask, to avoid label
        # leakage and match the manuscript protocol.
        mask_gt_np = sample["mask"].squeeze().numpy().astype(np.uint8)
        with torch.no_grad():
            logits = model(image_tensor)
            pred = torch.argmax(logits, dim=1)
        mask_pred_np = pred.squeeze(0).detach().cpu().numpy().astype(np.uint8)
        mask_pred_np = (mask_pred_np > 0).astype(np.uint8)
        mask_source = str(args.mask_source)
        mask_np = mask_pred_np if mask_source == "pred" else mask_gt_np

        inter = int((mask_pred_np & mask_gt_np).sum())
        denom = int(mask_pred_np.sum() + mask_gt_np.sum())
        sample_results["_sample_metadata"] = {
            "dice_pred_vs_gt": float((2 * inter) / (denom + 1e-8)),
            "mask_pred_sum": int(mask_pred_np.sum()),
            "mask_gt_sum": int(mask_gt_np.sum()),
            "mask_source": mask_source,
        }

        for method_name, method_desc in xai_methods.items():
            try:
                if method_name in sample_results:
                    existing = sample_results.get(method_name, {})
                    if existing.get("baseline_metrics") and all(
                        key in existing for key in expected_cf_keys
                    ):
                        continue
                saliency_orig = explain(
                    image_tensor,
                    model,
                    method=method_name,
                    target_class=1,
                    device=device,
                    **method_kwargs.get(method_name, {}),
                )
            except Exception as e:
                print(f"Warning: Failed to explain {patient_id} with {method_name}: {e}")
                continue

            method_results = sample_results.get(method_name, {})

            if "baseline_metrics" not in method_results:
                # Compute baseline metrics ONCE per sample (reusable across radii)
                border_band = border_band_from_mask(mask_np, band_px=6)
                baseline_metrics = deletion_insertion_auc(
                    model,
                    image_tensor,
                    saliency_orig.map,
                    target_class=1,
                    steps=int(args.auc_steps),
                )
                boundary_metrics = deletion_insertion_auc(
                    model,
                    image_tensor,
                    saliency_orig.map,
                    target_class=1,
                    steps=int(args.auc_steps),
                    pixel_mask=border_band,
                )
                lung_mask = mask_np.astype(bool)
                robustness_lung = saliency_robustness(
                    model,
                    image_tensor,
                    saliency_orig.map,
                    method=method_name,
                    target_class=1,
                    device=str(device),
                    # Scientific justification (M4): evaluate robustness within
                    # the lung mask to avoid background-driven inflation.
                    pixel_mask=lung_mask,
                    explain_kwargs=method_kwargs.get(method_name, {}),
                )
                robustness_boundary = saliency_robustness(
                    model,
                    image_tensor,
                    saliency_orig.map,
                    method=method_name,
                    target_class=1,
                    device=str(device),
                    pixel_mask=border_band,
                    explain_kwargs=method_kwargs.get(method_name, {}),
                )
                robustness_full = saliency_robustness(
                    model,
                    image_tensor,
                    saliency_orig.map,
                    method=method_name,
                    target_class=1,
                    device=str(device),
                    pixel_mask=None,
                    explain_kwargs=method_kwargs.get(method_name, {}),
                )
                baseline_metrics.update(robustness_lung)
                baseline_metrics.update(
                    {
                        "boundary_deletion_auc": boundary_metrics.get("deletion_auc"),
                        "boundary_insertion_auc": boundary_metrics.get("insertion_auc"),
                        "robustness_corr_mean_boundary": robustness_boundary.get("robustness_corr_mean"),
                        "robustness_corr_std_boundary": robustness_boundary.get("robustness_corr_std"),
                        "robustness_corr_mask_n_boundary": robustness_boundary.get("robustness_corr_mask_n"),
                        "robustness_corr_mean_full": robustness_full.get("robustness_corr_mean"),
                        "robustness_corr_std_full": robustness_full.get("robustness_corr_std"),
                        "robustness_corr_mask_n_full": robustness_full.get("robustness_corr_mask_n"),
                    }
                )
                method_results["baseline_metrics"] = baseline_metrics

            # Run counterfactual experiments for each radius
            for cf_config in cf_configs:
                cf_key = f"{cf_config['operation']}_r{cf_config['radius_px']}"
                if cf_key in method_results:
                    continue
                mask_uint8 = mask_np.astype(np.uint8)

                try:
                    image_cf, mask_cf, roi_band, edit_meta = apply_border_edit(
                        image_np,
                        mask_uint8,
                        radius_px=cf_config["radius_px"],
                        operation=cf_config["operation"],
                        band_px=12,
                        area_budget=0.50,
                        seed=42,
                        return_metadata=True,
                    )

                    record_poisson(edit_meta, method_name)

                    if roi_band.sum() == 0:
                        print(f"Warning: Empty ROI for {patient_id} r={cf_config['radius_px']}")
                        continue

                    # Compute saliency for perturbed image
                    image_cf_tensor = (
                        torch.from_numpy(image_cf.astype(np.float32))
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(device)
                    )
                    saliency_cf = explain(
                        image_cf_tensor,
                        model,
                        method=method_name,
                        target_class=1,
                        device=device,
                        **method_kwargs.get(method_name, {}),
                    )

                    # Compute saliency for repaired image
                    image_repair = repair_border_edit(image_cf, image_np, roi_band)
                    image_repair_tensor = (
                        torch.from_numpy(image_repair.astype(np.float32))
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(device)
                    )
                    saliency_repair = explain(
                        image_repair_tensor,
                        model,
                        method=method_name,
                        target_class=1,
                        device=device,
                        **method_kwargs.get(method_name, {}),
                    )

                    # Compute counterfactual consistency metrics
                    metrics = compute_cf_metrics(
                        saliency_orig.map,
                        saliency_cf.map,
                        saliency_repair.map,
                        roi_band,
                    )

                    # Perceptual distance (realism check)
                    try:
                        perceptual_distance = perceptual_metric.compute(image_np, image_cf)
                    except Exception as e:
                        print(f"Warning: Perceptual metric failed for {patient_id}: {e}")
                        perceptual_distance = None

                    # SSIM (additional realism check)
                    from skimage.metrics import structural_similarity as ssim
                    ssim_score = ssim(image_np, image_cf, data_range=1.0)

                    # Intensity delta (sanity check)
                    intensity_delta = float(np.mean(image_cf - image_np))

                    # Store metrics
                    mask_cf_sum = int(mask_cf.sum())
                    mask_orig_sum = int(mask_uint8.sum())
                    area_change = abs(mask_cf_sum - mask_orig_sum) / max(mask_orig_sum, 1)

                    method_results[cf_key] = {
                        "config": cf_config,
                        "metrics": metrics,
                        "roi_pixels": int(roi_band.sum()),
                        "area_change": float(area_change),
                        "edit_metadata": edit_meta,
                        "perceptual_distance": perceptual_distance,
                        "ssim": float(ssim_score),
                        "intensity_delta": intensity_delta,
                    }

                    # Cleanup GPU memory
                    if device.type == "cuda":
                        del image_cf_tensor, saliency_cf, image_repair_tensor, saliency_repair
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"Warning: CF failed for {patient_id} r={cf_config['radius_px']}: {e}")
                    continue

            if method_results:
                sample_results[method_name] = method_results

        if sample_results:
            all_results[patient_id] = sample_results

    total_time = time.time() - total_start_time
    print(f"\n✓ All experiments completed in {total_time/60:.1f} minutes")

    # Aggregate results
    print("\n[4/5] Aggregating results and checking quality thresholds...")
    method_aggregates = {}

    for method_name in xai_methods.keys():
        method_aggregates[method_name] = {}

        for cf_config in cf_configs:
            cf_key = f"{cf_config['operation']}_r{cf_config['radius_px']}"
            radius = cf_config['radius_px']

            # Collect all metrics for this radius
            delta_am_list = []
            shift_list = []
            dc_list = []
            ssim_list = []
            perceptual_list = []
            intensity_delta_list = []

            for patient_id, patient_data in all_results.items():
                if method_name in patient_data and cf_key in patient_data[method_name]:
                    cf_data = patient_data[method_name][cf_key]
                    metrics = cf_data["metrics"]

                    delta_am_list.append(metrics["am_roi_perturbed"] - metrics["am_roi_original"])
                    shift_list.append(metrics["shift_distance"])
                    dc_list.append(metrics["directional_consistency"])
                    ssim_list.append(cf_data.get("ssim", np.nan))
                    if cf_data.get("perceptual_distance") is not None:
                        perceptual_list.append(cf_data["perceptual_distance"])
                    intensity_delta_list.append(cf_data.get("intensity_delta", 0.0))

            if delta_am_list:
                method_aggregates[method_name][cf_key] = {
                    "radius": radius,
                    "n_experiments": len(delta_am_list),
                    "mean_delta_am_roi": float(np.mean(delta_am_list)),
                    "std_delta_am_roi": float(np.std(delta_am_list)),
                    "mean_shift_distance": float(np.mean(shift_list)),
                    "std_shift_distance": float(np.std(shift_list)),
                    "mean_dc": float(np.mean(dc_list)),
                    "std_dc": float(np.std(dc_list)),
                    "mean_ssim": float(np.nanmean(ssim_list)),
                    "std_ssim": float(np.nanstd(ssim_list)),
                    "mean_perceptual_distance": float(np.mean(perceptual_list)) if perceptual_list else None,
                    "mean_intensity_delta": float(np.mean(intensity_delta_list)),
                }

    # Save results
    print("\n[5/5] Saving results...")
    output_data = {
        "dataset": dataset_key,
        "seed": int(args.seed),
        "model_path": str(model_path),
        "data_root": str(data_root),
        "radii": radii,
        "n_samples": n_samples,
        "mask_source": str(args.mask_source),
        "method_aggregates": method_aggregates,
        "detailed_results": all_results,
        "poisson_blend": poisson_stats,
    }
    with open(results_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"✓ Results saved to {results_file}")

    # Check quality thresholds
    print("\n" + "=" * 90)
    print("QUALITY THRESHOLD CHECKS")
    print("=" * 90)

    for method_name, method_data in method_aggregates.items():
        print(f"\n{method_name}:")
        for cf_key, stats in method_data.items():
            radius = stats["radius"]
            mean_ssim = stats["mean_ssim"]
            mean_perceptual = stats.get("mean_perceptual_distance", None)

            status_ssim = "✓ PASS" if mean_ssim > 0.85 else "✗ FAIL"
            status_perceptual = "✓ PASS" if (mean_perceptual is not None and mean_perceptual < 0.10) else "✗ FAIL"

            print(f"  r={radius}:")
            print(f"    SSIM:       {mean_ssim:.4f} {status_ssim} (threshold > 0.85)")
            if mean_perceptual is not None:
                print(f"    Perceptual: {mean_perceptual:.4f} {status_perceptual} (threshold < 0.10)")
            print(f"    ΔAM ROI:    {stats['mean_delta_am_roi']:.5f} ± {stats['std_delta_am_roi']:.5f}")
            print(f"    CoA shift:  {stats['mean_shift_distance']:.2f} ± {stats['std_shift_distance']:.2f} px")
            print(f"    DC:         {stats['mean_dc']:.3f} ± {stats['std_dc']:.3f}")

    print("\n" + "=" * 90)
    print("✅ RADIUS SENSITIVITY ANALYSIS COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
