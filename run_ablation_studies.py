"""
SCBA Ablation Study: TPS warp parameters and realism thresholds.

Generates summary tables comparing placeholder vs TPS warping and
parameter sensitivity for realism-aware counterfactual audits.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from tqdm import tqdm

from scba.cf.borders import apply_border_edit
from scba.cf.inpaint import repair_border_edit
from scba.data.loaders.jsrt import JSRTDataset
from scba.data.loaders.montgomery import MontgomeryDataset
from scba.data.loaders.shenzhen import ShenzhenDataset
from scba.data.transforms.standard import get_composed_transform, get_val_transforms
from scba.metrics.boundary import border_band_from_mask
from scba.metrics.cf_consistency import compute_cf_metrics
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
    "shenzhen": {
        "loader": ShenzhenDataset,
        "model_path": "runs/shenzhen_unet_baseline_20260201_000000.pt",
        "model_path_env": "SCBA_SHENZHEN_MODEL_PATH",
        "data_root": None,
        "data_root_env": "SCBA_SHENZHEN_ROOT",
        "dataset_name": "Shenzhen",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run SCBA ablation studies.")
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
        help="Dataset to evaluate.",
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
        "--output-dir",
        type=str,
        default="experiments/results/scba_ablation",
        help="Directory to store ablation outputs.",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Optional limit on number of test samples.",
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


def get_ablation_configs():
    return [
        {"name": "placeholder_warp", "warp_mode": "none"},
        {
            "name": "tps_base",
            "warp_mode": "tps",
            "warp_points": 160,
            "warp_smooth": 1.0,
            "min_ssim": 0.60,
        },
        {
            "name": "tps_low_smooth",
            "warp_mode": "tps",
            "warp_points": 160,
            "warp_smooth": 0.5,
            "min_ssim": 0.60,
        },
        {
            "name": "tps_high_smooth",
            "warp_mode": "tps",
            "warp_points": 160,
            "warp_smooth": 3.0,
            "min_ssim": 0.60,
        },
        {
            "name": "tps_low_ssim",
            "warp_mode": "tps",
            "warp_points": 160,
            "warp_smooth": 1.0,
            "min_ssim": 0.50,
        },
        {
            "name": "tps_high_ssim",
            "warp_mode": "tps",
            "warp_points": 160,
            "warp_smooth": 1.0,
            "min_ssim": 0.70,
        },
        {
            "name": "tps_low_points",
            "warp_mode": "tps",
            "warp_points": 80,
            "warp_smooth": 1.0,
            "min_ssim": 0.60,
        },
    ]


def summarize_metrics(records):
    def mean(values):
        return float(np.nanmean(values)) if values else None

    if not records:
        return (
            {
                "mean_delta_am_roi": None,
                "mean_shift_distance": None,
                "mean_dc": None,
                "mean_deletion_auc": None,
                "mean_insertion_auc": None,
                "mean_robustness_corr": None,
                "mean_perceptual_distance": None,
                "mean_boundary_deletion_auc": None,
                "mean_boundary_insertion_auc": None,
                "mean_ssim": None,
                "mean_intensity_delta": None,
                "n_experiments": 0,
                "n_patients": 0,
            },
            {},
        )

    grouped = {}
    for record in records:
        grouped.setdefault(record["patient_id"], []).append(record)

    def patient_values(key):
        values = []
        for rows in grouped.values():
            vals = [r.get(key) for r in rows if r.get(key) is not None]
            if vals:
                values.append(float(np.nanmean(vals)))
        return values

    keys = [
        "delta_am_roi",
        "shift_distance",
        "directional_consistency",
        "deletion_auc",
        "insertion_auc",
        "boundary_deletion_auc",
        "boundary_insertion_auc",
        "robustness_corr",
        "perceptual_distance",
        "ssim",
        "mean_delta",
    ]

    patient_metrics = {}
    for pid, rows in grouped.items():
        patient_metrics[pid] = {}
        for key in keys:
            vals = [r.get(key) for r in rows if r.get(key) is not None]
            patient_metrics[pid][key] = float(np.nanmean(vals)) if vals else None

    summary = {
        "mean_delta_am_roi": mean(patient_values("delta_am_roi")),
        "mean_shift_distance": mean(patient_values("shift_distance")),
        "mean_dc": mean(patient_values("directional_consistency")),
        "mean_deletion_auc": mean(patient_values("deletion_auc")),
        "mean_insertion_auc": mean(patient_values("insertion_auc")),
        "mean_robustness_corr": mean(patient_values("robustness_corr")),
        "mean_perceptual_distance": mean(patient_values("perceptual_distance")),
        "mean_boundary_deletion_auc": mean(patient_values("boundary_deletion_auc")),
        "mean_boundary_insertion_auc": mean(patient_values("boundary_insertion_auc")),
        "mean_ssim": mean(patient_values("ssim")),
        "mean_intensity_delta": mean(patient_values("mean_delta")),
        "n_experiments": len(records),
        "n_patients": len(grouped),
    }
    return summary, patient_metrics


def main():
    args = parse_args()
    cfg = DATASET_CONFIG[args.dataset]
    DatasetClass = cfg["loader"]
    dataset_label = cfg["dataset_name"]

    from scba.utils.reproducibility import seed_everything
    # Scientific justification (R4): ablations compare subtle methodological
    # variants; fixed seeds prevent stochastic XAI baselines from dominating.
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

    results_file = output_dir / f"scba_ablation_{args.dataset}.json"
    table_file = output_dir / f"scba_ablation_{args.dataset}.csv"
    table_tex = output_dir / f"scba_ablation_{args.dataset}.tex"
    compare_file = output_dir / f"scba_warp_comparison_{args.dataset}.csv"
    compare_tex = output_dir / f"scba_warp_comparison_{args.dataset}.tex"

    print("=" * 90)
    print(f"SCBA ABLATION STUDY ({dataset_label})")
    print("=" * 90)
    print(f"✓ Model: {model_path}")
    print(f"✓ Data: {data_root}")
    print(f"✓ Output: {output_dir}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. "
            f"Pass --model-path or set {cfg.get('model_path_env')}."
        )

    model = UNet(n_channels=1, n_classes=2)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    target_size = (1024, 1024)
    val_transform = get_composed_transform(get_val_transforms(target_size))
    dataset = DatasetClass(
        data_root,
        split="test",
        transform=val_transform,
        return_patient_id=True,
        # Scientific justification (R2): write deterministic splits under the
        # experiment output directory, not into the dataset root.
        splits_path=str(output_dir / f"{args.dataset}_splits.csv"),
    )
    indices = list(range(len(dataset)))
    if args.limit_samples is not None:
        indices = indices[: max(args.limit_samples, 0)]
    if not indices:
        raise ValueError("No samples selected for ablation.")

    decoder_layers = get_decoder_target_layers(model, n_layers=2)
    method_kwargs = {"target_layers": [layer for _, layer in decoder_layers]}
    method_name = "multi_layer_cam"

    cf_configs = [
        {"radius_px": 2, "operation": "dilate"},
        {"radius_px": 3, "operation": "dilate"},
        {"radius_px": 2, "operation": "erode"},
    ]

    perceptual_metric = PerceptualDistance(device=str(device))
    ablation_configs = get_ablation_configs()

    all_results = {}
    if args.resume and results_file.exists():
        with open(results_file, "r") as f:
            previous = json.load(f)
        all_results = previous.get("ablation_results", {}) or {}
        print(f"✓ Resuming from {results_file}")
    total_start = time.time()

    for config in ablation_configs:
        records = []
        config_name = config["name"]
        if config_name in all_results:
            print(f"\n[Config] {config_name} (skip: already complete)")
            continue
        print(f"\n[Config] {config_name}")
        config_start = time.time()
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

        disable_tqdm = os.getenv("SCBA_DISABLE_TQDM", "").lower() in {"1", "true", "yes"}
        sample_iter = tqdm(indices, desc=f"Samples ({config_name})", disable=disable_tqdm)
        for loop_index, idx in enumerate(sample_iter, start=1):
            sample = dataset[idx]
            image_np = sample["image"].squeeze().numpy()
            image_tensor = sample["image"].unsqueeze(0).to(device)
            if disable_tqdm:
                print(f"[Config {config_name}] sample {loop_index}/{len(indices)} patient_id={sample.get('patient_id')}")

            # Scientific justification (M2): use model prediction for edits/ROIs.
            mask_gt_np = sample["mask"].squeeze().numpy().astype(np.uint8)
            with torch.no_grad():
                logits = model(image_tensor)
                pred = torch.argmax(logits, dim=1)
            mask_pred_np = pred.squeeze(0).detach().cpu().numpy().astype(np.uint8)
            mask_pred_np = (mask_pred_np > 0).astype(np.uint8)
            mask_source = str(args.mask_source)
            mask_np = mask_pred_np if mask_source == "pred" else mask_gt_np

            saliency_orig = explain(
                image_tensor,
                model,
                method=method_name,
                target_class=1,
                device=device,
                **method_kwargs,
            )

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
                explain_kwargs=method_kwargs,
            )
            robustness_boundary = saliency_robustness(
                model,
                image_tensor,
                saliency_orig.map,
                method=method_name,
                target_class=1,
                device=str(device),
                pixel_mask=border_band,
                explain_kwargs=method_kwargs,
            )
            robustness_full = saliency_robustness(
                model,
                image_tensor,
                saliency_orig.map,
                method=method_name,
                target_class=1,
                device=str(device),
                pixel_mask=None,
                explain_kwargs=method_kwargs,
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

            for cf_config in cf_configs:
                cf_key = f"{cf_config['operation']}_r{cf_config['radius_px']}"
                image_cf, mask_cf, roi_band, edit_meta = apply_border_edit(
                    image_np,
                    mask_np,
                    radius_px=cf_config["radius_px"],
                    operation=cf_config["operation"],
                    band_px=12,
                    area_budget=0.50,
                    seed=42,
                    warp_mode=config.get("warp_mode", "tps"),
                    warp_points=config.get("warp_points", 160),
                    warp_smooth=config.get("warp_smooth", 1.0),
                    min_ssim=config.get("min_ssim", 0.60),
                    return_metadata=True,
                )

                record_poisson(edit_meta, method_name)

                if roi_band.sum() == 0:
                    continue

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
                    **method_kwargs,
                )

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
                    **method_kwargs,
                )

                metrics = compute_cf_metrics(
                    saliency_orig.map,
                    saliency_cf.map,
                    saliency_repair.map,
                    roi_band,
                )

                try:
                    perceptual_distance = perceptual_metric.compute(image_np, image_cf)
                except Exception:
                    perceptual_distance = None

                validation = edit_meta.get("validation", {}) if edit_meta else {}

                records.append(
                    {
                        "patient_id": sample.get("patient_id"),
                        "cf_key": cf_key,
                        "delta_am_roi": metrics["delta_am_roi"],
                        "shift_distance": metrics["shift_distance"],
                        "directional_consistency": metrics["directional_consistency"],
                        "deletion_auc": baseline_metrics.get("deletion_auc"),
                        "insertion_auc": baseline_metrics.get("insertion_auc"),
                        "boundary_deletion_auc": boundary_metrics.get("deletion_auc"),
                        "boundary_insertion_auc": boundary_metrics.get("insertion_auc"),
                        "robustness_corr": baseline_metrics.get("robustness_corr_mean"),
                        "robustness_corr_boundary": baseline_metrics.get("robustness_corr_mean_boundary"),
                        "robustness_corr_full": baseline_metrics.get("robustness_corr_mean_full"),
                        "perceptual_distance": perceptual_distance,
                        "ssim": validation.get("ssim"),
                        "mean_delta": validation.get("mean_delta"),
                    }
                )

                if device.type == "cuda":
                    del image_cf_tensor, saliency_cf, image_repair_tensor, saliency_repair
                    torch.cuda.empty_cache()

        summary, patient_metrics = summarize_metrics(records)
        all_results[config_name] = {
            "config": config,
            "summary": summary,
            "patient_metrics": patient_metrics,
            "poisson_blend": poisson_stats,
            # Scientific justification: persist per-experiment records so that
            # summary statistics and manuscript tables can be independently
            # recomputed and audited from a single JSON artifact.
            "records": records,
        }
        elapsed = (time.time() - config_start) / 60.0
        print(f"✓ {config_name} completed in {elapsed:.1f} min")

    total_elapsed = (time.time() - total_start) / 60.0
    print(f"\n✓ Ablation finished in {total_elapsed:.1f} min")

    with open(results_file, "w") as f:
        json.dump(
            {
                "dataset": dataset_label,
                "seed": int(args.seed),
                "model_path": str(model_path),
                "n_samples": len(indices),
                "method": method_name,
                "mask_source": str(args.mask_source),
                "ablation_results": all_results,
            },
            f,
            indent=2,
        )

    headers = [
        "config",
        "mean_delta_am_roi",
        "mean_shift_distance",
        "mean_dc",
        "mean_deletion_auc",
        "mean_insertion_auc",
        "mean_boundary_deletion_auc",
        "mean_boundary_insertion_auc",
        "mean_robustness_corr",
        "mean_perceptual_distance",
        "mean_ssim",
        "mean_intensity_delta",
        "n_experiments",
    ]
    def fmt(val, precision=4):
        return f"{val:.{precision}f}" if val is not None else "nan"

    with open(table_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for name, payload in all_results.items():
            summary = payload["summary"]
            writer.writerow([name] + [summary.get(h) for h in headers[1:]])

    with open(table_tex, "w") as f:
        f.write("\\begin{table}[t]\\centering\\small\\n")
        f.write("\\begin{tabular}{lrrrrrrrrrrr}\\hline\\n")
        f.write(
            "Config & $\\Delta$AM & CoA & DC & DelAUC & InsAUC & BDel & BIns & Rob & Perc & SSIM & $\\Delta I$ \\\\\\hline\\n"
        )
        for name, payload in all_results.items():
            s = payload["summary"]
            f.write(
                f"{name} & {fmt(s.get('mean_delta_am_roi'))} & {fmt(s.get('mean_shift_distance'), 2)} & "
                f"{fmt(s.get('mean_dc'), 2)} & {fmt(s.get('mean_deletion_auc'))} & {fmt(s.get('mean_insertion_auc'))} & "
                f"{fmt(s.get('mean_boundary_deletion_auc'))} & {fmt(s.get('mean_boundary_insertion_auc'))} & "
                f"{fmt(s.get('mean_robustness_corr'))} & {fmt(s.get('mean_perceptual_distance'))} & "
                f"{fmt(s.get('mean_ssim'), 3)} & {fmt(s.get('mean_intensity_delta'), 3)} \\\\\n"
            )
        f.write("\\hline\\end{tabular}\\n")
        f.write(f"\\caption{{SCBA TPS ablation study on {dataset_label}.}}\\n")
        f.write("\\end{table}\\n")

    compare_pairs = {"placeholder_warp": "tps_base"}
    with open(compare_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "placeholder_warp", "tps_base"])
        for metric in headers[1:]:
            placeholder = all_results.get("placeholder_warp", {}).get("summary", {}).get(metric)
            tps_base = all_results.get("tps_base", {}).get("summary", {}).get(metric)
            writer.writerow([metric, placeholder, tps_base])

    with open(compare_tex, "w") as f:
        f.write("\\begin{table}[t]\\centering\\small\\n")
        f.write("\\begin{tabular}{lrr}\\hline\\n")
        f.write("Metric & Placeholder & TPS \\\\\\hline\\n")
        for metric in headers[1:]:
            placeholder = all_results.get("placeholder_warp", {}).get("summary", {}).get(metric, 0)
            tps_base = all_results.get("tps_base", {}).get("summary", {}).get(metric, 0)
            f.write(f"{metric} & {fmt(placeholder)} & {fmt(tps_base)} \\\\\n")
        f.write("\\hline\\end{tabular}\\n")
        f.write(f"\\caption{{Placeholder vs. TPS warping comparison ({dataset_label}).}}\\n")
        f.write("\\end{table}\\n")

    print(f"✓ Ablation results saved to {results_file}")
    print(f"✓ Table saved to {table_file} and {table_tex}")
    print(f"✓ Comparison table saved to {compare_file} and {compare_tex}")


if __name__ == "__main__":
    main()
