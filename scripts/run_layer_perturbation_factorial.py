#!/usr/bin/env python3
"""
Full factorial study: layer (outc vs decoder) x perturbation (morph vs TPS).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from scba.cf.borders import apply_border_edit
from scba.cf.inpaint import repair_border_edit
from scba.data.loaders.jsrt import JSRTDataset
from scba.data.loaders.montgomery import MontgomeryDataset
from scba.data.transforms.standard import get_composed_transform, get_val_transforms
from scba.metrics.cf_consistency import compute_cf_metrics
from scba.metrics.statistical_tests import compare_all_methods, friedman_test
from scba.models.unet import UNet
from scba.xai.common import explain, get_decoder_target_layers


METHOD_LABELS = {
    "seg_grad_cam": "Seg-Grad-CAM",
    "hires_cam": "HiResCAM",
    "seg_xres_cam": "Seg-XRes-CAM",
    "grad_cam_pp": "Grad-CAM++",
}


def _set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def _load_model(model_path: Path, device: torch.device) -> UNet:
    model = UNet(n_channels=1, n_classes=2, bilinear=True).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def _load_dataset(dataset_key: str, data_root: Path, target_size: int, splits_path: Path):
    val_transform = get_composed_transform(get_val_transforms((target_size, target_size)))
    if dataset_key == "jsrt":
        return JSRTDataset(
            str(data_root),
            split="test",
            transform=val_transform,
            return_patient_id=True,
            splits_path=str(splits_path),
        )
    return MontgomeryDataset(
        str(data_root),
        split="test",
        transform=val_transform,
        return_patient_id=True,
        splits_path=str(splits_path),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run layer/perturbation factorial study.")
    parser.add_argument("--dataset", type=str, choices=["jsrt", "montgomery"], required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--target-size", type=int, default=1024)
    parser.add_argument("--mask-source", type=str, choices=["pred", "gt"], default="pred")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--warp-points", type=int, default=160)
    parser.add_argument("--warp-smooth", type=float, default=1.0)
    parser.add_argument("--min-ssim", type=float, default=0.60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data_root = args.data_root or os.environ.get(
        "SCBA_JSRT_ROOT" if args.dataset == "jsrt" else "SCBA_MONTGOMERY_ROOT"
    )
    model_path = args.model_path or os.environ.get(
        "SCBA_JSRT_MODEL_PATH" if args.dataset == "jsrt" else "SCBA_MONTGOMERY_MODEL_PATH"
    )
    if not data_root or not model_path:
        raise ValueError("Set SCBA_*_ROOT and SCBA_*_MODEL_PATH (or pass --data-root/--model-path).")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_determinism(args.seed)

    model = _load_model(Path(model_path), device)
    decoder_layers = get_decoder_target_layers(model, n_layers=2)
    if not decoder_layers:
        raise RuntimeError("No decoder layers found for factorial study.")
    decoder_name, decoder_layer = decoder_layers[-1]

    layer_conditions = {
        "outc": {"name": "outc.conv", "layer": model.outc.conv},
        "decoder": {"name": decoder_name, "layer": decoder_layer},
    }

    warp_conditions = {
        "morph": {"warp_mode": "none"},
        "tps": {
            "warp_mode": "tps",
            "warp_points": int(args.warp_points),
            "warp_smooth": float(args.warp_smooth),
            "min_ssim": float(args.min_ssim),
        },
    }

    dataset = _load_dataset(
        args.dataset,
        Path(data_root),
        args.target_size,
        output_dir / f"{args.dataset}_splits.csv",
    )

    total_samples = len(dataset)
    indices = list(range(total_samples))
    if args.limit_samples is not None:
        indices = indices[: max(args.limit_samples, 0)]
    n_samples = len(indices)
    if n_samples == 0:
        raise ValueError("No samples selected.")

    methods = list(METHOD_LABELS.keys())
    cf_configs = [
        {"radius_px": 2, "operation": "dilate", "desc": "Dilate r=2"},
        {"radius_px": 3, "operation": "dilate", "desc": "Dilate r=3"},
        {"radius_px": 2, "operation": "erode", "desc": "Erode r=2"},
    ]
    expected_cf_keys = [f"{c['operation']}_r{c['radius_px']}" for c in cf_configs]

    conditions: Dict[str, Dict] = {}
    for layer_key, layer_cfg in layer_conditions.items():
        for warp_key, warp_cfg in warp_conditions.items():
            cond_key = f"{layer_key}|{warp_key}"
            conditions[cond_key] = {
                "layer_key": layer_key,
                "layer_name": layer_cfg["name"],
                "warp_key": warp_key,
                "warp_mode": warp_cfg["warp_mode"],
                "warp_config": warp_cfg,
                "xai_methods": METHOD_LABELS,
                "detailed_results": {},
                "audit": {
                    "failure_counts": {
                        "explain_original": 0,
                        "cf_edit": 0,
                        "explain_cf": 0,
                        "explain_repair": 0,
                        "roi_empty": 0,
                    },
                    "failure_log": [],
                    "poisson_blend": {
                        "total_edits": 0,
                        "poisson_requested": 0,
                        "poisson_used": 0,
                        "poisson_failed": 0,
                        "validation_fallback": 0,
                        "by_method": {},
                    },
                },
            }

    def record_failure(cond_key, stage, patient_id, method_name, cf_key=None, error=None):
        audit = conditions[cond_key]["audit"]
        audit["failure_counts"][stage] = audit["failure_counts"].get(stage, 0) + 1
        if len(audit["failure_log"]) < 50:
            audit["failure_log"].append(
                {
                    "stage": stage,
                    "patient_id": patient_id,
                    "method": method_name,
                    "cf_key": cf_key,
                    "error": str(error) if error is not None else None,
                }
            )

    def record_poisson(cond_key, method_name, edit_meta):
        if not edit_meta:
            return
        blend = edit_meta.get("blend", {}) if isinstance(edit_meta, dict) else {}
        requested = blend.get("requested")
        used = blend.get("used")
        validation_fallback = bool(blend.get("validation_fallback", False))

        stats = conditions[cond_key]["audit"]["poisson_blend"]
        stats["total_edits"] += 1
        if requested == "poisson":
            stats["poisson_requested"] += 1
        if used == "poisson":
            stats["poisson_used"] += 1
        if requested == "poisson" and used != "poisson":
            stats["poisson_failed"] += 1
        if validation_fallback:
            stats["validation_fallback"] += 1

        by_method = stats["by_method"].setdefault(
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

    for idx in indices:
        sample = dataset[idx]
        patient_id = sample["patient_id"]
        image_np = sample["image"].squeeze().numpy()
        image_tensor = sample["image"].unsqueeze(0).to(device)

        mask_gt_np = sample["mask"].squeeze().numpy().astype(np.uint8)
        if args.mask_source == "pred":
            with torch.no_grad():
                logits = model(image_tensor)
                pred = torch.argmax(logits, dim=1)
            mask_pred_np = pred.squeeze(0).detach().cpu().numpy().astype(np.uint8)
            mask_pred_np = (mask_pred_np > 0).astype(np.uint8)
            mask_np = mask_pred_np
        else:
            mask_np = mask_gt_np

        for layer_key, layer_cfg in layer_conditions.items():
            target_layer = layer_cfg["layer"]
            saliency_orig_by_method = {}
            for method in methods:
                try:
                    saliency_orig_by_method[method] = explain(
                        image_tensor,
                        model,
                        method=method,
                        target_class=1,
                        device=device,
                        target_layer=target_layer,
                    )
                except Exception as e:
                    for warp_key in warp_conditions:
                        cond_key = f"{layer_key}|{warp_key}"
                        record_failure(cond_key, "explain_original", patient_id, method, error=e)
                    continue

            for warp_key, warp_cfg in warp_conditions.items():
                cond_key = f"{layer_key}|{warp_key}"
                cond_results = conditions[cond_key]["detailed_results"].setdefault(patient_id, {})
                cond_results.setdefault(
                    "_sample_metadata",
                    {
                        "mask_source": args.mask_source,
                        "mask_pred_sum": int(mask_np.sum()),
                        "mask_gt_sum": int(mask_gt_np.sum()),
                    },
                )

                for method in methods:
                    saliency_orig = saliency_orig_by_method.get(method)
                    if saliency_orig is None:
                        continue
                    method_results = cond_results.get(method, {})

                    for cf_config in cf_configs:
                        cf_key = f"{cf_config['operation']}_r{cf_config['radius_px']}"
                        try:
                            image_cf, mask_cf, roi_band, edit_meta = apply_border_edit(
                                image_np,
                                mask_np.astype(np.uint8),
                                radius_px=cf_config["radius_px"],
                                operation=cf_config["operation"],
                                band_px=12,
                                area_budget=0.50,
                                seed=42,
                                warp_mode=warp_cfg.get("warp_mode", "tps"),
                                warp_points=warp_cfg.get("warp_points", 160),
                                warp_smooth=warp_cfg.get("warp_smooth", 1.0),
                                min_ssim=warp_cfg.get("min_ssim", 0.60),
                                return_metadata=True,
                            )
                            record_poisson(cond_key, method, edit_meta)
                        except Exception as e:
                            record_failure(cond_key, "cf_edit", patient_id, method, cf_key=cf_key, error=e)
                            continue

                        if roi_band.sum() == 0:
                            record_failure(cond_key, "roi_empty", patient_id, method, cf_key=cf_key)
                            continue

                        image_cf_tensor = (
                            torch.from_numpy(image_cf.astype(np.float32))
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .to(device)
                        )
                        try:
                            saliency_cf = explain(
                                image_cf_tensor,
                                model,
                                method=method,
                                target_class=1,
                                device=device,
                                target_layer=target_layer,
                            )
                        except Exception as e:
                            record_failure(cond_key, "explain_cf", patient_id, method, cf_key=cf_key, error=e)
                            continue

                        image_repair = repair_border_edit(image_cf, image_np, roi_band)
                        image_repair_tensor = (
                            torch.from_numpy(image_repair.astype(np.float32))
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .to(device)
                        )
                        try:
                            saliency_repair = explain(
                                image_repair_tensor,
                                model,
                                method=method,
                                target_class=1,
                                device=device,
                                target_layer=target_layer,
                            )
                        except Exception as e:
                            record_failure(cond_key, "explain_repair", patient_id, method, cf_key=cf_key, error=e)
                            continue

                        metrics = compute_cf_metrics(
                            saliency_orig.map,
                            saliency_cf.map,
                            saliency_repair.map,
                            roi_band,
                        )

                        method_results[cf_key] = {
                            "config": cf_config,
                            "metrics": metrics,
                            "roi_pixels": int(roi_band.sum()),
                            "area_change": float(abs(mask_cf.sum() - mask_np.sum()) / max(mask_np.sum(), 1)),
                        }

                        if device.type == "cuda":
                            del image_cf_tensor, image_repair_tensor, saliency_cf, saliency_repair
                            torch.cuda.empty_cache()

                    if method_results:
                        cond_results[method] = method_results

            if device.type == "cuda":
                del saliency_orig_by_method
                torch.cuda.empty_cache()

    # Aggregate summaries/statistics per condition
    for cond_key, cond in conditions.items():
        detailed = cond["detailed_results"]
        per_method_patient = {m: {} for m in methods}
        complete = {m: set() for m in methods}

        for patient_id, patient_results in detailed.items():
            for method in methods:
                method_results = patient_results.get(method)
                if not method_results or not all(k in method_results for k in expected_cf_keys):
                    continue
                delta_vals = []
                shift_vals = []
                dc_vals = []
                for cf_key in expected_cf_keys:
                    metrics = method_results[cf_key]["metrics"]
                    delta_vals.append(float(metrics["delta_am_roi"]))
                    shift_vals.append(float(metrics["shift_distance"]))
                    dc_vals.append(float(metrics["directional_consistency"]))
                per_method_patient[method][patient_id] = {
                    "delta_am_roi": float(np.mean(delta_vals)),
                    "shift_distance": float(np.mean(shift_vals)),
                    "directional_consistency": float(np.mean(dc_vals)),
                }
                complete[method].add(patient_id)

        common_patients = sorted(set.intersection(*(complete[m] for m in methods)))
        if not common_patients:
            raise RuntimeError(f"No complete-case patients for condition {cond_key}.")

        summary = {}
        for method in methods:
            values = [per_method_patient[method][pid]["delta_am_roi"] for pid in common_patients]
            shifts = [per_method_patient[method][pid]["shift_distance"] for pid in common_patients]
            dc = [per_method_patient[method][pid]["directional_consistency"] for pid in common_patients]
            summary[method] = {
                "mean_delta_am_roi": float(np.mean(values)),
                "std_delta_am_roi": float(np.std(values, ddof=1)),
                "median_delta_am_roi": float(np.median(values)),
                "mean_shift_distance": float(np.mean(shifts)),
                "std_shift_distance": float(np.std(shifts, ddof=1)),
                "median_shift_distance": float(np.median(shifts)),
                "mean_dc": float(np.mean(dc)),
                "std_dc": float(np.std(dc, ddof=1)),
                "median_dc": float(np.median(dc)),
                "n_patients": int(len(common_patients)),
                "n_experiments": int(len(common_patients) * len(expected_cf_keys)),
                "patient_ids_used": common_patients,
            }

        def _metric_arrays(metric_key: str) -> Dict[str, np.ndarray]:
            out = {}
            for method in methods:
                label = METHOD_LABELS[method]
                out[label] = np.array(
                    [per_method_patient[method][pid][metric_key] for pid in common_patients],
                    dtype=float,
                )
            return out

        cond["summary"] = summary
        cond["statistics"] = {
            "delta_am_roi": compare_all_methods(_metric_arrays("delta_am_roi"), metric_name="ΔAM-ROI", seed=args.seed),
            "shift_distance": compare_all_methods(_metric_arrays("shift_distance"), metric_name="CoA Shift (pixels)", seed=args.seed),
            "directional_consistency": compare_all_methods(
                _metric_arrays("directional_consistency"), metric_name="Directional Consistency", seed=args.seed
            ),
            "friedman_tests": {
                "delta_am_roi": friedman_test(_metric_arrays("delta_am_roi")),
                "shift_distance": friedman_test(_metric_arrays("shift_distance")),
                "directional_consistency": friedman_test(_metric_arrays("directional_consistency")),
            },
        }

    payload = {
        "config": {
            "dataset": args.dataset,
            "seed": args.seed,
            "model_path": str(model_path),
            "data_root": str(data_root),
            "n_samples_evaluated": n_samples,
            "mask_source": args.mask_source,
            "methods": methods,
            "method_labels": METHOD_LABELS,
            "cf_configs": cf_configs,
            "target_size": int(args.target_size),
            "layers": {k: v["name"] for k, v in layer_conditions.items()},
            "warp_conditions": warp_conditions,
        },
        "conditions": conditions,
    }

    output_path = output_dir / f"factorial_layer_perturbation_{args.dataset}.json"
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
