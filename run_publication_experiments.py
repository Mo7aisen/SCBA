"""
PUBLICATION-READY SCBA EXPERIMENTS
===================================

Comprehensive analysis with:
- All test samples for statistical power
- 4 CAM methods by default (optional extended suite via --method-suite full)
- Bootstrap confidence intervals
- Statistical significance testing
- Professional visualization and tables

Professional medical imaging research standard.
"""

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path
import time

# Scientific justification: disable albumentations' network version checks by
# default to keep runs offline-safe and logs clean in reviewer environments.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from scba.cf.borders import apply_border_edit
from scba.cf.inpaint import repair_border_edit
from scba.data.loaders.jsrt import JSRTDataset
from scba.data.loaders.montgomery import MontgomeryDataset
from scba.data.loaders.shenzhen import ShenzhenDataset
from scba.data.transforms.standard import get_composed_transform, get_val_transforms
from scba.metrics.cf_consistency import compute_cf_metrics
from scba.metrics.boundary import border_band_from_mask
from scba.metrics.faithfulness import deletion_insertion_auc
from scba.metrics.perceptual import PerceptualDistance
from scba.metrics.robustness import saliency_robustness
from scba.metrics.statistical_tests import (
    compare_all_methods,
    generate_comparison_table,
    friedman_test,
    format_p_value,
)
from scba.models.unet import UNet
from scba.xai.common import explain, get_decoder_target_layers


DATASET_CONFIG = {
    "jsrt": {
        "loader": JSRTDataset,
        "model_path": "runs/jsrt_unet_baseline_20251101_203253.pt",
        "model_path_env": "SCBA_JSRT_MODEL_PATH",
        "data_root": None,
        "data_root_env": "SCBA_JSRT_ROOT",
        "output_dir": "experiments/results/scba_publication",
        "results_filename": "scba_publication_results.json",
        "figure_filename": "scba_publication_figure.png",
        "dataset_name": "JSRT",
    },
    "montgomery": {
        "loader": MontgomeryDataset,
        "model_path": "runs/montgomery_unet_baseline_20251101_203253.pt",
        "model_path_env": "SCBA_MONTGOMERY_MODEL_PATH",
        "data_root": None,
        "data_root_env": "SCBA_MONTGOMERY_ROOT",
        "output_dir": "experiments/results/scba_publication_montgomery",
        "results_filename": "scba_publication_montgomery_results.json",
        "figure_filename": "scba_publication_montgomery_figure.png",
        "dataset_name": "Montgomery",
    },
    "shenzhen": {
        "loader": ShenzhenDataset,
        "model_path": "runs/shenzhen_unet_baseline_20260201_000000.pt",
        "model_path_env": "SCBA_SHENZHEN_MODEL_PATH",
        "data_root": None,
        "data_root_env": "SCBA_SHENZHEN_ROOT",
        "output_dir": "experiments/results/scba_publication_shenzhen",
        "results_filename": "scba_publication_shenzhen_results.json",
        "figure_filename": "scba_publication_shenzhen_figure.png",
        "dataset_name": "Shenzhen",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run publication-grade SCBA experiments."
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
        "--method-suite",
        type=str,
        default="cam",
        choices=["cam", "phase1", "full"],
        help=(
            "Method suite to run: cam (default), phase1 (cam + Integrated Gradients + random control), "
            "or full (adds perturbation-based methods)."
        ),
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
        "--target-size",
        type=int,
        default=1024,
        help="Resize images to target_size×target_size (publication default: 1024).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store experiment outputs.",
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
        "--rise-masks",
        type=int,
        default=1000,
        help="Number of masks for RISE when --method-suite full (default: 1000).",
    )
    parser.add_argument(
        "--lime-samples",
        type=int,
        default=500,
        help="Number of perturbation samples for LIME when --method-suite full (default: 500).",
    )
    parser.add_argument(
        "--shap-samples",
        type=int,
        default=1000,
        help="Number of samples for KernelSHAP when --method-suite full (default: 1000).",
    )
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=50,
        help="Steps for Integrated Gradients when --method-suite phase1/full (default: 50).",
    )
    parser.add_argument(
        "--ig-baseline",
        type=str,
        default="black",
        choices=["black", "gaussian", "blur"],
        help="Integrated Gradients baseline when --method-suite phase1/full (default: black).",
    )
    parser.add_argument(
        "--baseline-metrics",
        type=str,
        default="auto",
        choices=["auto", "all", "cam-only", "none"],
        help=(
            "Which methods should compute auxiliary baselines (AUC deletion/insertion + robustness). "
            "auto: cam->all, phase1->cam-only, full->all."
        ),
    )
    parser.add_argument(
        "--occlusion-patch",
        type=int,
        default=32,
        help="Occlusion patch size (pixels) when --method-suite full (default: 32).",
    )
    parser.add_argument(
        "--occlusion-stride",
        type=int,
        default=16,
        help="Occlusion stride (pixels) when --method-suite full (default: 16).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing results file if available.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Write partial results JSON every N samples (0 disables).",
    )
    parser.add_argument(
        "--method-filter",
        type=str,
        default=None,
        help=(
            "Optional comma-separated list of method keys to run (e.g., "
            "'multi_layer_cam,integrated_gradients'). Useful for focused comparisons."
        ),
    )
    parser.add_argument(
        "--mask-source",
        type=str,
        default="pred",
        choices=["pred", "gt"],
        help="Mask source for counterfactual edits/ROIs: pred (default) or gt.",
    )
    return parser.parse_args()

def _write_comparison_csvs(
    comparison: dict,
    summary_csv: Path,
    pairwise_csv: Path,
) -> None:
    """
    Write machine-readable tables for manuscript cross-validation.

    Scientific justification: a printed console table is not a reproducible
    artifact. Persisting CSV tables allows (i) independent verification that
    manuscript tables match the exact computed statistics and (ii) automated
    regression checks when the pipeline is rerun.
    """
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "n", "mean", "std", "median", "ci_lower", "ci_upper"])
        for method, stats in comparison["summary"].items():
            writer.writerow(
                [
                    method,
                    stats["n"],
                    stats["mean"],
                    stats["std"],
                    stats["median"],
                    stats["ci_lower"],
                    stats["ci_upper"],
                ]
            )

    with open(pairwise_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method_1",
                "method_2",
                "mean_diff",
                "t_statistic",
                "t_p_value",
                "wilcoxon_statistic",
                "wilcoxon_p_value",
                "cohens_d",
                "significant_parametric",
                "significant_nonparametric",
                "bonferroni_alpha",
                "significant_bonferroni_t",
                "significant_bonferroni_w",
            ]
        )
        for comp in comparison["pairwise_comparisons"]:
            writer.writerow(
                [
                    comp["method_1"],
                    comp["method_2"],
                    comp["mean_diff"],
                    comp["t_statistic"],
                    comp["t_p_value"],
                    comp["wilcoxon_statistic"],
                    comp["wilcoxon_p_value"],
                    comp["cohens_d"],
                    comp["significant_parametric"],
                    comp["significant_nonparametric"],
                    comparison["bonferroni_alpha"],
                    comp.get("significant_bonferroni_t"),
                    comp.get("significant_bonferroni_w"),
                ]
            )


def main():
    args = parse_args()
    dataset_key = args.dataset
    cfg = DATASET_CONFIG[dataset_key]
    dataset_label = cfg["dataset_name"]
    DatasetClass = cfg["loader"]

    print("=" * 90)
    print("PUBLICATION-READY SCBA: COMPREHENSIVE STATISTICAL ANALYSIS")
    print(f"Dataset: {dataset_label}")
    print("=" * 90)

    # Configuration
    # Scientific justification (R4): enforce a single global seed so that
    # sampling-based XAI baselines and bootstrap CIs are exactly reproducible.
    from scba.utils.reproducibility import seed_everything
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
    output_dir = Path(args.output_dir or cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / cfg["results_filename"]
    figure_filename = cfg["figure_filename"]
    figure_path = output_dir / figure_filename
    checkpoint_every = int(args.checkpoint_every or 0)

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
    print("\n[1/7] Loading trained model...")
    model = UNet(n_channels=1, n_classes=2)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint.get("epoch", "unknown")
        # Scientific justification: checkpoints historically used either
        # `best_score` or `best_dice`; support both to avoid ambiguity in logs.
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

    # Load ALL test samples
    print("\n[2/7] Loading ALL test samples...")
    # Scientific justification: the manuscript experiments use 1024×1024.
    # Keep this default, but allow a smaller target for smoke-testing wiring.
    target_size = (int(args.target_size), int(args.target_size))
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
    if n_samples == total_samples:
        print(f"✓ Using all {n_samples} test samples for {dataset_label}")
    else:
        print(f"✓ Using {n_samples}/{total_samples} test samples for {dataset_label}")
    if n_samples == 0:
        raise ValueError("No samples selected for evaluation.")

    decoder_layers = get_decoder_target_layers(model, n_layers=2)
    decoder_layer_names = [name for name, _ in decoder_layers]
    primary_layer = decoder_layers[-1][1]
    method_kwargs = {
        "multi_layer_cam": {"target_layers": [layer for _, layer in decoder_layers]},
        "layer_cam": {"target_layer": primary_layer},
        "hires_cam": {"target_layer": primary_layer},
        "grad_cam_pp": {"target_layer": primary_layer},
    }
    method_config = {
        "multi_layer_cam": {"target_layers": decoder_layer_names},
        "layer_cam": {"target_layer": decoder_layer_names[-1]},
        "hires_cam": {"target_layer": decoder_layer_names[-1]},
        "grad_cam_pp": {"target_layer": decoder_layer_names[-1]},
    }

    # XAI methods: decoder-focused CAMs to avoid 1x1 head degeneracy
    xai_methods = {
        "multi_layer_cam": f"Multi-Layer LayerCAM ({', '.join(decoder_layer_names)})",
        "layer_cam": f"LayerCAM ({decoder_layer_names[-1]})",
        "hires_cam": f"HiResCAM ({decoder_layer_names[-1]})",
        "grad_cam_pp": f"Grad-CAM++ ({decoder_layer_names[-1]})",
    }
    if args.method_suite in {"phase1", "full"}:
        xai_methods.update(
            {
                "gradient": "Gradient",
                "input_x_gradient": "Input×Gradient",
                "random_map": "Random Map (Negative Control)",
            }
        )
        method_kwargs.update(
            {
                "gradient": {},
                "input_x_gradient": {},
                "random_map": {"seed": int(args.seed)},
            }
        )
        method_config.update(
            {
                "gradient": {},
                "input_x_gradient": {},
                "random_map": {"seed": int(args.seed)},
            }
        )
    if args.method_suite == "full":
        xai_methods.update(
            {
                "rise": "RISE",
                "occlusion": "Occlusion",
                "lime": "LIME",
                "shap": "KernelSHAP",
            }
        )
        method_kwargs.update(
            {
                "rise": {"n_masks": int(args.rise_masks), "seed": int(args.seed)},
                "occlusion": {
                    "patch_size": int(args.occlusion_patch),
                    "stride": int(args.occlusion_stride),
                },
                "lime": {"n_samples": int(args.lime_samples), "seed": int(args.seed)},
                "shap": {"n_samples": int(args.shap_samples), "seed": int(args.seed)},
            }
        )
        method_config.update(
            {
                "rise": {"n_masks": int(args.rise_masks), "seed": int(args.seed)},
                "occlusion": {
                    "patch_size": int(args.occlusion_patch),
                    "stride": int(args.occlusion_stride),
                },
                "lime": {"n_samples": int(args.lime_samples), "seed": int(args.seed)},
                "shap": {"n_samples": int(args.shap_samples), "seed": int(args.seed)},
            }
        )
    if os.getenv("SCBA_QUICK", "").lower() in {"1", "true", "yes"}:
        # Scientific justification: quick mode is a wiring smoke test only.
        # Restricting methods/CFs reduces runtime without changing core code paths.
        xai_methods = {"layer_cam": xai_methods["layer_cam"]}
        method_kwargs = {"layer_cam": method_kwargs["layer_cam"]}
        method_config = {"layer_cam": method_config["layer_cam"]}

    if args.method_filter:
        keep = [m.strip() for m in str(args.method_filter).split(",") if m.strip()]
        unknown = [m for m in keep if m not in xai_methods]
        if unknown:
            raise ValueError(f"--method-filter has unknown methods: {unknown}. Available: {list(xai_methods.keys())}")
        xai_methods = {m: xai_methods[m] for m in keep}
        method_kwargs = {m: method_kwargs.get(m, {}) for m in keep}
        method_config = {m: method_config.get(m, {}) for m in keep}
    print(f"✓ Testing {len(xai_methods)} XAI methods")

    cam_methods = {"multi_layer_cam", "layer_cam", "hires_cam", "grad_cam_pp"}
    baseline_mode = str(args.baseline_metrics)
    if baseline_mode == "auto":
        if args.method_suite == "phase1":
            baseline_mode = "cam-only"
        else:
            baseline_mode = "all"

    def should_compute_baseline_metrics(method_name: str) -> bool:
        # Scientific justification: auxiliary baselines (deletion/insertion AUC,
        # robustness) are very compute-heavy for integrated-gradient explainers.
        # For Phase 1, we focus on counterfactual consistency endpoints; keep
        # baselines enabled for CAMs and allow overriding via --baseline-metrics.
        if method_name == "random_map":
            return False
        if baseline_mode == "none":
            return False
        if baseline_mode == "all":
            return True
        if baseline_mode == "cam-only":
            return method_name in cam_methods
        raise ValueError(f"Unknown baseline mode: {baseline_mode}")

    # Counterfactual configurations
    cf_configs = [
        {"radius_px": 2, "operation": "dilate", "desc": "Dilate r=2"},
        {"radius_px": 3, "operation": "dilate", "desc": "Dilate r=3"},
        {"radius_px": 2, "operation": "erode", "desc": "Erode r=2"},
    ]
    if os.getenv("SCBA_QUICK", "").lower() in {"1", "true", "yes"}:
        cf_configs = [cf_configs[0]]
    print(f"✓ Testing {len(cf_configs)} counterfactual perturbations")

    total_experiments = n_samples * len(xai_methods) * len(cf_configs)
    print(f"\n✓ Total experiments: {total_experiments} ({n_samples} samples × {len(xai_methods)} methods × {len(cf_configs)} CFs)")
    print(f"✓ Estimated time: ~{total_experiments * 0.5 / 60:.1f} minutes on GPU")

    # Set up perceptual metric (radiologist-proxy realism)
    perceptual_metric = PerceptualDistance(device=str(device))

    def _sync_cuda() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    timing = {
        "by_method": {
            method: {
                "explain_orig_sec_total": 0.0,
                "explain_cf_sec_total": 0.0,
                "explain_repair_sec_total": 0.0,
                "n_explain_orig": 0,
                "n_explain_cf": 0,
                "n_explain_repair": 0,
            }
            for method in xai_methods.keys()
        },
        "cf_edit_sec_total": 0.0,
        "n_cf_edits": 0,
        "repair_image_sec_total": 0.0,
        "n_repairs": 0,
        "baseline_metrics_sec_total": 0.0,
        "n_baseline_metrics": 0,
    }

    # Run experiments
    print("\n[3/7] Running comprehensive SCBA experiments...")
    print("=" * 90)

    all_results = {}
    if args.resume and results_file.exists():
        with open(results_file, "r") as f:
            previous = json.load(f)
        all_results = previous.get("detailed_results", {})
        print(f"✓ Resuming from {results_file}")
    total_start_time = time.time()

    expected_cf_keys = [
        f"{cfg['operation']}_r{cfg['radius_px']}" for cfg in cf_configs
    ]
    failure_counts = {
        "explain_original": 0,
        "cf_edit": 0,
        "explain_cf": 0,
        "explain_repair": 0,
        "roi_empty": 0,
        "perceptual": 0,
    }
    failure_log = []
    poisson_stats = {
        "total_edits": 0,
        "poisson_requested": 0,
        "poisson_used": 0,
        "poisson_failed": 0,
        "validation_fallback": 0,
        "by_method": {},
    }

    def record_failure(stage, patient_id, method_name, cf_key=None, error=None):
        failure_counts[stage] = failure_counts.get(stage, 0) + 1
        if len(failure_log) < 50:
            failure_log.append(
                {
                    "stage": stage,
                    "patient_id": patient_id,
                    "method": method_name,
                    "cf_key": cf_key,
                    "error": str(error) if error is not None else None,
                }
            )

    def record_poisson(edit_meta, method_name):
        if not edit_meta:
            return
        blend = edit_meta.get("blend", {}) if isinstance(edit_meta, dict) else {}
        requested = blend.get("requested")
        used = blend.get("used")
        poisson_ok = blend.get("poisson_ok")
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
    sample_iter = tqdm(test_indices, desc="Samples", disable=disable_tqdm)
    for loop_index, sample_idx in enumerate(sample_iter, start=1):
        sample = dataset[sample_idx]
        patient_id = sample["patient_id"]
        if disable_tqdm:
            print(f"[Sample {loop_index}/{n_samples}] patient_id={patient_id}")
        image_np = sample["image"].squeeze().numpy()
        image_tensor = sample["image"].unsqueeze(0).to(device)

        sample_results = all_results.get(patient_id, {})

        # Scientific justification (M2): counterfactual edits and ROI bands must
        # be defined from the model prediction (ŷ), not the ground-truth mask, to
        # avoid label leakage at test time and to match the manuscript protocol.
        mask_gt_np = sample["mask"].squeeze().numpy().astype(np.uint8)
        with torch.no_grad():
            logits = model(image_tensor)
            pred = torch.argmax(logits, dim=1)  # (1, H, W) for 2-logit UNet
        mask_pred_np = pred.squeeze(0).detach().cpu().numpy().astype(np.uint8)
        mask_pred_np = (mask_pred_np > 0).astype(np.uint8)

        inter = int((mask_pred_np & mask_gt_np).sum())
        denom = int(mask_pred_np.sum() + mask_gt_np.sum())
        dice_pred_gt = float((2 * inter) / (denom + 1e-8))

        mask_source = str(args.mask_source)
        mask_np = mask_pred_np if mask_source == "pred" else mask_gt_np

        sample_results["_sample_metadata"] = {
            "dice_pred_vs_gt": dice_pred_gt,
            "mask_pred_sum": int(mask_pred_np.sum()),
            "mask_gt_sum": int(mask_gt_np.sum()),
            "mask_source": mask_source,
        }

        for method_name, method_desc in xai_methods.items():
            if disable_tqdm:
                print(f"  [Method] {method_name}")
            try:
                method_results = sample_results.get(method_name, {})
                if method_results.get("baseline_metrics") and all(
                    key in method_results for key in expected_cf_keys
                ):
                    continue

                _sync_cuda()
                explain_orig_start = time.perf_counter()
                saliency_orig = explain(
                    image_tensor,
                    model,
                    method=method_name,
                    target_class=1,
                    device=device,
                    **method_kwargs.get(method_name, {}),
                )
                _sync_cuda()
                timing["by_method"][method_name]["explain_orig_sec_total"] += float(
                    time.perf_counter() - explain_orig_start
                )
                timing["by_method"][method_name]["n_explain_orig"] += 1
            except Exception as e:
                record_failure("explain_original", patient_id, method_name, error=e)
                continue

            if should_compute_baseline_metrics(method_name) and "baseline_metrics" not in method_results:
                baseline_start = time.perf_counter()
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
                _sync_cuda()
                baseline_elapsed = time.perf_counter() - baseline_start
                timing["baseline_metrics_sec_total"] += float(baseline_elapsed)
                timing["n_baseline_metrics"] += 1

            for cf_config in cf_configs:
                cf_key = f"{cf_config['operation']}_r{cf_config['radius_px']}"
                if disable_tqdm:
                    print(f"    [CF] {cf_key}")
                if cf_key in method_results:
                    continue

                mask_uint8 = mask_np.astype(np.uint8)

                try:
                    cf_start = time.perf_counter()
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
                    cf_elapsed = time.perf_counter() - cf_start
                    timing["cf_edit_sec_total"] += float(cf_elapsed)
                    timing["n_cf_edits"] += 1

                    record_poisson(edit_meta, method_name)

                    if roi_band.sum() == 0:
                        record_failure("roi_empty", patient_id, method_name, cf_key=cf_key)
                        continue

                    image_cf_tensor = (
                        torch.from_numpy(image_cf.astype(np.float32))
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(device)
                    )
                    try:
                        _sync_cuda()
                        explain_cf_start = time.perf_counter()
                        saliency_cf = explain(
                            image_cf_tensor,
                            model,
                            method=method_name,
                            target_class=1,
                            device=device,
                            **method_kwargs.get(method_name, {}),
                        )
                        _sync_cuda()
                        timing["by_method"][method_name]["explain_cf_sec_total"] += float(
                            time.perf_counter() - explain_cf_start
                        )
                        timing["by_method"][method_name]["n_explain_cf"] += 1
                    except Exception as e:
                        record_failure("explain_cf", patient_id, method_name, cf_key=cf_key, error=e)
                        continue

                    repair_start = time.perf_counter()
                    image_repair = repair_border_edit(image_cf, image_np, roi_band)
                    repair_elapsed = time.perf_counter() - repair_start
                    timing["repair_image_sec_total"] += float(repair_elapsed)
                    timing["n_repairs"] += 1
                    image_repair_tensor = (
                        torch.from_numpy(image_repair.astype(np.float32))
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(device)
                    )
                    try:
                        _sync_cuda()
                        explain_repair_start = time.perf_counter()
                        saliency_repair = explain(
                            image_repair_tensor,
                            model,
                            method=method_name,
                            target_class=1,
                            device=device,
                            **method_kwargs.get(method_name, {}),
                        )
                        _sync_cuda()
                        timing["by_method"][method_name]["explain_repair_sec_total"] += float(
                            time.perf_counter() - explain_repair_start
                        )
                        timing["by_method"][method_name]["n_explain_repair"] += 1
                    except Exception as e:
                        record_failure("explain_repair", patient_id, method_name, cf_key=cf_key, error=e)
                        continue

                    metrics = compute_cf_metrics(
                        saliency_orig.map,
                        saliency_cf.map,
                        saliency_repair.map,
                        roi_band,
                        mask_original=mask_uint8,
                        mask_perturbed=mask_cf,
                        operation=str(cf_config["operation"]),
                    )
                    try:
                        perceptual_distance = perceptual_metric.compute(image_np, image_cf)
                    except Exception as e:
                        record_failure("perceptual", patient_id, method_name, cf_key=cf_key, error=e)
                        perceptual_distance = None

                    mask_cf_sum = int(mask_cf.sum())
                    mask_orig_sum = int(mask_uint8.sum())
                    area_change = (
                        abs(mask_cf_sum - mask_orig_sum) / max(mask_orig_sum, 1)
                    )
                    method_results[cf_key] = {
                        "config": cf_config,
                        "metrics": metrics,
                        "roi_pixels": int(roi_band.sum()),
                        "area_change": float(area_change),
                        "edit_metadata": edit_meta,
                        "perceptual_distance": perceptual_distance,
                    }

                    if device.type == "cuda":
                        del image_cf_tensor, saliency_cf, image_repair_tensor, saliency_repair
                        torch.cuda.empty_cache()

                except Exception as e:
                    record_failure("cf_edit", patient_id, method_name, cf_key=cf_key, error=e)
                    continue

            if method_results:
                sample_results[method_name] = method_results

        if sample_results:
            all_results[patient_id] = sample_results
            if checkpoint_every and (loop_index % checkpoint_every == 0):
                checkpoint_payload = {
                    "dataset": dataset_label,
                    "partial": True,
                    "checkpoint_index": loop_index,
                    "total_samples": n_samples,
                    "timestamp_sec": time.time(),
                    "detailed_results": all_results,
                }
                with open(results_file, "w") as f:
                    json.dump(checkpoint_payload, f, indent=2)
                if disable_tqdm:
                    print(f"✓ Checkpoint saved to {results_file} ({loop_index}/{n_samples})", flush=True)

    total_time = time.time() - total_start_time
    print(f"\n✓ All experiments completed in {total_time/60:.1f} minutes")

    # Aggregate results
    print("\n[4/7] Aggregating results for statistical analysis...")
    method_aggregates = {}
    method_data = {}  # Store raw data for statistics

    # Scientific justification (M1): The unit of statistical inference is the
    # patient, not the (patient × counterfactual) pseudo-replicate. We therefore
    # aggregate per patient across CF variants and run all hypothesis tests and
    # bootstrapped CIs on patient-level values.
    per_method_patient_metrics = {m: {} for m in xai_methods.keys()}
    per_method_baseline = {m: {} for m in xai_methods.keys()}
    complete_patients_by_method = {m: set() for m in xai_methods.keys()}

    for patient_id, patient_results in all_results.items():
        for method_name in xai_methods.keys():
            if method_name not in patient_results:
                continue
            method_results = patient_results[method_name]
            if not all(key in method_results for key in expected_cf_keys):
                continue

            delta_vals = []
            shift_vals = []
            dc_vals = []
            perceptual_vals = []

            for cf_key in expected_cf_keys:
                cf = method_results[cf_key]
                metrics = cf.get("metrics", {})
                delta_vals.append(float(metrics["delta_am_roi"]))
                shift_vals.append(float(metrics["shift_distance"]))
                dc_vals.append(float(metrics["directional_consistency"]))
                if cf.get("perceptual_distance") is not None:
                    perceptual_vals.append(float(cf["perceptual_distance"]))

            per_method_patient_metrics[method_name][patient_id] = {
                "delta_am_roi": float(np.mean(delta_vals)),
                "shift_distance": float(np.mean(shift_vals)),
                "directional_consistency": float(np.mean(dc_vals)),
                "perceptual_distance": float(np.mean(perceptual_vals)) if perceptual_vals else float("nan"),
            }

            baseline = method_results.get("baseline_metrics")
            if baseline:
                per_method_baseline[method_name][patient_id] = baseline

            complete_patients_by_method[method_name].add(patient_id)

    common_patients = sorted(set.intersection(*(s for s in complete_patients_by_method.values())))
    if not common_patients:
        raise RuntimeError("No complete-case patients across all methods; cannot run paired patient-level statistics.")
    if len(common_patients) < 2:
        # Scientific justification: a single patient cannot support variance
        # estimates or hypothesis testing; this case occurs only in smoke tests
        # (e.g., --limit-samples 1). We still save per-sample JSON for wiring
        # validation, but we skip inferential statistics to avoid invalid claims.
        print(
            f"⚠ Only {len(common_patients)} patient available after complete-case filtering; "
            "skipping statistical tests and publication figure."
        )

    for method_name in xai_methods.keys():
        patient_metrics = per_method_patient_metrics[method_name]
        delta_am = np.array([patient_metrics[pid]["delta_am_roi"] for pid in common_patients], dtype=float)
        shifts = np.array([patient_metrics[pid]["shift_distance"] for pid in common_patients], dtype=float)
        dc = np.array([patient_metrics[pid]["directional_consistency"] for pid in common_patients], dtype=float)
        ddof = 1 if len(common_patients) > 1 else 0

        method_aggregates[method_name] = {
            "mean_delta_am_roi": float(np.mean(delta_am)),
            "std_delta_am_roi": float(np.std(delta_am, ddof=ddof)),
            "median_delta_am_roi": float(np.median(delta_am)),
            "mean_shift_distance": float(np.mean(shifts)),
            "std_shift_distance": float(np.std(shifts, ddof=ddof)),
            "median_shift_distance": float(np.median(shifts)),
            "mean_dc": float(np.mean(dc)),
            "std_dc": float(np.std(dc, ddof=ddof)),
            "median_dc": float(np.median(dc)),
            "n_patients": int(len(common_patients)),
            "n_experiments": int(len(common_patients) * len(expected_cf_keys)),
            "patient_ids_used": common_patients,
        }

        baseline_by_patient = per_method_baseline[method_name]
        if baseline_by_patient:
            deletion = np.array([baseline_by_patient.get(pid, {}).get("deletion_auc", np.nan) for pid in common_patients], dtype=float)
            insertion = np.array([baseline_by_patient.get(pid, {}).get("insertion_auc", np.nan) for pid in common_patients], dtype=float)
            robustness = np.array([baseline_by_patient.get(pid, {}).get("robustness_corr_mean", np.nan) for pid in common_patients], dtype=float)
            robustness_boundary = np.array(
                [baseline_by_patient.get(pid, {}).get("robustness_corr_mean_boundary", np.nan) for pid in common_patients],
                dtype=float,
            )
            robustness_full = np.array(
                [baseline_by_patient.get(pid, {}).get("robustness_corr_mean_full", np.nan) for pid in common_patients],
                dtype=float,
            )
            boundary_del = np.array([baseline_by_patient.get(pid, {}).get("boundary_deletion_auc", np.nan) for pid in common_patients], dtype=float)
            boundary_ins = np.array([baseline_by_patient.get(pid, {}).get("boundary_insertion_auc", np.nan) for pid in common_patients], dtype=float)

            method_aggregates[method_name].update(
                {
                    "mean_deletion_auc": float(np.nanmean(deletion)),
                    "mean_insertion_auc": float(np.nanmean(insertion)),
                    "mean_robustness_corr": float(np.nanmean(robustness)),
                    "mean_robustness_corr_boundary": float(np.nanmean(robustness_boundary)),
                    "mean_robustness_corr_full": float(np.nanmean(robustness_full)),
                    "mean_boundary_deletion_auc": float(np.nanmean(boundary_del)),
                    "mean_boundary_insertion_auc": float(np.nanmean(boundary_ins)),
                }
            )

        perceptual = np.array([patient_metrics[pid]["perceptual_distance"] for pid in common_patients], dtype=float)
        if np.isfinite(perceptual).any():
            method_aggregates[method_name]["mean_perceptual_distance"] = float(np.nanmean(perceptual))

        method_data[xai_methods[method_name]] = {
            "delta_am_roi": delta_am,
            "shift_distance": shifts,
            "directional_consistency": dc,
            "patient_ids": common_patients,
        }

    delta_am_comparison = None
    shift_comparison = None
    dc_comparison = None
    friedman_delta = None
    friedman_shift = None
    friedman_dc = None

    if len(common_patients) >= 2:
        # Statistical testing
        print("\n[5/7] Performing statistical tests...")
        print("=" * 90)

        # Extract data for each metric
        delta_am_data = {name: data["delta_am_roi"] for name, data in method_data.items()}
        shift_data = {name: data["shift_distance"] for name, data in method_data.items()}
        dc_data = {name: data["directional_consistency"] for name, data in method_data.items()}

        # Comprehensive comparisons
        delta_am_comparison = compare_all_methods(
            delta_am_data, metric_name="ΔAM-ROI", seed=int(args.seed)
        )
        shift_comparison = compare_all_methods(
            shift_data, metric_name="CoA Shift (pixels)", seed=int(args.seed)
        )
        dc_comparison = compare_all_methods(
            dc_data, metric_name="Directional Consistency", seed=int(args.seed)
        )

        # Friedman tests
        # Scientific justification: Friedman requires >=3 related samples. For
        # focused comparisons (e.g., --method-filter with 2 methods) we skip it
        # and rely on paired pairwise tests/effect sizes.
        if len(method_data) >= 3:
            friedman_delta = friedman_test(delta_am_data)
            friedman_shift = friedman_test(shift_data)
            friedman_dc = friedman_test(dc_data)
        else:
            friedman_delta = None
            friedman_shift = None
            friedman_dc = None

        # Print statistical tables
        print("\n" + generate_comparison_table(delta_am_comparison))
        if friedman_delta is not None:
            print(
                f"\nFriedman Test (ΔAM-ROI): χ²={friedman_delta['statistic']:.2f}, "
                f"{format_p_value(friedman_delta['p_value'])}"
            )

        print("\n" + generate_comparison_table(shift_comparison))
        if friedman_shift is not None:
            print(
                f"\nFriedman Test (CoA Shift): χ²={friedman_shift['statistic']:.2f}, "
                f"{format_p_value(friedman_shift['p_value'])}"
            )

    # Save results
    print("\n[6/7] Saving results and statistics...")

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    timing_summary = {
        "by_method": {},
        "cf_edit_sec_total": float(timing["cf_edit_sec_total"]),
        "n_cf_edits": int(timing["n_cf_edits"]),
        "repair_image_sec_total": float(timing["repair_image_sec_total"]),
        "n_repairs": int(timing["n_repairs"]),
        "baseline_metrics_sec_total": float(timing["baseline_metrics_sec_total"]),
        "n_baseline_metrics": int(timing["n_baseline_metrics"]),
    }
    for method_name, t in timing["by_method"].items():
        n_orig = int(t["n_explain_orig"])
        n_cf = int(t["n_explain_cf"])
        n_rep = int(t["n_explain_repair"])
        timing_summary["by_method"][method_name] = {
            "n_explain_orig": n_orig,
            "mean_explain_orig_sec": float(t["explain_orig_sec_total"]) / n_orig if n_orig else None,
            "n_explain_cf": n_cf,
            "mean_explain_cf_sec": float(t["explain_cf_sec_total"]) / n_cf if n_cf else None,
            "n_explain_repair": n_rep,
            "mean_explain_repair_sec": float(t["explain_repair_sec_total"]) / n_rep if n_rep else None,
        }

    timing_csv = tables_dir / "timing_by_method.csv"
    with open(timing_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "n_explain_orig",
                "mean_explain_orig_sec",
                "n_explain_cf",
                "mean_explain_cf_sec",
                "n_explain_repair",
                "mean_explain_repair_sec",
            ]
        )
        for method_name, stats in timing_summary["by_method"].items():
            writer.writerow(
                [
                    method_name,
                    stats["n_explain_orig"],
                    stats["mean_explain_orig_sec"],
                    stats["n_explain_cf"],
                    stats["mean_explain_cf_sec"],
                    stats["n_explain_repair"],
                    stats["mean_explain_repair_sec"],
                ]
            )
    statistics_payload = {}
    if len(common_patients) >= 2:
        _write_comparison_csvs(
            delta_am_comparison,
            tables_dir / "delta_am_roi_summary.csv",
            tables_dir / "delta_am_roi_pairwise.csv",
        )
        _write_comparison_csvs(
            shift_comparison,
            tables_dir / "shift_distance_summary.csv",
            tables_dir / "shift_distance_pairwise.csv",
        )
        _write_comparison_csvs(
            dc_comparison,
            tables_dir / "directional_consistency_summary.csv",
            tables_dir / "directional_consistency_pairwise.csv",
        )
        with open(tables_dir / "delta_am_roi_table.txt", "w") as f:
            f.write(generate_comparison_table(delta_am_comparison) + "\n")
            if friedman_delta is not None:
                f.write(
                    f"\nFriedman Test (ΔAM-ROI): χ²={friedman_delta['statistic']:.2f}, "
                    f"{format_p_value(friedman_delta['p_value'])}\n"
                )
        with open(tables_dir / "shift_distance_table.txt", "w") as f:
            f.write(generate_comparison_table(shift_comparison) + "\n")
            if friedman_shift is not None:
                f.write(
                    f"\nFriedman Test (CoA Shift): χ²={friedman_shift['statistic']:.2f}, "
                    f"{format_p_value(friedman_shift['p_value'])}\n"
                )
        with open(tables_dir / "directional_consistency_table.txt", "w") as f:
            f.write(generate_comparison_table(dc_comparison) + "\n")
            if friedman_dc is not None:
                f.write(
                    f"\nFriedman Test (DC): χ²={friedman_dc['statistic']:.2f}, "
                    f"{format_p_value(friedman_dc['p_value'])}\n"
                )

        statistics_payload = {
            # Scientific justification: store full comparison objects (incl.
            # bootstrap CIs) so reviewers can independently validate every
            # table entry from the JSON without re-running experiments.
            "delta_am_roi": delta_am_comparison,
            "shift_distance": shift_comparison,
            "directional_consistency": dc_comparison,
            "friedman_tests": {
                "delta_am_roi": friedman_delta,
                "shift_distance": friedman_shift,
                "directional_consistency": friedman_dc,
            },
        }

    method_completeness = {m: {"complete": 0, "incomplete": 0} for m in xai_methods.keys()}
    for patient_id, patient_results in all_results.items():
        for method_name in xai_methods.keys():
            method_results = patient_results.get(method_name)
            if not method_results:
                continue
            if all(key in method_results for key in expected_cf_keys):
                method_completeness[method_name]["complete"] += 1
            else:
                method_completeness[method_name]["incomplete"] += 1

    with open(results_file, "w") as f:
        json.dump({
            "summary": method_aggregates,
            "detailed_results": all_results,
            "statistics": statistics_payload,
            "audit": {
                "failure_counts": failure_counts,
                "failure_log": failure_log,
                "method_completeness": method_completeness,
                "poisson_blend": poisson_stats,
                "timing": timing_summary,
            },
            "config": {
                "dataset": dataset_label,
                "seed": int(args.seed),
                "model_path": str(model_path),
                "data_root": str(data_root),
                "total_dataset_samples": total_samples,
                "n_samples_evaluated": n_samples,
                "method_suite": args.method_suite,
                "mask_source": str(args.mask_source),
                "xai_methods": list(xai_methods.keys()),
                "xai_method_labels": xai_methods,
                "method_params": method_config,
                "cf_configs": cf_configs,
                "target_size": int(args.target_size),
                "auc_steps": int(args.auc_steps),
                "ig_steps": int(args.ig_steps),
                "ig_baseline": str(args.ig_baseline),
                "total_time_minutes": total_time / 60,
            }
        }, f, indent=2)
    print(f"✓ Results saved to {results_file}")

    # Generate publication figure
    if len(common_patients) < 2:
        print("\n[7/7] Skipping publication figure (insufficient n for statistics).")
        return

    print("\n[7/7] Generating publication-quality figures...")

    method_order = [m for m in xai_methods.keys() if m in method_aggregates]
    method_labels = [xai_methods[m] for m in method_order]
    y_positions = np.arange(len(method_order))

    metrics = [
        ("ΔAM-ROI", "mean_delta_am_roi", delta_am_comparison, "steelblue"),
        ("CoA Shift (px)", "mean_shift_distance", shift_comparison, "forestgreen"),
        ("Directional Consistency", "mean_dc", dc_comparison, "coral"),
    ]

    if not method_order:
        raise RuntimeError("No methods produced aggregate metrics; cannot build figure.")

    fig, axes = plt.subplots(len(metrics), 1, figsize=(4.6, 6.2), sharex=False)

    for ax, (metric_label, key, comparison, color) in zip(axes, metrics):
        values = []
        lower_err = []
        upper_err = []
        medians = []

        for method_name in method_order:
            agg = method_aggregates[method_name]
            values.append(agg[key])

            display_name = xai_methods[method_name]
            ci = comparison["summary"][display_name]
            lower_err.append(agg[key] - ci["ci_lower"])
            upper_err.append(ci["ci_upper"] - agg[key])

            median_key = {
                "mean_delta_am_roi": "median_delta_am_roi",
                "mean_shift_distance": "median_shift_distance",
                "mean_dc": "median_dc",
            }[key]
            medians.append(agg[median_key])

        bars = ax.barh(
            y_positions,
            values,
            color=color,
            alpha=0.85,
            xerr=np.array([lower_err, upper_err]),
            capsize=4,
        )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(method_labels, fontsize=9)
        ax.set_xlabel(metric_label, fontsize=9)
        ax.set_title(metric_label, fontsize=10, loc="left", pad=4)
        ax.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.8)

        if key == "mean_delta_am_roi":
            ax.axvline(0, color="black", linewidth=0.8)
        if key == "mean_dc":
            ax.set_xlim(0, 1)

        for y, median in zip(y_positions, medians):
            ax.plot(median, y, marker="D", color="#d62728", markersize=4, label="Median")

        ax.invert_yaxis()

    handles, _ = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles[:1], ["Median"], loc="upper right", fontsize=8)

    fig.suptitle(
        f"SCBA Metrics – {dataset_label} (n={n_samples})",
        fontsize=12,
        fontweight="bold",
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Tight layout not applied.*",
            category=UserWarning,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])

    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    print(f"✓ Figure saved to {figure_path}")

    print("\n" + "=" * 90)
    print("✅ PUBLICATION-READY EXPERIMENTS COMPLETE!")
    print("=" * 90)
    print(f"\n📊 Results: {results_file}")
    print(f"📈 Figure: {figure_path}")
    print(f"\n🎯 Key Findings ({dataset_label}, n={n_samples} samples):")

    # Rank methods by ΔAM-ROI
    ranked = sorted(method_aggregates.items(), key=lambda x: x[1]["mean_delta_am_roi"], reverse=True)
    print("\nRanked by ΔAM-ROI (higher = better CF consistency):")
    for rank, (method_name, agg) in enumerate(ranked, 1):
        ci = delta_am_comparison["summary"][xai_methods[method_name]]
        print(f"  {rank}. {xai_methods[method_name]}: {agg['mean_delta_am_roi']:.4f} "
              f"(95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}])")

    print("\n" + "=" * 90)
    print("Ready for MICCAI/TMI submission!")
    print("=" * 90)


if __name__ == "__main__":
    main()
