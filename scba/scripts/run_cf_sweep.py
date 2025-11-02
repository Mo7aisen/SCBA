"""
Run counterfactual sweep experiment for SCBA.

Systematic evaluation of XAI methods using synthetic counterfactuals:
- Border edits (dilate/erode)
- Gaussian nodule insertion
- Multiple XAI methods
- Comprehensive metrics

Usage:
    python -m scba.scripts.run_cf_sweep \
        --data jsrt \
        --ckpt runs/jsrt_unet.pt \
        --methods seg_grad_cam \
        --border_radii 2,4,8 \
        --lesion_sigmas 4,8,12 \
        --out results/jsrt_cf
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from scba.cf.borders import BorderEditConfig, BorderEditor
from scba.cf.gaussian_nodules import GaussianNoduleGenerator, NoduleConfig
from scba.data.loaders import get_dataloader
from scba.metrics.cf_consistency import (
    attribution_mass_roi,
    delta_attribution_mass_roi,
    center_of_attribution,
    coa_shift,
)
from scba.models import get_model
from scba.xai.common import explain


def parse_args():
    parser = argparse.ArgumentParser(description="Run counterfactual sweep")

    # Data and model
    parser.add_argument("--data", type=str, required=True, choices=["jsrt", "montgomery"],
                        help="Dataset name")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Path to dataset root (auto-detected if not provided)")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to use")

    # XAI methods
    parser.add_argument("--methods", type=str, default="seg_grad_cam",
                        help="Comma-separated list of XAI methods")

    # Counterfactual parameters
    parser.add_argument("--border_radii", type=str, default="2,4,8",
                        help="Comma-separated border edit radii (pixels)")
    parser.add_argument("--lesion_sigmas", type=str, default="4,8,12",
                        help="Comma-separated Gaussian nodule sigmas")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Number of samples to process (default: all)")

    # Output
    parser.add_argument("--out", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--save_vis", action="store_true",
                        help="Save visualizations for each sample")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")

    return parser.parse_args()


def load_model(ckpt_path: str, device: str):
    """Load trained model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)

    # Extract config
    if "config" in ckpt:
        config = ckpt["config"]
        arch = config.get("arch", "unet")
    else:
        arch = "unet"

    # Load model
    model = get_model(arch, in_channels=1, out_channels=1)

    # Load weights
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model = model.to(device)
    model.eval()

    print(f"✓ Loaded model from {ckpt_path}")
    return model


def process_border_edit(
    image: np.ndarray,
    mask: np.ndarray,
    model: torch.nn.Module,
    methods: List[str],
    radius: int,
    device: str,
) -> Dict:
    """Process one border edit perturbation."""

    # Generate dilate edit
    config_dilate = BorderEditConfig(radius_px=radius, operation="dilate", seed=42)
    editor_dilate = BorderEditor(config_dilate)
    img_dilate, mask_dilate, roi_dilate = editor_dilate.apply_border_edit(image, mask)

    # Generate erode edit
    config_erode = BorderEditConfig(radius_px=radius, operation="erode", seed=42)
    editor_erode = BorderEditor(config_erode)
    img_erode, mask_erode, roi_erode = editor_erode.apply_border_edit(image, mask)

    # Convert to tensor
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
    img_dilate_tensor = torch.from_numpy(img_dilate).unsqueeze(0).unsqueeze(0).float().to(device)
    img_erode_tensor = torch.from_numpy(img_erode).unsqueeze(0).unsqueeze(0).float().to(device)

    results = {}

    for method in methods:
        # Compute explanations
        try:
            sal_orig = explain(image_tensor, model, method=method, device=device)
            sal_dilate = explain(img_dilate_tensor, model, method=method, device=device)
            sal_erode = explain(img_erode_tensor, model, method=method, device=device)

            # Compute metrics for dilate
            delta_am_dilate = delta_attribution_mass_roi(sal_orig.map, sal_dilate.map, roi_dilate)
            am_orig_dilate = attribution_mass_roi(sal_orig.map, roi_dilate)
            am_pert_dilate = attribution_mass_roi(sal_dilate.map, roi_dilate)

            # Compute metrics for erode
            delta_am_erode = delta_attribution_mass_roi(sal_orig.map, sal_erode.map, roi_erode)
            am_orig_erode = attribution_mass_roi(sal_orig.map, roi_erode)
            am_pert_erode = attribution_mass_roi(sal_erode.map, roi_erode)

            results[method] = {
                "dilate": {
                    "delta_am_roi": float(delta_am_dilate),
                    "am_roi_original": float(am_orig_dilate),
                    "am_roi_perturbed": float(am_pert_dilate),
                },
                "erode": {
                    "delta_am_roi": float(delta_am_erode),
                    "am_roi_original": float(am_orig_erode),
                    "am_roi_perturbed": float(am_pert_erode),
                },
            }
        except Exception as e:
            print(f"Warning: Failed to compute {method}: {e}")
            results[method] = {"error": str(e)}

    return results


def process_nodule_insertion(
    image: np.ndarray,
    mask: np.ndarray,
    model: torch.nn.Module,
    methods: List[str],
    sigma: float,
    device: str,
) -> Dict:
    """Process one nodule insertion perturbation."""

    # Generate nodule
    config = NoduleConfig(sigma_x=sigma, sigma_y=sigma, seed=42)
    generator = GaussianNoduleGenerator(config)

    try:
        img_nodule, nodule_mask, center = generator.insert_nodule(image, mask)
    except Exception as e:
        print(f"Warning: Nodule insertion failed: {e}")
        return {}

    # Convert to tensor
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
    img_nodule_tensor = torch.from_numpy(img_nodule).unsqueeze(0).unsqueeze(0).float().to(device)

    results = {}

    for method in methods:
        try:
            # Compute explanations
            sal_orig = explain(image_tensor, model, method=method, device=device)
            sal_nodule = explain(img_nodule_tensor, model, method=method, device=device)

            # Compute metrics
            delta_am = delta_attribution_mass_roi(sal_orig.map, sal_nodule.map, nodule_mask)
            am_orig = attribution_mass_roi(sal_orig.map, nodule_mask)
            am_pert = attribution_mass_roi(sal_nodule.map, nodule_mask)

            # CoA shift
            shift_metrics = coa_shift(sal_orig.map, sal_nodule.map, roi_center=center)

            results[method] = {
                "delta_am_roi": float(delta_am),
                "am_roi_original": float(am_orig),
                "am_roi_perturbed": float(am_pert),
                "coa_shift_distance": float(shift_metrics.get("distance", 0)),
            }
        except Exception as e:
            print(f"Warning: Failed to compute {method}: {e}")
            results[method] = {"error": str(e)}

    return results


def visualize_cf_results(
    image: np.ndarray,
    mask: np.ndarray,
    cf_images: Dict[str, np.ndarray],
    saliency_maps: Dict[str, np.ndarray],
    save_path: Path,
):
    """Create visualization of CF results."""

    n_cf = len(cf_images)
    n_methods = len(next(iter(saliency_maps.values())))

    fig, axes = plt.subplots(n_methods + 1, n_cf + 1, figsize=(4 * (n_cf + 1), 4 * (n_methods + 1)))

    if n_methods == 1:
        axes = axes.reshape(1, -1)

    # Original image
    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    # CF images
    for i, (cf_name, cf_img) in enumerate(cf_images.items(), 1):
        axes[0, i].imshow(cf_img, cmap="gray")
        axes[0, i].set_title(cf_name)
        axes[0, i].axis("off")

    # Saliency maps
    for i, (method_name, method_saliency) in enumerate(saliency_maps.items(), 1):
        # Original saliency
        axes[i, 0].imshow(image, cmap="gray", alpha=0.5)
        axes[i, 0].imshow(method_saliency["original"], cmap="jet", alpha=0.5)
        axes[i, 0].set_title(f"{method_name}\n(original)")
        axes[i, 0].axis("off")

        # CF saliency
        for j, cf_name in enumerate(cf_images.keys(), 1):
            if cf_name in method_saliency:
                axes[i, j].imshow(cf_images[cf_name], cmap="gray", alpha=0.5)
                axes[i, j].imshow(method_saliency[cf_name], cmap="jet", alpha=0.5)
                axes[i, j].set_title(f"{method_name}\n({cf_name})")
                axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()

    # Setup
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect data root if not provided
    if args.data_root is None:
        if args.data == "jsrt":
            args.data_root = "/home/mohaisen_mohammed/Datasets/JSRT"
        elif args.data == "montgomery":
            args.data_root = "/home/mohaisen_mohammed/Datasets/Montgomery"

    # Parse parameters
    methods = args.methods.split(",")
    border_radii = [int(r) for r in args.border_radii.split(",")]
    lesion_sigmas = [float(s) for s in args.lesion_sigmas.split(",")]

    print(f"\n{'='*80}")
    print("SCBA Counterfactual Sweep")
    print(f"{'='*80}")
    print(f"Dataset: {args.data}")
    print(f"Model: {args.ckpt}")
    print(f"XAI Methods: {methods}")
    print(f"Border Radii: {border_radii}")
    print(f"Lesion Sigmas: {lesion_sigmas}")
    print(f"Output: {out_dir}")
    print(f"{'='*80}\n")

    # Load model
    model = load_model(args.ckpt, args.device)

    # Load dataset
    dataloader = get_dataloader(args.data, args.data_root, split=args.split, batch_size=1, shuffle=False)

    # Aggregate results
    all_results = []

    # Process samples
    n_samples = args.n_samples if args.n_samples is not None else len(dataloader)

    for idx, batch in enumerate(tqdm(dataloader, total=n_samples, desc="Processing samples")):
        if idx >= n_samples:
            break

        # Extract data
        image_tensor = batch["image"].to(args.device)
        mask_tensor = batch["mask"].to(args.device)

        # Convert to numpy
        image = image_tensor.squeeze().cpu().numpy()
        mask = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)

        sample_results = {
            "sample_idx": idx,
            "border_edits": {},
            "nodule_insertions": {},
        }

        # Border edits
        for radius in border_radii:
            results = process_border_edit(image, mask, model, methods, radius, args.device)
            sample_results["border_edits"][f"radius_{radius}"] = results

        # Nodule insertions
        for sigma in lesion_sigmas:
            results = process_nodule_insertion(image, mask, model, methods, sigma, args.device)
            if results:  # Only add if successful
                sample_results["nodule_insertions"][f"sigma_{sigma}"] = results

        all_results.append(sample_results)

    # Save results
    results_file = out_dir / "cf_sweep_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved to {results_file}")

    # Compute aggregate statistics
    print(f"\n{'='*80}")
    print("Summary Statistics")
    print(f"{'='*80}")

    for method in methods:
        print(f"\nMethod: {method}")

        # Border edit statistics
        for radius in border_radii:
            delta_ams_dilate = []
            delta_ams_erode = []

            for result in all_results:
                key = f"radius_{radius}"
                if key in result["border_edits"]:
                    border_result = result["border_edits"][key]
                    if method in border_result and "error" not in border_result[method]:
                        if "dilate" in border_result[method]:
                            delta_ams_dilate.append(border_result[method]["dilate"]["delta_am_roi"])
                        if "erode" in border_result[method]:
                            delta_ams_erode.append(border_result[method]["erode"]["delta_am_roi"])

            if delta_ams_dilate:
                print(f"  Border dilate (r={radius}): ΔAM-ROI = {np.mean(delta_ams_dilate):.4f} ± {np.std(delta_ams_dilate):.4f}")
            if delta_ams_erode:
                print(f"  Border erode (r={radius}):  ΔAM-ROI = {np.mean(delta_ams_erode):.4f} ± {np.std(delta_ams_erode):.4f}")

        # Nodule statistics
        for sigma in lesion_sigmas:
            delta_ams = []

            for result in all_results:
                key = f"sigma_{sigma}"
                if key in result["nodule_insertions"]:
                    nodule_result = result["nodule_insertions"][key]
                    if method in nodule_result and "error" not in nodule_result[method]:
                        delta_ams.append(nodule_result[method]["delta_am_roi"])

            if delta_ams:
                print(f"  Nodule (σ={sigma}):         ΔAM-ROI = {np.mean(delta_ams):.4f} ± {np.std(delta_ams):.4f}")

    print(f"\n{'='*80}")
    print("✅ Counterfactual sweep complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
