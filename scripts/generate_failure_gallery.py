#!/usr/bin/env python3
"""
Generate a failure gallery for cross dataset Montgomery to JSRT.
"""

from __future__ import annotations

import os
import logging
import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from skimage import measure

logging.getLogger("albumentations.check_version").setLevel(logging.CRITICAL)
logging.getLogger("albumentations.check_version").disabled = True

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scba.cf.borders import BorderEditor, BorderEditConfig
from scba.data.loaders.jsrt import JSRTDataset
from scba.data.transforms.standard import get_composed_transform, get_val_transforms
from scba.models.unet import UNet
from scba.xai.cam.layer_cam import MultiLayerCAM


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_model(model_path: Path, device: torch.device) -> UNet:
    model = UNet(n_channels=1, n_classes=2, bilinear=True).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def load_jsrt_dataset() -> JSRTDataset:
    val_transform = get_composed_transform(get_val_transforms((1024, 1024)))
    # Scientific justification (R1): avoid hard-coded absolute dataset paths.
    data_root = os.environ.get("SCBA_JSRT_ROOT")
    if not data_root:
        raise ValueError("Set SCBA_JSRT_ROOT to run reproducibly.")
    return JSRTDataset(
        data_root,
        split="test",
        transform=val_transform,
        return_patient_id=True,
        splits_path=str(Path("experiments/results/scba_publication") / "jsrt_splits.csv"),
    )


def normalize_map(arr: np.ndarray) -> np.ndarray:
    arr_min = float(np.min(arr))
    arr_max = float(np.max(arr))
    denom = arr_max - arr_min
    if denom < 1e-8:
        return np.zeros_like(arr)
    return (arr - arr_min) / denom


def overlay_heatmap(image: np.ndarray, cam: np.ndarray, cmap: str = "inferno", alpha: float = 0.55) -> np.ndarray:
    cam_norm = normalize_map(cam)
    colormap = plt.get_cmap(cmap)
    heatmap = colormap(cam_norm)[..., :3]
    image_rgb = np.stack([image] * 3, axis=-1)
    overlay = (1 - alpha) * image_rgb + alpha * heatmap
    return np.clip(overlay, 0, 1)


def draw_roi_band(ax, roi_band: np.ndarray) -> None:
    roi_alpha = (roi_band > 0).astype(float) * 0.32
    roi_rgb = np.zeros((roi_band.shape[0], roi_band.shape[1], 3), dtype=float)
    roi_rgb[..., 1] = 1.0
    roi_rgb[..., 2] = 1.0
    ax.imshow(roi_rgb, alpha=roi_alpha)
    contours = measure.find_contours(roi_band.astype(float), 0.5)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], color="#00e5ff", linewidth=2.2)


def draw_boundary(ax, mask: np.ndarray) -> None:
    contours = measure.find_contours(mask.astype(float), 0.5)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], color="#ffd54f", linewidth=1.2)


def highlight_spurious_region(ax, cam: np.ndarray, roi_band: np.ndarray) -> None:
    cam_norm = normalize_map(cam)
    outside = (roi_band == 0)
    if not np.any(outside):
        return
    masked = cam_norm.copy()
    masked[~outside] = -1.0
    y, x = np.unravel_index(np.argmax(masked), masked.shape)
    circ = patches.Circle((x, y), radius=55, fill=False, edgecolor="red", linewidth=2.8)
    ax.add_patch(circ)
    ax.text(x + 60, y, "Actual: ribs or texture", color="red", fontsize=8, va="center")


def get_worst_cases(results_path: Path, top_k: int = 3) -> List[Tuple[str, Dict]]:
    with results_path.open() as f:
        data = json.load(f)

    worst_cases = []
    for patient_id, methods in data["detailed_results"].items():
        method = methods.get("multi_layer_cam", {})
        worst_cf = None
        worst_value = None
        for cf_key, cf_results in method.items():
            if cf_key == "baseline_metrics":
                continue
            metric = cf_results.get("metrics", {}).get("delta_am_roi")
            if metric is None:
                continue
            if worst_value is None or metric < worst_value:
                worst_value = metric
                worst_cf = (cf_key, cf_results)
        if worst_cf is None:
            continue
        worst_cases.append((patient_id, {"delta": worst_value, "cf_key": worst_cf[0], "cf": worst_cf[1]}))

    worst_cases.sort(key=lambda x: x[1]["delta"])
    return worst_cases[:top_k]


def apply_counterfactual(image: np.ndarray, mask: np.ndarray, cf_config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    config = BorderEditConfig(
        radius_px=cf_config.get("radius_px", 3),
        operation=cf_config.get("operation", "dilate"),
        band_px=14,
        area_budget=0.50,
        seed=42,
        blend_method="poisson",
    )
    editor = BorderEditor(config)
    cf_image, cf_mask, roi_band = editor.apply_border_edit(image, mask)
    return cf_image, roi_band


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate failure gallery for Montgomery to JSRT.")
    parser.add_argument("--results", type=str, default="experiments/results/scba_cross_montgomery_to_jsrt/scba_publication_results.json")
    parser.add_argument("--output", type=str, default="submissions/journal/figures/scba_failure_gallery.pdf")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    configure_matplotlib()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(
        Path(os.environ.get("SCBA_MONTGOMERY_MODEL_PATH", "runs/montgomery_unet_baseline_20251101_203253.pt")),
        device,
    )
    dataset = load_jsrt_dataset()

    worst_cases = get_worst_cases(Path(args.results), top_k=3)

    fig, axes = plt.subplots(1, len(worst_cases), figsize=(9.2, 3.4), constrained_layout=True)
    if len(worst_cases) == 1:
        axes = [axes]

    for ax, (patient_id, info) in zip(axes, worst_cases):
        sample = next((s for s in dataset if s.get("patient_id") == patient_id), None)
        if sample is None:
            continue
        image = sample["image"].squeeze().numpy()
        mask = sample["mask"].squeeze().numpy().astype(np.uint8)

        cf_image, roi_band = apply_counterfactual(image, mask, info["cf"]["config"])
        image_tensor = torch.from_numpy(cf_image).float().unsqueeze(0).unsqueeze(0).to(device)

        explainer = MultiLayerCAM(model, device=str(device))
        cam = explainer.explain(image_tensor, target_class=1).map

        overlay = overlay_heatmap(cf_image, cam, cmap="inferno", alpha=0.55)
        ax.imshow(overlay)

        draw_boundary(ax, mask)
        draw_roi_band(ax, roi_band)
        highlight_spurious_region(ax, cam, roi_band)
        ax.text(10, 18, "Expected: ROI band", color="#00e5ff", fontsize=8, va="top")

        ax.set_title(f"{patient_id}\nΔAM ROI {info['delta']:.4f}", fontsize=9)
        ax.axis("off")

    fig.suptitle("Failure Gallery: Montgomery to JSRT (MultiLayer CAM)", fontsize=11, fontweight="bold")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".tiff"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
