#!/usr/bin/env python3
"""
Generate old versus new SCBA++ comparison with zoom insets.
"""

from __future__ import annotations

import os
import logging
import argparse
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
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
from scba.data.loaders.montgomery import MontgomeryDataset
from scba.data.transforms.standard import get_composed_transform, get_val_transforms
from scba.models.unet import UNet
from scba.xai.cam.layer_cam import LayerCAM, MultiLayerCAM
from scba.xai.common import get_decoder_target_layers


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
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


def load_sample(dataset: str, data_root: Path, sample_idx: int):
    val_transform = get_composed_transform(get_val_transforms((1024, 1024)))
    if dataset == "jsrt":
        ds = JSRTDataset(str(data_root), split="test", transform=val_transform, return_patient_id=True)
    else:
        ds = MontgomeryDataset(str(data_root), split="test", transform=val_transform, return_patient_id=True)
    sample = ds[sample_idx]
    image = sample["image"]
    mask = sample["mask"]
    patient_id = sample.get("patient_id", f"sample_{sample_idx:02d}")
    image_np = image.squeeze().cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy().astype(np.uint8)
    return image_np, mask_np, patient_id


def normalize_map(cam: np.ndarray) -> np.ndarray:
    cam_min = float(np.min(cam))
    cam_max = float(np.max(cam))
    denom = cam_max - cam_min
    if denom < 1e-8:
        return np.zeros_like(cam)
    return (cam - cam_min) / denom


def overlay_heatmap(image: np.ndarray, cam: np.ndarray, cmap: str = "inferno", alpha: float = 0.55) -> np.ndarray:
    cam_norm = normalize_map(cam)
    colormap = plt.get_cmap(cmap)
    heatmap = colormap(cam_norm)[..., :3]
    image_rgb = np.stack([image] * 3, axis=-1)
    overlay = (1 - alpha) * image_rgb + alpha * heatmap
    return np.clip(overlay, 0, 1)


def overlay_diverging_diff(diff: np.ndarray, cmap: str = "PuOr") -> np.ndarray:
    abs_diff = np.abs(diff).astype(float)
    scale = float(np.percentile(abs_diff, 99.0)) if abs_diff.size else 0.0
    if scale < 1e-8:
        diff_norm = np.full_like(diff, 0.5, dtype=float)
    else:
        diff_norm = (diff / (2.0 * scale)) + 0.5
    diff_norm = np.clip(diff_norm, 0.0, 1.0)
    return plt.get_cmap(cmap)(diff_norm)[..., :3]


def draw_boundary(ax, mask: np.ndarray, color: str = "#ffd54f", linewidth: float = 1.2) -> None:
    contours = measure.find_contours(mask.astype(float), 0.5)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=linewidth)


def draw_roi_band_contour(ax, roi_band: np.ndarray, color: str = "#00e5ff", linewidth: float = 1.6) -> None:
    contours = measure.find_contours(roi_band.astype(float), 0.5)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=linewidth)


def add_scale_bar(ax, length_px: int, label: str) -> None:
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    x0 = x_lim[0] + 30
    y0 = y_lim[0] - 30
    ax.plot([x0, x0 + length_px], [y0, y0], color="white", linewidth=3, solid_capstyle="butt")
    ax.text(x0, y0 - 15, label, color="white", fontsize=8, va="top")


def get_roi_bbox(roi_band: np.ndarray, pad: int = 18, min_size: int = 140):
    coords = np.argwhere(roi_band > 0)
    if coords.size == 0:
        h, w = roi_band.shape
        return 0, min(h, min_size), 0, min(w, min_size)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    cy = int((y0 + y1) / 2)
    cx = int((x0 + x1) / 2)
    size = max(y1 - y0, x1 - x0, min_size) + 2 * pad
    half = size // 2
    y0 = max(0, cy - half)
    x0 = max(0, cx - half)
    y1 = min(roi_band.shape[0], y0 + size)
    x1 = min(roi_band.shape[1], x0 + size)
    y0 = max(0, y1 - size)
    x0 = max(0, x1 - size)
    return int(y0), int(y1), int(x0), int(x1)


def add_zoom_inset(
    ax,
    image: np.ndarray,
    bbox,
    inset_loc: str = "lower right",
    title: str = "Zoom",
    *,
    width: str = "38%",
    height: str = "38%",
    draw_rect: bool = True,
    draw_connectors: bool = True,
) -> None:
    y0, y1, x0, x1 = bbox
    if draw_rect:
        rect = mpl.patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0, linewidth=1.2, edgecolor="#ffd54f", facecolor="none"
        )
        ax.add_patch(rect)
    axins = inset_axes(ax, width=width, height=height, loc=inset_loc, borderpad=1)
    axins.imshow(image[y0:y1, x0:x1])
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_title(title, fontsize=8)
    if draw_connectors:
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="#ffd54f", linewidth=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SCBA old versus new comparison figure.")
    parser.add_argument("--dataset", choices=["jsrt", "montgomery"], default="jsrt")
    parser.add_argument("--sample-idx", type=int, default=15)
    parser.add_argument("--output", type=str, default="submissions/journal/figures/scba_old_vs_new_comparison.pdf")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    configure_matplotlib()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.dataset == "jsrt":
        # Scientific justification (R1): avoid hard-coded absolute paths.
        data_root_env = os.environ.get("SCBA_JSRT_ROOT")
        if not data_root_env:
            raise ValueError("Set SCBA_JSRT_ROOT (and optionally SCBA_JSRT_MODEL_PATH) to run reproducibly.")
        data_root = Path(data_root_env)
        model_path = Path(os.environ.get("SCBA_JSRT_MODEL_PATH", "runs/jsrt_unet_baseline_20251101_203253.pt"))
        dataset_label = "JSRT"
    else:
        data_root_env = os.environ.get("SCBA_MONTGOMERY_ROOT")
        if not data_root_env:
            raise ValueError(
                "Set SCBA_MONTGOMERY_ROOT (and optionally SCBA_MONTGOMERY_MODEL_PATH) to run reproducibly."
            )
        data_root = Path(data_root_env)
        model_path = Path(os.environ.get("SCBA_MONTGOMERY_MODEL_PATH", "runs/montgomery_unet_baseline_20251101_203253.pt"))
        dataset_label = "Montgomery"

    image_np, mask_np, patient_id = load_sample(args.dataset, data_root, args.sample_idx)
    model = load_model(model_path, device)

    base_config = BorderEditConfig(
        radius_px=3,
        operation="dilate",
        band_px=14,
        area_budget=0.50,
        seed=42,
        blend_method="poisson",
    )

    editor_old = BorderEditor(BorderEditConfig(**{**base_config.__dict__, "warp_mode": "none"}))
    editor_new = BorderEditor(BorderEditConfig(**{**base_config.__dict__, "warp_mode": "tps"}))

    cf_old, mask_old, roi_band = editor_old.apply_border_edit(image_np, mask_np)
    cf_new, mask_new, _ = editor_new.apply_border_edit(image_np, mask_np)

    image_tensor_old = torch.from_numpy(cf_old).float().unsqueeze(0).unsqueeze(0).to(device)
    image_tensor_new = torch.from_numpy(cf_new).float().unsqueeze(0).unsqueeze(0).to(device)

    decoder_layers = get_decoder_target_layers(model, n_layers=2)
    target_layer = decoder_layers[-1][1]
    old_explainer = LayerCAM(model, device=str(device), target_layer=target_layer)
    new_explainer = MultiLayerCAM(model, device=str(device))

    cam_old_result = old_explainer.explain(image_tensor_old, target_class=1)
    cam_new_result = new_explainer.explain(image_tensor_new, target_class=1)
    cam_old = cam_old_result.map
    cam_new = cam_new_result.map

    overlay_old = overlay_heatmap(cf_old, cam_old, cmap="inferno", alpha=0.55)
    overlay_new = overlay_heatmap(cf_new, cam_new, cmap="inferno", alpha=0.55)

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.2), constrained_layout=True)

    axes[0].imshow(image_np, cmap="gray")
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")

    draw_boundary(axes[0], mask_np, color="#ffd54f", linewidth=1.2)

    axes[1].imshow(overlay_old)
    axes[1].set_title("SCBA baseline", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(overlay_new)
    axes[2].set_title("SCBA++", fontsize=10)
    axes[2].axis("off")

    diff_overlay = overlay_diverging_diff(cam_new_result.raw_map - cam_old_result.raw_map, cmap="PuOr")

    draw_boundary(axes[1], mask_np, color="#ffd54f", linewidth=1.2)
    draw_boundary(axes[2], mask_np, color="#ffd54f", linewidth=1.2)
    draw_roi_band_contour(axes[1], roi_band, color="#00e5ff", linewidth=1.6)
    draw_roi_band_contour(axes[2], roi_band, color="#00e5ff", linewidth=1.6)

    bbox = get_roi_bbox(roi_band)
    add_zoom_inset(
        axes[1],
        overlay_old,
        bbox,
        inset_loc="upper right",
        title="Zoom",
        width="40%",
        height="40%",
        draw_rect=True,
        draw_connectors=True,
    )
    add_zoom_inset(
        axes[2],
        overlay_new,
        bbox,
        inset_loc="upper right",
        title="Zoom",
        width="40%",
        height="40%",
        draw_rect=True,
        draw_connectors=True,
    )
    add_zoom_inset(
        axes[1],
        diff_overlay,
        bbox,
        inset_loc="lower right",
        title="ΔCAM",
        width="40%",
        height="40%",
        draw_rect=False,
        draw_connectors=False,
    )
    add_zoom_inset(
        axes[2],
        diff_overlay,
        bbox,
        inset_loc="lower right",
        title="ΔCAM",
        width="40%",
        height="40%",
        draw_rect=False,
        draw_connectors=False,
    )

    arrow_targets = [
        (bbox[2] + 0.30 * (bbox[3] - bbox[2]), bbox[0] + 0.30 * (bbox[1] - bbox[0])),
        (bbox[2] + 0.70 * (bbox[3] - bbox[2]), bbox[0] + 0.60 * (bbox[1] - bbox[0])),
    ]
    for target in arrow_targets:
        ann = axes[2].annotate(
            "",
            xy=target,
            xytext=(target[0] - 180, target[1] - 180),
            arrowprops=dict(arrowstyle="->", color="red", lw=3),
        )
        if ann.arrow_patch is not None:
            ann.arrow_patch.set_path_effects(
                [pe.Stroke(linewidth=5.5, foreground="white"), pe.Normal()]
            )

    for ax in axes:
        add_scale_bar(ax, length_px=120, label="120 px")

    fig.suptitle(
        f"Old versus New SCBA++ Comparison ({dataset_label}, {patient_id})",
        fontsize=11,
        fontweight="bold",
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".tiff"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
