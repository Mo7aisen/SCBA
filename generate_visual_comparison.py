#!/usr/bin/env python3
"""
Generate Visual Comparison Figure for SCBA Manuscript

Creates Figure 2: Side-by-side comparison of original image with
all 4 Grad-CAM variant attribution maps.

Shows:
- Original chest X-ray
- Seg-Grad-CAM heatmap overlay
- HiResCAM heatmap overlay
- Grad-CAM++ heatmap overlay
- Seg-XRes-CAM heatmap overlay

Author: Dr. Mohaisen Mohammed
Date: November 2025
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).parent))

from scba.data.loaders.jsrt import JSRTDataset
from scba.data.transforms.standard import get_composed_transform, get_val_transforms
from scba.models.unet import UNet
from scba.xai.common import get_decoder_target_layers
from scba.xai.cam.seg_grad_cam import SegGradCAM, SegXResCAM
from scba.xai.cam.hires_cam import HiResCAM
from scba.xai.cam.grad_cam_pp import GradCAMPlusPlus


def apply_colormap_overlay(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Apply colored heatmap overlay on grayscale image.

    Args:
        image: Grayscale image (H, W) normalized to [0, 1]
        heatmap: Heatmap (H, W) normalized to [0, 1]
        alpha: Overlay transparency (0=transparent, 1=opaque)
        colormap: OpenCV colormap (default: JET)

    Returns:
        RGB image with heatmap overlay
    """
    # Convert grayscale to RGB
    image_rgb = np.stack([image] * 3, axis=-1)

    # Apply colormap to heatmap
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0

    # Blend
    overlay = alpha * heatmap_colored + (1 - alpha) * image_rgb

    return overlay


def generate_comparison_figure(
    model_path="runs/jsrt_unet_baseline_20251101_203253.pt",
    sample_idx=15,  # Choose a representative sample
    output_path="manuscript/figures/visual_comparison.png",
    device="cuda"
):
    """Generate visual comparison figure"""

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("Loading model...")
    model = UNet(n_channels=1, n_classes=2, bilinear=True).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Load dataset
    print("Loading dataset...")
    val_transform = get_composed_transform(get_val_transforms((1024, 1024)))
    # Scientific justification (R1): avoid hard-coded absolute dataset paths.
    data_root = os.environ.get("SCBA_JSRT_ROOT")
    if not data_root:
        raise ValueError("Set SCBA_JSRT_ROOT (or refactor to pass a CLI --data-root) to run reproducibly.")
    dataset = JSRTDataset(
        data_root=data_root,
        split='test',
        transform=val_transform,
        return_patient_id=True,
        # Scientific justification (R2): write deterministic splits outside the dataset root.
        splits_path=str(Path("experiments/results/scba_publication") / "jsrt_splits.csv"),
    )
    sample = dataset[sample_idx]
    image = sample["image"]
    sample_id = sample["patient_id"]

    print(f"Processing sample: {sample_id}")

    # Move to device
    image_tensor = image.unsqueeze(0).to(device)
    image_np = image.squeeze().numpy()

    # Initialize XAI methods
    # Scientific justification (M6): avoid targeting the 1×1 segmentation head.
    target_layer = get_decoder_target_layers(model)[-1][1]
    explainers = {
        'Seg-Grad-CAM': SegGradCAM(model, device=str(device), target_layer=target_layer),
        'HiResCAM': HiResCAM(model, device=str(device), target_layer=target_layer),
        'Grad-CAM++': GradCAMPlusPlus(model, device=str(device), target_layer=target_layer),
        'Seg-XRes-CAM': SegXResCAM(model, device=str(device), target_layer=target_layer)
    }

    # Generate attributions
    print("Generating attributions...")
    attributions = {}
    for name, explainer in explainers.items():
        attr = explainer.explain(image_tensor, target_class=1)
        attr_np = attr.map
        attributions[name] = attr_np

    # Create figure
    print("Creating figure...")
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 9.0))
    axes_flat = axes.flatten()

    panel_data = [
        ('Original', image_np, False),
        ('Seg-Grad-CAM', attributions['Seg-Grad-CAM'], True),
        ('HiResCAM', attributions['HiResCAM'], True),
        ('Grad-CAM++', attributions['Grad-CAM++'], True),
        ('Seg-XRes-CAM', attributions['Seg-XRes-CAM'], True),
    ]

    for ax, (title, data, is_attr) in zip(axes_flat, panel_data):
        if is_attr:
            overlay = apply_colormap_overlay(image_np, data, alpha=0.55)
            ax.imshow(overlay)
        else:
            ax.imshow(data, cmap='gray')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    # Reserve last axis for colorbar if not used
    for ax in axes_flat[len(panel_data):]:
        ax.axis('off')

    # Add colorbar
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap='jet', norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Attribution Intensity', fontsize=10, fontweight='bold')

    plt.suptitle(f'Grad-CAM Variants (Sample: {sample_id})', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])

    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved: {output_path}")
    plt.close()

    # Also create individual panels at higher resolution
    print("Creating individual high-res panels...")
    for name, attr in attributions.items():
        panel_path = output_path.replace('.png', f'_{name.lower().replace("-", "_")}.png')
        fig_panel, ax = plt.subplots(1, 1, figsize=(6, 6))
        overlay = apply_colormap_overlay(image_np, attr, alpha=0.5)
        ax.imshow(overlay)
        ax.set_title(name, fontsize=16, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(panel_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {panel_path}")

    print("\n✅ Visual comparison complete!")
    return attributions


def generate_multiple_examples(
    n_examples=3,
    output_dir="manuscript/figures/examples"
):
    """Generate comparison figures for multiple representative samples"""

    # Select diverse samples (different lung shapes/pathologies)
    sample_indices = [5, 15, 25]  # Adjust based on visual inspection

    os.makedirs(output_dir, exist_ok=True)

    for idx in sample_indices:
        output_path = os.path.join(output_dir, f'visual_comparison_sample_{idx:02d}.png')
        print(f"\n--- Generating example {idx} ---")
        generate_comparison_figure(sample_idx=idx, output_path=output_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate visual comparison figure')
    parser.add_argument('--sample_idx', type=int, default=15,
                       help='Test sample index (0-37)')
    parser.add_argument('--output', type=str,
                       default='manuscript/figures/visual_comparison.png',
                       help='Output figure path')
    parser.add_argument('--multiple', action='store_true',
                       help='Generate multiple examples')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')

    args = parser.parse_args()

    if args.multiple:
        generate_multiple_examples()
    else:
        generate_comparison_figure(
            sample_idx=args.sample_idx,
            output_path=args.output,
            device=args.device
        )
