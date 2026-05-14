#!/usr/bin/env python3
"""
Ablation Study: Effect of Perturbation Magnitude on Counterfactual Consistency

This script tests how different border perturbation radii (r=1,2,3,4,5 pixels)
affect counterfactual consistency metrics for Grad-CAM family methods.

Research Question: Does CF consistency degrade with larger perturbations?
Hypothesis: Larger perturbations should show larger but still consistent
attribution shifts (monotonic relationship).

Author: Dr. Mohaisen Mohammed
Date: November 2025
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scba.data.loaders.jsrt import JSRTDataset
from scba.models.unet import UNet
from scba.xai.cam.seg_grad_cam import SegGradCAM
from scba.xai.cam.hires_cam import HiResCAM
from scba.xai.cam.grad_cam_pp import GradCAMPlusPlus
from scba.xai.common import get_decoder_target_layers
from scba.cf.borders import apply_border_edit
from scba.metrics.cf_consistency import (
    attribution_mass_roi,
    center_of_attribution,
    directional_consistency
)

@dataclass
class AblationConfig:
    """Configuration for ablation study"""
    model_path: str = "runs/jsrt_unet_baseline_20251101_203253.pt"
    data_root: Optional[str] = None
    n_samples: int = 20  # Subset for ablation (faster)
    perturbation_radii: List[int] = None  # [1, 2, 3, 4, 5]
    operations: List[str] = None  # ['dilate', 'erode']
    xai_methods: List[str] = None  # All 4 Grad-CAM variants
    device: str = "cuda"
    output_dir: str = "experiments/results/ablation_perturbation_magnitude"
    seed: int = 42

    def __post_init__(self):
        if self.perturbation_radii is None:
            self.perturbation_radii = [1, 2, 3, 4, 5]
        if self.operations is None:
            self.operations = ['dilate', 'erode']
        if self.xai_methods is None:
            self.xai_methods = ['seg_grad_cam', 'hires_cam', 'grad_cam_pp']


class AblationExperiment:
    """Ablation study executor"""

    def __init__(self, config: AblationConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Set random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

        print(f"Ablation Study: Perturbation Magnitude")
        print(f"Device: {self.device}")
        print(f"Radii tested: {config.perturbation_radii}")
        print(f"Operations: {config.operations}")
        print(f"Methods: {config.xai_methods}")
        print(f"Samples: {config.n_samples}")

    def load_model_and_data(self):
        """Load trained model and test dataset"""
        print("\n[1/5] Loading model and data...")

        # Load model (2 classes: background and lung)
        self.model = UNet(n_channels=1, n_classes=2, bilinear=True).to(self.device)
        checkpoint = torch.load(self.config.model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

        # Load test dataset
        data_root = self.config.data_root or os.environ.get("SCBA_JSRT_ROOT")
        if not data_root:
            raise ValueError(
                "JSRT data root is required for ablation (set AblationConfig.data_root or SCBA_JSRT_ROOT)."
            )
        self.dataset = JSRTDataset(
            data_root=data_root,
            split='test',
            target_size=(1024, 1024),
            return_patient_id=True,
            # Scientific justification (R2): write deterministic splits under the
            # experiment output directory, not into the dataset root.
            splits_path=str(Path(self.config.output_dir) / "jsrt_splits.csv"),
        )
        # Limit to subset for faster ablation
        self.dataset_indices = list(range(min(self.config.n_samples, len(self.dataset))))
        print(f"✓ Model loaded: {self.config.model_path}")
        print(f"✓ Dataset loaded: {len(self.dataset_indices)} samples (from {len(self.dataset)} total)")

    def initialize_xai_methods(self):
        """Initialize all XAI methods"""
        print("\n[2/5] Initializing XAI methods...")

        self.explainers = {}
        # Scientific justification (M6): avoid targeting the 1×1 segmentation
        # head (`outc`) for CAM methods because it can be spatially degenerate.
        # Align ablation with the main experiments by targeting a decoder layer.
        decoder_layers = get_decoder_target_layers(self.model)
        target_layer = decoder_layers[-1][1] if decoder_layers else self.model.outc.conv
        device_str = str(self.device)

        if 'seg_grad_cam' in self.config.xai_methods:
            self.explainers['seg_grad_cam'] = SegGradCAM(self.model, device=device_str, target_layer=target_layer)
        if 'hires_cam' in self.config.xai_methods:
            self.explainers['hires_cam'] = HiResCAM(self.model, device=device_str, target_layer=target_layer)
        if 'grad_cam_pp' in self.config.xai_methods:
            self.explainers['grad_cam_pp'] = GradCAMPlusPlus(self.model, device=device_str, target_layer=target_layer)

        print(f"✓ Initialized {len(self.explainers)} XAI methods")

    def run_experiments(self):
        """Run ablation experiments across all radii"""
        print("\n[3/5] Running ablation experiments...")

        results = {
            'config': asdict(self.config),
            'per_radius': {},  # Organized by radius
            'per_method': {},  # Organized by method
            'summary': {}
        }

        # Total experiments: n_samples × n_methods × n_radii × n_operations
        total_exp = (len(self.dataset_indices) * len(self.config.xai_methods) *
                     len(self.config.perturbation_radii) * len(self.config.operations))

        print(f"Total experiments: {total_exp}")
        pbar = tqdm(total=total_exp, desc="Experiments")

        # Loop over samples
        for sample_num, idx in enumerate(self.dataset_indices):
            sample = self.dataset[idx]
            if isinstance(sample, dict):
                image_data = sample["image"]
                mask_data = sample["mask"]
                sample_id = sample.get("patient_id", f"sample_{idx:03d}")
            else:
                if len(sample) == 3:
                    image_data, mask_data, sample_id = sample
                else:
                    image_data, mask_data = sample
                    sample_id = f"sample_{idx:03d}"

            if isinstance(image_data, np.ndarray):
                image_tensor = torch.from_numpy(image_data.astype(np.float32))
            else:
                image_tensor = image_data.float()

            if image_tensor.ndim == 2:
                image_tensor = image_tensor.unsqueeze(0)
            elif image_tensor.ndim == 3 and image_tensor.shape[0] not in (1,):
                # Assume channel-last, move channels to first dimension
                image_tensor = image_tensor.permute(2, 0, 1)

            image_np = image_tensor.squeeze().cpu().numpy()
            image = image_tensor.unsqueeze(0).to(self.device)

            if isinstance(mask_data, np.ndarray):
                mask_tensor = torch.from_numpy(mask_data.astype(np.uint8))
            else:
                mask_tensor = mask_data

            mask_gt_np = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)

            # Get model prediction
            with torch.no_grad():
                pred = self.model(image)
                # Scientific justification (M2/M6): use argmax over logits to
                # obtain the predicted segmentation mask (ŷ) without introducing
                # an arbitrary probability threshold.
                mask_pred = torch.argmax(pred, dim=1).cpu().numpy().astype(np.uint8).squeeze()
                mask_pred = (mask_pred > 0).astype(np.uint8)

            # Do not silently drop failures: record empty predictions to avoid
            # selection bias in ablation statistics (M6).
            if mask_pred.sum() == 0:
                results.setdefault("failures", []).append(
                    {"sample_id": sample_id, "reason": "empty_prediction"}
                )
                pbar.update(
                    len(self.explainers)
                    * len(self.config.perturbation_radii)
                    * len(self.config.operations)
                )
                continue

            # Test each radius
            for radius in self.config.perturbation_radii:
                if radius not in results['per_radius']:
                    results['per_radius'][radius] = {method: [] for method in self.config.xai_methods}

                # Test each operation
                for operation in self.config.operations:
                    # Generate counterfactual
                    try:
                        img_cf, mask_cf, roi_band = apply_border_edit(
                            image=image_np,
                            mask=mask_pred,
                            radius_px=radius,
                            operation=operation,
                            seed=self.config.seed + idx
                        )
                    except Exception as e:
                        results.setdefault("failures", []).append(
                            {
                                "sample_id": sample_id,
                                "radius": radius,
                                "operation": operation,
                                "reason": "cf_generation_failed",
                                "error": str(e),
                            }
                        )
                        pbar.update(len(self.explainers))
                        continue

                    if roi_band.sum() == 0:
                        results.setdefault("failures", []).append(
                            {
                                "sample_id": sample_id,
                                "radius": radius,
                                "operation": operation,
                                "reason": "empty_roi_band",
                            }
                        )
                        pbar.update(len(self.explainers))
                        continue

                    # Convert back to tensor
                    img_cf_tensor = torch.from_numpy(img_cf).unsqueeze(0).unsqueeze(0).float().to(self.device)

                    # Test each XAI method
                    for method_name, explainer in self.explainers.items():
                        try:
                            # Original attribution
                            attr_orig = explainer.explain(image, target_class=1)
                            attr_orig_np = attr_orig.map

                            # Counterfactual attribution
                            attr_cf = explainer.explain(img_cf_tensor, target_class=1)
                            attr_cf_np = attr_cf.map
                        except Exception as e:
                            results.setdefault("failures", []).append(
                                {
                                    "sample_id": sample_id,
                                    "radius": radius,
                                    "operation": operation,
                                    "method": method_name,
                                    "reason": "xai_failed",
                                    "error": str(e),
                                }
                            )
                            pbar.update(1)
                            continue

                        # Compute metrics
                        am_orig = attribution_mass_roi(attr_orig_np, roi_band)
                        am_cf = attribution_mass_roi(attr_cf_np, roi_band)
                        delta_am = am_cf - am_orig

                        coa_orig = center_of_attribution(attr_orig_np)
                        coa_cf = center_of_attribution(attr_cf_np)
                        coa_shift = np.sqrt((coa_cf[0] - coa_orig[0])**2 + (coa_cf[1] - coa_orig[1])**2)

                        dc = directional_consistency(attr_orig_np, attr_cf_np, roi_band)["directional_consistency"]

                        # Store result
                        result = {
                            'sample_id': sample_id,
                            'radius': radius,
                            'operation': operation,
                            'method': method_name,
                            'delta_am_roi': float(delta_am),
                            'coa_shift': float(coa_shift),
                            'directional_consistency': float(dc),
                            'am_original': float(am_orig),
                            'am_perturbed': float(am_cf)
                        }

                        results['per_radius'][radius][method_name].append(result)

                        pbar.update(1)

        pbar.close()
        self.results = results
        print(f"✓ Completed {total_exp} experiments")

    def analyze_results(self):
        """Analyze ablation results"""
        print("\n[4/5] Analyzing results...")

        summary = {}

        # Aggregate by radius and method
        for radius in self.config.perturbation_radii:
            summary[f'r={radius}'] = {}

            for method in self.config.xai_methods:
                results_list = self.results['per_radius'][radius][method]

                if len(results_list) == 0:
                    continue

                delta_am_values = [r['delta_am_roi'] for r in results_list]
                shift_values = [r['coa_shift'] for r in results_list]

                summary[f'r={radius}'][method] = {
                    'mean_delta_am_roi': float(np.mean(delta_am_values)),
                    'std_delta_am_roi': float(np.std(delta_am_values)),
                    'mean_coa_shift': float(np.mean(shift_values)),
                    'std_coa_shift': float(np.std(shift_values)),
                    'n_experiments': len(results_list)
                }

        self.results['summary'] = summary

        # Print summary table
        print("\n" + "="*80)
        print("ABLATION STUDY SUMMARY: Effect of Perturbation Magnitude")
        print("="*80)
        print(f"{'Radius':<10} {'Method':<20} {'ΔAM-ROI':<20} {'CoA Shift (px)':<20} {'N':<10}")
        print("-"*80)

        for radius in self.config.perturbation_radii:
            for method in self.config.xai_methods:
                if method in summary[f'r={radius}']:
                    stats = summary[f'r={radius}'][method]
                    delta_am_str = f"{stats['mean_delta_am_roi']:.4f}±{stats['std_delta_am_roi']:.4f}"
                    shift_str = f"{stats['mean_coa_shift']:.2f}±{stats['std_coa_shift']:.2f}"
                    print(f"r={radius:<7} {method:<20} {delta_am_str:<20} {shift_str:<20} {stats['n_experiments']:<10}")

        print("="*80)

    def save_results(self):
        """Save results and generate plots"""
        print("\n[5/5] Saving results and generating plots...")

        # Save JSON results
        results_path = os.path.join(self.config.output_dir, 'ablation_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Results saved: {results_path}")

        # Generate plots
        self._plot_radius_vs_metrics()

    def _plot_radius_vs_metrics(self):
        """Plot metrics vs perturbation radius"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Prepare data for plotting
        radii = self.config.perturbation_radii
        methods = self.config.xai_methods

        method_display = {
            'seg_grad_cam': 'Seg-Grad-CAM',
            'hires_cam': 'HiResCAM',
            'grad_cam_pp': 'Grad-CAM++',
        }

        for method in methods:
            mean_delta_am = []
            std_delta_am = []
            mean_shift = []
            std_shift = []

            for radius in radii:
                summary_key = f'r={radius}'
                if summary_key in self.results['summary'] and method in self.results['summary'][summary_key]:
                    stats = self.results['summary'][summary_key][method]
                    mean_delta_am.append(stats['mean_delta_am_roi'])
                    std_delta_am.append(stats['std_delta_am_roi'])
                    mean_shift.append(stats['mean_coa_shift'])
                    std_shift.append(stats['std_coa_shift'])
                else:
                    mean_delta_am.append(0)
                    std_delta_am.append(0)
                    mean_shift.append(0)
                    std_shift.append(0)

            # Plot ΔAM-ROI
            axes[0].errorbar(radii, mean_delta_am, yerr=std_delta_am,
                            marker='o', label=method_display.get(method, method), linewidth=2, capsize=5)

            # Plot CoA Shift
            axes[1].errorbar(radii, mean_shift, yerr=std_shift,
                            marker='s', label=method_display.get(method, method), linewidth=2, capsize=5)

        # Formatting
        axes[0].set_xlabel('Perturbation Radius (pixels)', fontsize=12)
        axes[0].set_ylabel('ΔAM-ROI', fontsize=12)
        axes[0].set_title('Attribution Mass Change vs Radius', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Perturbation Radius (pixels)', fontsize=12)
        axes[1].set_ylabel('CoA Shift (pixels)', fontsize=12)
        axes[1].set_title('Spatial Shift vs Radius', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle('Ablation Study: Effect of Perturbation Magnitude on CF Consistency',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        # Save figure
        fig_path = os.path.join(self.config.output_dir, 'ablation_radius_vs_metrics.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved: {fig_path}")
        plt.close()

    def run(self):
        """Execute complete ablation study"""
        self.load_model_and_data()
        self.initialize_xai_methods()
        self.run_experiments()
        self.analyze_results()
        self.save_results()

        print("\n" + "="*80)
        print("✅ ABLATION STUDY COMPLETE!")
        print("="*80)
        print(f"Results: {self.config.output_dir}/ablation_results.json")
        print(f"Figure:  {self.config.output_dir}/ablation_radius_vs_metrics.png")
        print()


def main():
    """Main entry point"""
    config = AblationConfig(
        n_samples=20,  # Use 20 samples for faster ablation
        perturbation_radii=[1, 2, 3, 4, 5],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    experiment = AblationExperiment(config)
    experiment.run()


if __name__ == '__main__':
    main()
