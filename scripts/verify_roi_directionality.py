"""
Emergency debug script: verify ROI directionality + qualitative CAM behavior.

What it does (per sample):
1) Predict mask ŷ from the model.
2) Generate counterfactual masks/images for:
   - DILATE r=R
   - ERODE r=R
3) Define directional ROIs:
   - ROI_added   = (ŷ_cf == 1) & (ŷ == 0)   (new pixels for DILATE)
   - ROI_removed = (ŷ == 1) & (ŷ_cf == 0)   (removed pixels for ERODE)
4) Compute saliency maps on original vs counterfactual images.
5) Report ΔAM-ROI on:
   - expected ROI (added for DILATE / removed for ERODE)
   - symmetric ROI band returned by apply_border_edit (current pipeline ROI)
6) Save a figure showing: image, ROI overlays, saliency maps, and Δsaliency.

Example:
  python -W error scripts/verify_roi_directionality.py \\
    --dataset jsrt --n-samples 3 --target-size 1024 --method multi_layer_cam
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import matplotlib

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scba.cf.borders import apply_border_edit
from scba.data.loaders.jsrt import JSRTDataset
from scba.data.loaders.montgomery import MontgomeryDataset
from scba.data.loaders.shenzhen import ShenzhenDataset
from scba.data.transforms.standard import get_composed_transform, get_val_transforms
from scba.metrics.cf_consistency import attribution_mass_roi, coa_shift, directional_consistency
from scba.models.unet import UNet
from scba.utils.reproducibility import seed_everything
from scba.xai.common import explain, get_decoder_target_layers


DATASET_CONFIG = {
    "jsrt": {
        "loader": JSRTDataset,
        "model_path": "runs/jsrt_unet_baseline_20251101_203253.pt",
        "model_path_env": "SCBA_JSRT_MODEL_PATH",
        "data_root_env": "SCBA_JSRT_ROOT",
        "dataset_name": "JSRT",
    },
    "montgomery": {
        "loader": MontgomeryDataset,
        "model_path": "runs/montgomery_unet_baseline_20251101_203253.pt",
        "model_path_env": "SCBA_MONTGOMERY_MODEL_PATH",
        "data_root_env": "SCBA_MONTGOMERY_ROOT",
        "dataset_name": "Montgomery",
    },
    "shenzhen": {
        "loader": ShenzhenDataset,
        "model_path": "runs/shenzhen_unet_baseline_20260201_000000.pt",
        "model_path_env": "SCBA_SHENZHEN_MODEL_PATH",
        "data_root_env": "SCBA_SHENZHEN_ROOT",
        "dataset_name": "Shenzhen",
    },
}


def _overlay_mask(ax, mask: np.ndarray, color=(1.0, 0.0, 0.0), alpha=0.35) -> None:
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[..., 0] = float(color[0])
    rgba[..., 1] = float(color[1])
    rgba[..., 2] = float(color[2])
    rgba[..., 3] = (mask > 0).astype(np.float32) * float(alpha)
    ax.imshow(rgba)


def parse_args():
    p = argparse.ArgumentParser(description="Verify ROI sign convention + qualitative CAM behavior.")
    p.add_argument("--dataset", type=str, default="jsrt", choices=DATASET_CONFIG.keys())
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="experiments/results/roi_debug")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-size", type=int, default=1024)
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--indices", type=str, default=None, help="Optional comma-separated dataset indices.")
    p.add_argument("--method", type=str, default="multi_layer_cam")
    p.add_argument("--radius", type=int, default=3)
    p.add_argument("--band-px", type=int, default=12)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DATASET_CONFIG[args.dataset]
    Dataset = cfg["loader"]

    seed_everything(args.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(
        args.model_path
        or os.environ.get(cfg.get("model_path_env", ""), "")
        or cfg["model_path"]
    )
    default_data_root = os.environ.get(cfg.get("data_root_env", ""), "")
    if not (args.data_root or default_data_root):
        raise ValueError(f"--data-root is required (or set {cfg.get('data_root_env')}).")
    data_root = Path(args.data_root or default_data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = UNet(n_channels=1, n_classes=2)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device).eval()

    target_size = (int(args.target_size), int(args.target_size))
    dataset = Dataset(
        data_root,
        split="test",
        transform=get_composed_transform(get_val_transforms(target_size)),
        return_patient_id=True,
        splits_path=str(out_dir / f"{args.dataset}_splits.csv"),
    )

    if args.indices:
        indices = [int(x.strip()) for x in str(args.indices).split(",") if x.strip()]
    else:
        stride = max(len(dataset) // max(args.n_samples, 1), 1)
        indices = list(range(0, min(len(dataset), stride * args.n_samples), stride))[: args.n_samples]

    decoder_layers = get_decoder_target_layers(model, n_layers=2)
    method_kwargs: dict = {}
    if args.method == "multi_layer_cam":
        method_kwargs = {"target_layers": [layer for _, layer in decoder_layers]}
    elif args.method in {"layer_cam", "hires_cam", "grad_cam_pp"}:
        method_kwargs = {"target_layer": decoder_layers[-1][1]}

    operations = [("dilate", int(args.radius)), ("erode", int(args.radius))]
    print(
        f"Dataset={cfg['dataset_name']} | samples={len(indices)} | method={args.method} | "
        f"target_size={args.target_size} | radius={args.radius}"
    )

    rows: list[dict] = []
    for dataset_index in indices:
        sample = dataset[dataset_index]
        pid = sample["patient_id"]
        image = sample["image"].squeeze(0).numpy().astype(np.float32)
        image_tensor = sample["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(image_tensor)
            pred = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
        mask_orig = (pred > 0).astype(np.uint8)

        sal_orig = explain(
            image_tensor,
            model,
            method=args.method,
            target_class=1,
            device=device,
            **method_kwargs,
        ).map

        for op, r in operations:
            image_cf, mask_cf, roi_band, _meta = apply_border_edit(
                image,
                mask_orig,
                radius_px=r,
                operation=op,
                band_px=int(args.band_px),
                area_budget=0.50,
                seed=int(args.seed),
                return_metadata=True,
            )
            mask_cf = (mask_cf > 0).astype(np.uint8)

            roi_added = ((mask_cf > 0) & (mask_orig == 0)).astype(np.uint8)
            roi_removed = ((mask_orig > 0) & (mask_cf == 0)).astype(np.uint8)
            roi_expected = roi_added if op == "dilate" else roi_removed

            image_cf_tensor = (
                torch.from_numpy(image_cf.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            )
            sal_cf = explain(
                image_cf_tensor,
                model,
                method=args.method,
                target_class=1,
                device=device,
                **method_kwargs,
            ).map

            am_orig = attribution_mass_roi(sal_orig, roi_expected)
            am_cf = attribution_mass_roi(sal_cf, roi_expected)
            delta_expected = float(am_cf - am_orig)

            am_orig_band = attribution_mass_roi(sal_orig, roi_band)
            am_cf_band = attribution_mass_roi(sal_cf, roi_band)
            delta_band = float(am_cf_band - am_orig_band)

            shift_px = float(coa_shift(sal_orig, sal_cf)["shift_distance"])
            dc = float(directional_consistency(sal_orig, sal_cf, roi_expected)["directional_consistency"])

            rows.append(
                {
                    "dataset": args.dataset,
                    "dataset_index": int(dataset_index),
                    "patient_id": str(pid),
                    "operation": str(op),
                    "radius": int(r),
                    "delta_am_roi_expected": float(delta_expected),
                    "delta_am_roi_band": float(delta_band),
                    "coa_shift_px": float(shift_px),
                    "dc_expected": float(dc),
                    "roi_expected_pixels": int(roi_expected.sum()),
                    "roi_band_pixels": int(roi_band.sum()),
                }
            )

            # Visualization
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.ravel()

            axes[0].imshow(image, cmap="gray")
            axes[0].set_title(f"{pid} | original")
            axes[0].axis("off")

            axes[1].imshow(image_cf, cmap="gray")
            axes[1].set_title(f"{op} r={r} | counterfactual")
            axes[1].axis("off")

            axes[2].imshow(image, cmap="gray")
            _overlay_mask(axes[2], roi_expected, color=(1.0, 0.0, 0.0), alpha=0.40)
            axes[2].set_title(f"ROI expected ({'added' if op == 'dilate' else 'removed'})")
            axes[2].axis("off")

            axes[3].imshow(image, cmap="gray")
            _overlay_mask(axes[3], roi_band.astype(np.uint8), color=(0.0, 0.6, 1.0), alpha=0.35)
            axes[3].set_title("ROI band (current)")
            axes[3].axis("off")

            axes[4].imshow(sal_orig, cmap="magma")
            axes[4].set_title("saliency (orig)")
            axes[4].axis("off")

            axes[5].imshow(sal_cf, cmap="magma")
            axes[5].set_title("saliency (cf)")
            axes[5].axis("off")

            delta_map = sal_cf - sal_orig
            vmax = float(np.max(np.abs(delta_map)) + 1e-8)
            axes[6].imshow(delta_map, cmap="bwr", vmin=-vmax, vmax=vmax)
            _overlay_mask(axes[6], roi_expected, color=(0.0, 0.0, 0.0), alpha=0.20)
            axes[6].set_title("Δsaliency (cf - orig) + ROI")
            axes[6].axis("off")

            axes[7].axis("off")
            axes[7].text(
                0.0,
                0.95,
                "\n".join(
                    [
                        f"ΔAM-ROI expected: {delta_expected:+.5f}",
                        f"ΔAM-ROI band:     {delta_band:+.5f}",
                        f"CoA shift:        {shift_px:.2f} px",
                        f"DC (expected):    {dc:.3f}",
                        f"ROI px:           {int(roi_expected.sum())}",
                    ]
                ),
                fontsize=12,
                family="monospace",
                va="top",
            )

            fig.tight_layout()
            out_path = out_dir / f"roi_debug_{args.dataset}_{pid}_{op}_r{r}_{args.method}.png"
            fig.savefig(out_path, dpi=160)
            plt.close(fig)

            print(
                f"{pid} {op} r={r} | ΔAM(expected)={delta_expected:+.5f} | "
                f"ΔAM(band)={delta_band:+.5f} | DC={dc:.3f} | shift={shift_px:.2f}px"
            )

    csv_path = out_dir / f"roi_debug_{args.dataset}_{args.method}_summary.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nSaved summary CSV: {csv_path}")

    if rows:
        for op in ["dilate", "erode"]:
            sub = [r for r in rows if r["operation"] == op]
            if not sub:
                continue
            mean = lambda k: float(np.mean([float(x[k]) for x in sub]))
            print(f"\n[{op}] n={len(sub)}")
            print(
                "mean(delta_am_expected)={:+.6f} | mean(delta_am_band)={:+.6f} | "
                "mean(DC)={:.4f} | mean(shift_px)={:.3f}".format(
                    mean("delta_am_roi_expected"),
                    mean("delta_am_roi_band"),
                    mean("dc_expected"),
                    mean("coa_shift_px"),
                )
            )


if __name__ == "__main__":
    main()
