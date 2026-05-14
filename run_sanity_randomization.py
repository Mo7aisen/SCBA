"""
Cascading parameter randomization sanity check for SCBA saliency maps.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

from scba.data.loaders.jsrt import JSRTDataset
from scba.data.loaders.montgomery import MontgomeryDataset
from scba.data.loaders.shenzhen import ShenzhenDataset
from scba.data.transforms.standard import get_composed_transform, get_val_transforms
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
    parser = argparse.ArgumentParser(
        description="Cascading parameter randomization sanity check for SCBA."
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
        default="experiments/results/scba_sanity_randomization",
        help="Directory to store sanity check outputs.",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Optional limit on number of test samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for all stochastic components (default: 42).",
    )
    parser.add_argument(
        "--auc-steps",
        type=int,
        default=20,
        help="Unused (kept for runner parity across stages).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing results file if available.",
    )
    return parser.parse_args()


def _randomize_module(module: torch.nn.Module) -> None:
    for sub in module.modules():
        if isinstance(sub, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(sub.weight, mode="fan_out", nonlinearity="relu")
            if sub.bias is not None:
                torch.nn.init.zeros_(sub.bias)
        elif isinstance(sub, torch.nn.BatchNorm2d):
            if sub.weight is not None:
                torch.nn.init.ones_(sub.weight)
            if sub.bias is not None:
                torch.nn.init.zeros_(sub.bias)
            sub.running_mean.zero_()
            sub.running_var.fill_(1)


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    if a_flat.std() < 1e-8 or b_flat.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def _ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    data_range = float(max(a.max() - a.min(), b.max() - b.min(), 1e-6))
    return float(ssim(a, b, data_range=data_range))


def main():
    args = parse_args()
    cfg = DATASET_CONFIG[args.dataset]
    DatasetClass = cfg["loader"]
    dataset_label = cfg["dataset_name"]

    from scba.utils.reproducibility import seed_everything
    # Scientific justification (R4): randomization sanity checks must be
    # deterministic so that measured degradation reflects the procedure, not RNG.
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

    results_file = output_dir / f"scba_randomization_{args.dataset}.json"
    table_file = output_dir / f"scba_randomization_{args.dataset}.csv"

    print("=" * 90)
    print(f"SCBA SANITY CHECK: CASCADING PARAMETER RANDOMIZATION ({dataset_label})")
    print("=" * 90)
    print(f"✓ Model: {model_path}")
    print(f"✓ Data: {data_root}")
    print(f"✓ Output: {output_dir}")
    if args.resume and results_file.exists():
        print(f"✓ Resuming: found existing results at {results_file}")
        return
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
        raise ValueError("No samples selected for sanity check.")

    decoder_layers = get_decoder_target_layers(model, n_layers=2)
    method_kwargs = {"target_layers": [layer for _, layer in decoder_layers]}

    stage_order = [
        "outc",
        "up4",
        "up3",
        "up2",
        "up1",
        "down4",
        "down3",
        "down2",
        "down1",
        "inc",
    ]

    print(f"✓ Samples: {len(indices)}")
    print(f"✓ Randomization stages: {len(stage_order)}")

    samples = []
    for idx in indices:
        sample = dataset[idx]
        samples.append(
            {
                "patient_id": sample["patient_id"],
                "image": sample["image"],
            }
        )

    baseline_maps = []
    print("\n[1/3] Computing baseline saliency maps...")
    start = time.time()
    for sample in samples:
        image_tensor = sample["image"].unsqueeze(0).to(device)
        saliency = explain(
            image_tensor,
            model,
            method="multi_layer_cam",
            target_class=1,
            device=device,
            **method_kwargs,
        )
        baseline_maps.append(saliency.map)
    print(f"✓ Baselines computed in {(time.time() - start) / 60:.1f} min")

    results = {
        "dataset": dataset_label,
        "seed": int(args.seed),
        "model_path": str(model_path),
        "data_root": str(data_root),
        "n_samples": len(samples),
        "randomization_order": stage_order,
        "stages": {},
    }

    print("\n[2/3] Cascading randomization...")
    stage_start = time.time()
    for stage in stage_order:
        _randomize_module(getattr(model, stage))
        model.eval()

        correlations = []
        similarities = []
        per_sample = {}

        for sample, baseline_map in zip(samples, baseline_maps):
            image_tensor = sample["image"].unsqueeze(0).to(device)
            saliency = explain(
                image_tensor,
                model,
                method="multi_layer_cam",
                target_class=1,
                device=device,
                **method_kwargs,
            )
            corr = _pearson_corr(baseline_map, saliency.map)
            sim = _ssim_score(baseline_map, saliency.map)
            correlations.append(corr)
            similarities.append(sim)
            per_sample[sample["patient_id"]] = {
                "pearson_corr": corr,
                "ssim": sim,
            }

        results["stages"][stage] = {
            "mean_pearson_corr": float(np.mean(correlations)),
            "std_pearson_corr": float(np.std(correlations)),
            "mean_ssim": float(np.mean(similarities)),
            "std_ssim": float(np.std(similarities)),
            "per_sample": per_sample,
        }
        print(
            f"✓ Randomized {stage}: corr={results['stages'][stage]['mean_pearson_corr']:.3f}, "
            f"ssim={results['stages'][stage]['mean_ssim']:.3f}"
        )

    results["total_minutes"] = (time.time() - stage_start) / 60.0

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    with open(table_file, "w") as f:
        f.write("stage,mean_pearson_corr,std_pearson_corr,mean_ssim,std_ssim\n")
        for stage in stage_order:
            s = results["stages"][stage]
            f.write(
                f"{stage},{s['mean_pearson_corr']:.6f},{s['std_pearson_corr']:.6f},"
                f"{s['mean_ssim']:.6f},{s['std_ssim']:.6f}\n"
            )

    print("\n[3/3] Sanity check completed.")
    print(f"✓ Results saved to {results_file}")
    print(f"✓ Summary table saved to {table_file}")


if __name__ == "__main__":
    main()
