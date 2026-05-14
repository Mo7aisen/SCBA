#!/usr/bin/env python3
"""
Compute correlation between CAMs and Sobel edge maps.
"""

from __future__ import annotations

import os
import logging
import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import torch
from skimage.filters import sobel

logging.getLogger("albumentations.check_version").setLevel(logging.CRITICAL)
logging.getLogger("albumentations.check_version").disabled = True

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scba.data.loaders.jsrt import JSRTDataset
from scba.data.loaders.montgomery import MontgomeryDataset
from scba.data.loaders.shenzhen import ShenzhenDataset
from scba.data.transforms.standard import get_composed_transform, get_val_transforms
from scba.models.unet import UNet
from scba.xai.cam.grad_cam_pp import GradCAMPlusPlus
from scba.xai.cam.hires_cam import HiResCAM
from scba.xai.cam.layer_cam import LayerCAM, MultiLayerCAM


METHODS = {
    "multi_layer_cam": MultiLayerCAM,
    "layer_cam": LayerCAM,
    "hires_cam": HiResCAM,
    "grad_cam_pp": GradCAMPlusPlus,
}


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr_min = float(np.min(arr))
    arr_max = float(np.max(arr))
    denom = arr_max - arr_min
    if denom < 1e-8:
        return np.zeros_like(arr)
    return (arr - arr_min) / denom


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    if np.std(a_flat) < 1e-8 or np.std(b_flat) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def load_model(model_path: Path, device: torch.device) -> UNet:
    model = UNet(n_channels=1, n_classes=2, bilinear=True).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def load_dataset(dataset_name: str, data_root: Path):
    val_transform = get_composed_transform(get_val_transforms((1024, 1024)))
    if dataset_name == "jsrt":
        return JSRTDataset(str(data_root), split="test", transform=val_transform, return_patient_id=True)
    if dataset_name == "montgomery":
        return MontgomeryDataset(str(data_root), split="test", transform=val_transform, return_patient_id=True)
    return ShenzhenDataset(str(data_root), split="test", transform=val_transform, return_patient_id=True)


def compute_correlations(dataset_name: str, data_root: Path, model_path: Path, device: torch.device) -> Dict:
    dataset = load_dataset(dataset_name, data_root)
    model = load_model(model_path, device)

    results: Dict[str, List[float]] = {name: [] for name in METHODS}
    per_sample: List[Dict[str, float]] = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        image = sample["image"].squeeze().numpy()
        patient_id = sample.get("patient_id", f"sample_{idx:03d}")

        edge_map = sobel(image)
        edge_map = _normalize(edge_map)

        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(device)

        sample_row = {"patient_id": patient_id}

        for method_name, method_cls in METHODS.items():
            explainer = method_cls(model, device=str(device))
            cam = explainer.explain(image_tensor, target_class=1).map
            cam = _normalize(cam)
            corr = _pearson_corr(cam, edge_map)
            results[method_name].append(corr)
            sample_row[method_name] = corr

        per_sample.append(sample_row)

    summary = {}
    for method_name, values in results.items():
        vals = np.array(values, dtype=float)
        summary[method_name] = {
            "mean_corr": float(np.nanmean(vals)),
            "std_corr": float(np.nanstd(vals)),
            "n": int(np.sum(~np.isnan(vals))),
        }

    return {"dataset": dataset_name, "summary": summary, "per_sample": per_sample}


def write_outputs(output_dir: Path, dataset_name: str, results: Dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"sobel_corr_{dataset_name}.json"
    csv_path = output_dir / f"sobel_corr_{dataset_name}.csv"

    with json_path.open("w") as f:
        json.dump(results, f, indent=2)

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["patient_id"] + list(METHODS.keys())
        writer.writerow(header)
        for row in results["per_sample"]:
            writer.writerow([row.get(col, "") for col in header])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute CAM versus Sobel correlation.")
    parser.add_argument("--output-dir", type=str, default="experiments/results/scba_sobel_correlation")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)

    # Scientific justification (R1): avoid hard-coded absolute paths.
    jsrt_root = os.environ.get("SCBA_JSRT_ROOT")
    mont_root = os.environ.get("SCBA_MONTGOMERY_ROOT")
    shenzhen_root = os.environ.get("SCBA_SHENZHEN_ROOT")
    if not (jsrt_root and mont_root and shenzhen_root):
        raise ValueError("Set SCBA_JSRT_ROOT, SCBA_MONTGOMERY_ROOT, and SCBA_SHENZHEN_ROOT to run reproducibly.")

    datasets = [
        {
            "name": "jsrt",
            "data_root": Path(jsrt_root),
            "model_path": Path(os.environ.get("SCBA_JSRT_MODEL_PATH", "runs/jsrt_unet_baseline_20251101_203253.pt")),
        },
        {
            "name": "montgomery",
            "data_root": Path(mont_root),
            "model_path": Path(os.environ.get("SCBA_MONTGOMERY_MODEL_PATH", "runs/montgomery_unet_baseline_20251101_203253.pt")),
        },
        {
            "name": "shenzhen",
            "data_root": Path(shenzhen_root),
            "model_path": Path(os.environ.get("SCBA_SHENZHEN_MODEL_PATH", "runs/shenzhen_unet_baseline_20260201_000000.pt")),
        },
    ]

    for cfg in datasets:
        results = compute_correlations(cfg["name"], cfg["data_root"], cfg["model_path"], device)
        write_outputs(output_dir, cfg["name"], results)


if __name__ == "__main__":
    main()
