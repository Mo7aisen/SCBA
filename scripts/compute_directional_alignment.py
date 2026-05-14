#!/usr/bin/env python3
"""
Compute auxiliary Directional Alignment (DA) for existing SCBA experiments.

CRITICAL: This script is additive and preserves backward compatibility:
- It does not modify any existing experiment JSON/CSV.
- It does not rerun TPS warping / Poisson blending.
- It derives the ROI band from masks only (morphology + symmetric difference),
  matching the ROI definition used in `scba/cf/borders.py`.

DA is computed as a cosine-based alignment between:
- the CoA shift vector (stored as shift_y, shift_x in results JSON), and
- an expected direction vector toward the *nearest ROI band pixel* from the
  original CoA. This avoids the known weakness of using a global ROI centroid
  for thin, multi-component border bands.

Outputs (new files):
- experiments/results/scba_publication/directional_alignment_jsrt.json
- experiments/results/scba_publication/directional_alignment_montgomery.json
- experiments/results/scba_publication/da_statistical_analysis.json

If prior DA files already exist, they are preserved and renamed with `_proxy`
suffix before writing the new outputs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
from skimage import morphology
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scba.data.loaders.jsrt import JSRTDataset
from scba.data.loaders.montgomery import MontgomeryDataset
from scba.data.transforms.standard import get_composed_transform, get_val_transforms
from scba.metrics.cf_consistency import center_of_attribution
from scba.metrics.directional_alignment import directional_alignment_score
from scba.models.unet import UNet
from scba.xai.common import explain, get_decoder_target_layers


METHOD_ORDER = ["multi_layer_cam", "layer_cam", "hires_cam", "grad_cam_pp"]


def _load_json(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)


def _bootstrap_ci(values: np.ndarray, n_boot: int = 10000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    if values.size == 0:
        return float("nan"), float("nan")
    boots = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    # Scientific justification: DA comparisons are paired (same patient/CF keys
    # across methods), so the correct standardized effect size is paired Cohen's
    # d (d_z) computed on within-pair differences.
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size or a.size < 2:
        return float("nan")
    diffs = a - b
    sd = float(np.std(diffs, ddof=1))
    if sd < 1e-12:
        return 0.0
    return float(np.mean(diffs) / sd)


def _morph_edit(mask: np.ndarray, operation: str, radius_px: int) -> np.ndarray:
    selem = morphology.disk(int(radius_px))
    if operation == "dilate":
        out = morphology.binary_dilation(mask, selem)
    elif operation == "erode":
        out = morphology.binary_erosion(mask, selem)
    elif operation == "open":
        out = morphology.binary_opening(mask, selem)
    elif operation == "close":
        out = morphology.binary_closing(mask, selem)
    else:
        raise ValueError(f"Unknown operation: {operation}")
    return out.astype(np.uint8)


def _roi_band(mask_orig: np.ndarray, mask_pert: np.ndarray, band_px: int = 12) -> np.ndarray:
    diff = np.logical_xor(mask_orig, mask_pert).astype(np.uint8)
    selem = morphology.disk(int(band_px // 2))
    return morphology.binary_dilation(diff, selem).astype(np.uint8)


def _nearest_roi_point(roi: np.ndarray, point_yx: Tuple[float, float]) -> Tuple[float, float]:
    coords = np.argwhere(roi > 0)
    if coords.size == 0:
        return float("nan"), float("nan")
    dy = coords[:, 0].astype(float) - float(point_yx[0])
    dx = coords[:, 1].astype(float) - float(point_yx[1])
    idx = int(np.argmin(dy * dy + dx * dx))
    return float(coords[idx, 0]), float(coords[idx, 1])


def _local_roi_centroid(
    roi: np.ndarray,
    point_yx: Tuple[float, float],
    *,
    local_radius_px: float,
) -> Tuple[float, float]:
    coords = np.argwhere(roi > 0)
    if coords.size == 0:
        return float("nan"), float("nan")
    dy = coords[:, 0].astype(float) - float(point_yx[0])
    dx = coords[:, 1].astype(float) - float(point_yx[1])
    d2 = dy * dy + dx * dx
    keep = d2 <= float(local_radius_px) ** 2
    if not np.any(keep):
        return float("nan"), float("nan")
    sub = coords[keep]
    cy, cx = sub.mean(axis=0)
    return float(cy), float(cx)


def _load_model(model_path: Path, device: torch.device) -> UNet:
    model = UNet(n_channels=1, n_classes=2, bilinear=True).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _load_dataset(dataset: str) -> Tuple[object, Path, Path]:
    target_size = (1024, 1024)
    val_transform = get_composed_transform(get_val_transforms(target_size))
    if dataset == "jsrt":
        jsrt_root = os.environ.get("SCBA_JSRT_ROOT")
        if not jsrt_root:
            raise ValueError("Set SCBA_JSRT_ROOT (or add a --data-root arg) to run reproducibly.")
        return (
            JSRTDataset(
                jsrt_root,
                split="test",
                transform=val_transform,
                return_patient_id=True,
                splits_path=str(Path("experiments/results/scba_publication") / "jsrt_splits.csv"),
            ),
            Path("experiments/results/scba_publication/scba_publication_results.json"),
            Path("runs/jsrt_unet_baseline_20251101_203253.pt"),
        )
    mont_root = os.environ.get("SCBA_MONTGOMERY_ROOT")
    if not mont_root:
        raise ValueError("Set SCBA_MONTGOMERY_ROOT (or add a --data-root arg) to run reproducibly.")
    return (
        MontgomeryDataset(
            mont_root,
            split="test",
            transform=val_transform,
            return_patient_id=True,
            splits_path=str(Path("experiments/results/scba_publication_montgomery") / "montgomery_splits.csv"),
        ),
        Path("experiments/results/scba_publication_montgomery/scba_publication_montgomery_results.json"),
        Path("runs/montgomery_unet_baseline_20251101_203253.pt"),
    )


def _compute_coa_originals(
    dataset_obj,
    model: UNet,
    device: torch.device,
    patient_ids: List[str],
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    decoder_layers = get_decoder_target_layers(model, n_layers=2)
    primary_layer = decoder_layers[-1][1]
    method_kwargs = {
        "multi_layer_cam": {"target_layers": [layer for _, layer in decoder_layers]},
        "layer_cam": {"target_layer": primary_layer},
        "hires_cam": {"target_layer": primary_layer},
        "grad_cam_pp": {"target_layer": primary_layer},
    }

    by_id = {dataset_obj[i]["patient_id"]: i for i in range(len(dataset_obj))}
    coa_orig: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for pid in patient_ids:
        idx = by_id.get(pid)
        if idx is None:
            continue
        sample = dataset_obj[idx]
        image_tensor = sample["image"].unsqueeze(0).to(device)
        per_method = {}
        for method in METHOD_ORDER:
            sal = explain(
                image_tensor,
                model,
                method=method,
                target_class=1,
                device=device,
                **method_kwargs.get(method, {}),
            )
            per_method[method] = center_of_attribution(sal.map)
        coa_orig[pid] = per_method
    return coa_orig


def compute_da_for_dataset(dataset: str, device_str: str, *, local_radius_px: float) -> Dict:
    dataset_obj, results_path, model_path = _load_dataset(dataset)
    results = _load_json(results_path)
    detailed = results.get("detailed_results", {})

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model = _load_model(model_path, device)

    patient_ids = list(detailed.keys())
    coa_orig = _compute_coa_originals(dataset_obj, model, device, patient_ids)

    # Align by (patient_id, cf_key) to keep repeated-measures pairing.
    da_values: Dict[str, Dict[Tuple[str, str], float]] = {m: {} for m in METHOD_ORDER}
    dc_values: Dict[str, Dict[Tuple[str, str], float]] = {m: {} for m in METHOD_ORDER}

    by_id = {dataset_obj[i]["patient_id"]: i for i in range(len(dataset_obj))}

    for pid, methods in detailed.items():
        idx = by_id.get(pid)
        if idx is None:
            continue
        sample = dataset_obj[idx]
        mask = sample["mask"].squeeze().cpu().numpy().astype(np.uint8)
        if mask.max() > 1:
            mask = (mask > 0).astype(np.uint8)

        for method in METHOD_ORDER:
            method_results = methods.get(method, {})
            coa_o = coa_orig.get(pid, {}).get(method)
            if coa_o is None:
                continue
            for cf_key, cf in method_results.items():
                if cf_key == "baseline_metrics":
                    continue
                metrics = cf.get("metrics", {})
                shift_y = metrics.get("shift_y")
                shift_x = metrics.get("shift_x")
                dc = metrics.get("directional_consistency")
                cfg = cf.get("config", {})
                radius_px = int(cfg.get("radius_px", 0) or 0)
                operation = cfg.get("operation")
                if shift_y is None or shift_x is None or operation is None or radius_px <= 0:
                    continue

                mask_pert = _morph_edit(mask, operation, radius_px)
                roi = _roi_band(mask, mask_pert, band_px=12)
                if roi.sum() == 0:
                    continue

                coa_p = (float(coa_o[0] + float(shift_y)), float(coa_o[1] + float(shift_x)))
                roi_target = _local_roi_centroid(roi, coa_o, local_radius_px=local_radius_px)
                if np.isnan(roi_target[0]) or np.isnan(roi_target[1]):
                    roi_target = _nearest_roi_point(roi, coa_o)
                    if np.isnan(roi_target[0]) or np.isnan(roi_target[1]):
                        continue

                da = directional_alignment_score(coa_o, coa_p, roi_target)
                key = (pid, cf_key)
                da_values[method][key] = float(da)
                if dc is not None:
                    dc_values[method][key] = float(dc)

    # Convert to aligned arrays
    common_keys = set.intersection(*(set(da_values[m].keys()) for m in METHOD_ORDER))
    common_keys = sorted(common_keys)
    per_method_list = {m: np.array([da_values[m][k] for k in common_keys], dtype=float) for m in METHOD_ORDER}
    per_method_dc_list = {m: np.array([dc_values[m].get(k, float("nan")) for k in common_keys], dtype=float) for m in METHOD_ORDER}

    summary = {}
    for m in METHOD_ORDER:
        vals = per_method_list[m]
        dc_vals = per_method_dc_list[m]
        dc_vals = dc_vals[~np.isnan(dc_vals)]
        lo, hi = _bootstrap_ci(vals)
        dc_lo, dc_hi = _bootstrap_ci(dc_vals) if dc_vals.size else (float("nan"), float("nan"))
        summary[m] = {
            "n": int(vals.size),
            "mean_da": float(np.mean(vals)) if vals.size else float("nan"),
            "std_da": float(np.std(vals)) if vals.size else float("nan"),
            "median_da": float(np.median(vals)) if vals.size else float("nan"),
            "ci_lower_da": lo,
            "ci_upper_da": hi,
            "mean_dc": float(np.mean(dc_vals)) if dc_vals.size else float("nan"),
            "ci_lower_dc": dc_lo,
            "ci_upper_dc": dc_hi,
        }

    return {
        "dataset": dataset,
        "device": str(device),
        "source_results": str(results_path),
        "roi_definition": "ROI band = dilated symmetric difference of masks; band_px=12 (matches SCBA BorderEditor)",
        "da_definition": f"DA = cosine alignment between CoA shift vector and direction toward a local ROI centroid within {local_radius_px:g}px of the original CoA (fallback: nearest ROI pixel). Scores mapped to [0,1] with 0.5 = perpendicular.",
        "paired_n": int(len(common_keys)),
        "summary": summary,
        "values": {m: per_method_list[m].tolist() for m in METHOD_ORDER},
        "dc_values": {m: per_method_dc_list[m].tolist() for m in METHOD_ORDER},
        "pairing_keys": [{"patient_id": k[0], "cf_key": k[1]} for k in common_keys],
    }


def compute_stats(jsrt: Dict, mont: Dict) -> Dict:
    def _friedman(values_by_method: Dict[str, List[float]]) -> Dict[str, float]:
        arrays = [np.asarray(values_by_method[m], dtype=float) for m in METHOD_ORDER]
        min_len = min(a.size for a in arrays)
        if min_len == 0:
            return {"chi2": float("nan"), "p": float("nan"), "n": 0}
        arrays = [a[:min_len] for a in arrays]
        stat, p = friedmanchisquare(*arrays)
        return {"chi2": float(stat), "p": float(p), "n": int(min_len)}

    def _pairwise(values_by_method: Dict[str, List[float]]) -> List[Dict]:
        out = []
        arrays = {m: np.asarray(values_by_method[m], dtype=float) for m in METHOD_ORDER}
        for i in range(len(METHOD_ORDER)):
            for j in range(i + 1, len(METHOD_ORDER)):
                a = arrays[METHOD_ORDER[i]]
                b = arrays[METHOD_ORDER[j]]
                min_len = min(a.size, b.size)
                if min_len == 0:
                    continue
                a = a[:min_len]
                b = b[:min_len]
                try:
                    _w_stat, p = wilcoxon(a, b, zero_method="wilcox")
                except Exception:
                    p = float("nan")
                out.append(
                    {
                        "a": METHOD_ORDER[i],
                        "b": METHOD_ORDER[j],
                        "wilcoxon_p": float(p),
                        "cohens_d": _cohens_d(a, b),
                        "n": int(min_len),
                    }
                )
        return out

    return {
        "jsrt": {"friedman_da": _friedman(jsrt["values"]), "pairwise_da": _pairwise(jsrt["values"])},
        "montgomery": {"friedman_da": _friedman(mont["values"]), "pairwise_da": _pairwise(mont["values"])},
    }


def _preserve_existing(out_dir: Path, basename: str) -> None:
    p = out_dir / basename
    if not p.exists():
        return
    proxy = out_dir / p.with_suffix("").name
    proxy = proxy.with_name(proxy.name + "_proxy").with_suffix(".json")
    if proxy.exists():
        return
    p.rename(proxy)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute auxiliary DA for existing experiments.")
    p.add_argument("--out-dir", type=str, default="experiments/results/scba_publication")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--local-radius-px", type=float, default=120.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Preserve any prior DA drafts without overwriting.
    _preserve_existing(out_dir, "directional_alignment_jsrt.json")
    _preserve_existing(out_dir, "directional_alignment_montgomery.json")
    _preserve_existing(out_dir, "da_statistical_analysis.json")

    jsrt = compute_da_for_dataset("jsrt", args.device, local_radius_px=args.local_radius_px)
    mont = compute_da_for_dataset("montgomery", args.device, local_radius_px=args.local_radius_px)
    stats = compute_stats(jsrt, mont)

    (out_dir / "directional_alignment_jsrt.json").write_text(json.dumps(jsrt, indent=2))
    (out_dir / "directional_alignment_montgomery.json").write_text(json.dumps(mont, indent=2))
    (out_dir / "da_statistical_analysis.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
