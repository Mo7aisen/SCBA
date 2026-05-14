"""
RADIUS SENSITIVITY STATISTICAL ANALYSIS FOR SCBA++
====================================================

Aggregates and characterizes counterfactual consistency metrics across
perturbation radii r ∈ {2, 3, 4, 5} for dilate operations.

Inputs:
- Publication results (r=2, r=3): experiments/results/scba_publication/scba_publication_results.json
- Radius sensitivity results (r=4, r=5): experiments/results/scba_radius_sensitivity/*_radius_sensitivity.json

Outputs:
- Statistical summary with confidence intervals
- Trend characterization (linear fit R² reported only if high)
- JSON results for manuscript integration

Approach:
- Characterization, not validation
- Report observed trends with 95% CIs
- Compute R² for all metrics but only emphasize when ≥ 0.70
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def extract_publication_data(pub_file: Path, dataset: str, method: str = "multi_layer_cam") -> Dict[int, Dict]:
    """
    Extract r=2 and r=3 dilate operation metrics from publication results.

    Returns:
        Dict mapping radius -> {mean_*, std_*, n_experiments}
    """
    with open(pub_file) as f:
        data = json.load(f)

    results = {}

    # Extract from detailed_results and aggregate
    if "detailed_results" not in data:
        return results

    detailed_results = data["detailed_results"]

    # Identify CF keys by examining first sample
    sample_ids = list(detailed_results.keys())
    if not sample_ids:
        return results

    first_sample = detailed_results[sample_ids[0]]
    if method not in first_sample:
        return results

    # Get dilate CF keys
    cf_keys = [k for k in first_sample[method].keys() if k.startswith("dilate_r")]

    for cf_key in cf_keys:
        # Parse radius from key like "dilate_r2" or "dilate_r3"
        if "_r" in cf_key:
            try:
                radius = int(cf_key.split("_r")[1])
            except (IndexError, ValueError):
                continue

            # Collect metrics across all samples
            delta_am_list = []
            shift_list = []
            dc_list = []
            ssim_list = []
            perceptual_list = []

            for sample_id, sample_data in detailed_results.items():
                if method not in sample_data or cf_key not in sample_data[method]:
                    continue

                cf_data = sample_data[method][cf_key]
                metrics = cf_data.get("metrics", {})

                # Compute ΔAM ROI
                am_roi_orig = metrics.get("am_roi_original", np.nan)
                am_roi_pert = metrics.get("am_roi_perturbed", np.nan)
                if not np.isnan(am_roi_orig) and not np.isnan(am_roi_pert):
                    delta_am_list.append(am_roi_pert - am_roi_orig)

                # Other metrics
                shift_list.append(metrics.get("shift_distance", np.nan))
                dc_list.append(metrics.get("directional_consistency", np.nan))
                ssim_list.append(cf_data.get("ssim", np.nan))

                perceptual_dist = cf_data.get("perceptual_distance")
                if perceptual_dist is not None:
                    perceptual_list.append(perceptual_dist)

            # Remove NaN values and compute statistics
            delta_am_clean = [x for x in delta_am_list if not np.isnan(x)]
            shift_clean = [x for x in shift_list if not np.isnan(x)]
            dc_clean = [x for x in dc_list if not np.isnan(x)]
            ssim_clean = [x for x in ssim_list if not np.isnan(x)]

            results[radius] = {
                "mean_delta_am_roi": float(np.mean(delta_am_clean)) if delta_am_clean else np.nan,
                "std_delta_am_roi": float(np.std(delta_am_clean)) if delta_am_clean else np.nan,
                "mean_shift_distance": float(np.mean(shift_clean)) if shift_clean else np.nan,
                "std_shift_distance": float(np.std(shift_clean)) if shift_clean else np.nan,
                "mean_dc": float(np.mean(dc_clean)) if dc_clean else np.nan,
                "std_dc": float(np.std(dc_clean)) if dc_clean else np.nan,
                "mean_ssim": float(np.mean(ssim_clean)) if ssim_clean else np.nan,
                "std_ssim": float(np.std(ssim_clean)) if ssim_clean else np.nan,
                "mean_perceptual_distance": float(np.mean(perceptual_list)) if perceptual_list else np.nan,
                "n_experiments": len(delta_am_clean),
            }

    return results


def extract_radius_sensitivity_data(radius_file: Path, method: str = "multi_layer_cam") -> Dict[int, Dict]:
    """
    Extract r=4 and r=5 metrics from radius sensitivity results.

    Returns:
        Dict mapping radius -> {mean_*, std_*, n_experiments}
    """
    with open(radius_file) as f:
        data = json.load(f)

    results = {}

    if "method_aggregates" in data and method in data["method_aggregates"]:
        method_data = data["method_aggregates"][method]

        for cf_key, cf_data in method_data.items():
            radius = cf_data.get("radius")
            if radius is not None:
                results[radius] = {
                    "mean_delta_am_roi": cf_data.get("mean_delta_am_roi", np.nan),
                    "std_delta_am_roi": cf_data.get("std_delta_am_roi", np.nan),
                    "mean_shift_distance": cf_data.get("mean_shift_distance", np.nan),
                    "std_shift_distance": cf_data.get("std_shift_distance", np.nan),
                    "mean_dc": cf_data.get("mean_dc", np.nan),
                    "std_dc": cf_data.get("std_dc", np.nan),
                    "mean_ssim": cf_data.get("mean_ssim", np.nan),
                    "std_ssim": cf_data.get("std_ssim", np.nan),
                    "mean_perceptual_distance": cf_data.get("mean_perceptual_distance", np.nan),
                    "n_experiments": cf_data.get("n_experiments", 0),
                }

    return results


def merge_radius_data(pub_data: Dict[int, Dict], radius_data: Dict[int, Dict]) -> Dict[int, Dict]:
    """Merge publication and radius sensitivity data."""
    merged = {}
    merged.update(pub_data)
    merged.update(radius_data)
    return merged


def compute_confidence_intervals(values: List[float], n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute confidence interval for mean using t-distribution.

    Returns:
        (lower_bound, upper_bound)
    """
    if len(values) == 0 or n <= 1:
        return (np.nan, np.nan)

    mean = np.mean(values)
    std = np.std(values, ddof=1)
    sem = std / np.sqrt(n)

    # t-distribution critical value
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_crit * sem

    return (mean - margin, mean + margin)


def fit_linear_trend(radii: List[int], values: List[float]) -> Dict:
    """
    Fit linear trend and compute R² and p-value.

    Returns:
        Dict with slope, intercept, r_squared, p_value
    """
    if len(radii) < 3:
        return {"slope": np.nan, "intercept": np.nan, "r_squared": np.nan, "p_value": np.nan}

    # Remove NaN values
    valid_idx = ~np.isnan(values)
    if np.sum(valid_idx) < 3:
        return {"slope": np.nan, "intercept": np.nan, "r_squared": np.nan, "p_value": np.nan}

    radii_clean = np.array(radii)[valid_idx]
    values_clean = np.array(values)[valid_idx]

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(radii_clean, values_clean)
    r_squared = r_value ** 2

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "p_value": float(p_value),
        "std_err": float(std_err),
    }


def analyze_dataset(dataset_name: str, pub_file: Path, radius_file: Path) -> Dict:
    """
    Analyze radius sensitivity for a single dataset.

    Returns:
        Dict with aggregated statistics and trend analysis
    """
    print(f"\n{'=' * 80}")
    print(f"ANALYZING {dataset_name.upper()}")
    print(f"{'=' * 80}")

    method = "multi_layer_cam"  # Primary method

    # Load data
    pub_data = extract_publication_data(pub_file, dataset_name, method)
    radius_data = extract_radius_sensitivity_data(radius_file, method)
    all_data = merge_radius_data(pub_data, radius_data)

    if not all_data:
        print(f"Warning: No data found for {dataset_name}")
        return {}

    # Sort by radius
    radii = sorted(all_data.keys())
    print(f"\n✓ Found data for radii: {radii}")

    # Extract metric arrays
    metrics = {
        "delta_am_roi": [],
        "shift_distance": [],
        "dc": [],
        "ssim": [],
        "perceptual_distance": [],
    }

    n_samples = []

    for r in radii:
        data = all_data[r]
        metrics["delta_am_roi"].append(data["mean_delta_am_roi"])
        metrics["shift_distance"].append(data["mean_shift_distance"])
        metrics["dc"].append(data["mean_dc"])
        metrics["ssim"].append(data["mean_ssim"])
        metrics["perceptual_distance"].append(data["mean_perceptual_distance"])
        n_samples.append(data["n_experiments"])

    # Compute confidence intervals (using first radius n as representative)
    n_rep = n_samples[0] if n_samples else 0

    results = {
        "dataset": dataset_name,
        "radii": radii,
        "n_samples": n_samples,
        "metrics": {},
    }

    # Build dynamic header based on available radii
    header_parts = [f"{'Metric':<25}"]
    for r in radii:
        header_parts.append(f"{'r=' + str(r):<12}")
    header_parts.extend([f"{'Trend R²':<10}", f"{'p-value':<10}"])
    header = " ".join(header_parts)
    print(f"\n{header}")
    print("-" * len(header))

    for metric_name, values in metrics.items():
        # Fit linear trend
        trend = fit_linear_trend(radii, values)

        # Determine if R² is "high" (≥ 0.70)
        r2_str = f"{trend['r_squared']:.3f}"
        if trend['r_squared'] >= 0.70:
            r2_str = f"{r2_str} (high)"

        # Format p-value
        p_str = f"{trend['p_value']:.4f}" if not np.isnan(trend['p_value']) else "N/A"

        # Print summary row with dynamic radius columns
        row_parts = [f"{metric_name:<25}"]
        val_strs = [f"{v:.5f}" if not np.isnan(v) else "N/A" for v in values]
        for val_str in val_strs:
            row_parts.append(f"{val_str:<12}")
        row_parts.extend([f"{r2_str:<10}", f"{p_str:<10}"])
        print(" ".join(row_parts))

        results["metrics"][metric_name] = {
            "values": values,
            "trend": trend,
        }

    # Add detailed data for each radius
    results["by_radius"] = all_data

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze radius sensitivity trends for SCBA++."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/scba_radius_sensitivity",
        help="Directory to save analysis outputs.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SCBA++ RADIUS SENSITIVITY STATISTICAL ANALYSIS")
    print("=" * 80)

    # File paths
    jsrt_pub = Path("experiments/results/scba_publication/scba_publication_results.json")
    jsrt_radius = Path("experiments/results/scba_radius_sensitivity/jsrt_radius_sensitivity.json")

    # Check for Montgomery publication results
    montgomery_pub_candidates = [
        Path("experiments/results/scba_publication/montgomery_publication_results.json"),
        Path("experiments/results/scba_publication/scba_publication_results_montgomery.json"),
    ]
    montgomery_pub = None
    for candidate in montgomery_pub_candidates:
        if candidate.exists():
            montgomery_pub = candidate
            break

    montgomery_radius = Path("experiments/results/scba_radius_sensitivity/montgomery_radius_sensitivity.json")

    # Analyze JSRT
    jsrt_results = {}
    if jsrt_pub.exists() and jsrt_radius.exists():
        jsrt_results = analyze_dataset("JSRT", jsrt_pub, jsrt_radius)
    else:
        print(f"Warning: JSRT data files not found")
        print(f"  Publication: {jsrt_pub.exists()}")
        print(f"  Radius sensitivity: {jsrt_radius.exists()}")

    # Analyze Montgomery
    montgomery_results = {}
    if montgomery_pub and montgomery_radius.exists():
        montgomery_results = analyze_dataset("Montgomery", montgomery_pub, montgomery_radius)
    elif montgomery_radius.exists():
        print(f"\nNote: Montgomery publication results not found, using only r=4,5 data")
        # Extract just r=4,5 data
        method = "multi_layer_cam"
        radius_data = extract_radius_sensitivity_data(montgomery_radius, method)
        radii = sorted(radius_data.keys())

        print(f"\n{'=' * 80}")
        print(f"ANALYZING MONTGOMERY (r=4,5 only)")
        print(f"{'=' * 80}")
        print(f"\n✓ Found data for radii: {radii}")

        montgomery_results = {
            "dataset": "Montgomery",
            "radii": radii,
            "metrics": {},
            "by_radius": radius_data,
        }

        # Build dynamic header
        header_parts = [f"{'Metric':<25}"]
        for r in radii:
            header_parts.append(f"{'r=' + str(r):<12}")
        header = " ".join(header_parts)
        print(f"\n{header}")
        print("-" * len(header))

        for metric_key in ["delta_am_roi", "shift_distance", "dc", "ssim", "perceptual_distance"]:
            mean_key = f"mean_{metric_key}"
            values = [radius_data[r][mean_key] for r in radii]
            val_strs = [f"{v:.5f}" if not np.isnan(v) else "N/A" for v in values]
            row_parts = [f"{metric_key:<25}"]
            for val_str in val_strs:
                row_parts.append(f"{val_str:<12}")
            print(" ".join(row_parts))

            montgomery_results["metrics"][metric_key] = {
                "values": values,
                "trend": {"note": "Insufficient radii for trend analysis (need r=2,3 data)"},
            }
    else:
        print(f"Warning: Montgomery data files not found")

    # Save combined results
    output_file = output_dir / "radius_sensitivity_statistical_analysis.json"
    combined_results = {
        "jsrt": jsrt_results,
        "montgomery": montgomery_results,
        "summary": {
            "analysis_date": "2026-01-15",
            "radii_analyzed": [2, 3, 4, 5],
            "method": "multi_layer_cam",
            "interpretation": {
                "r_squared_threshold": 0.70,
                "note": "R² ≥ 0.70 indicates strong linear trend; lower values suggest non-linear or high-variance behavior.",
            },
        },
    }

    with open(output_file, "w") as f:
        json.dump(combined_results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"✅ ANALYSIS COMPLETE")
    print(f"{'=' * 80}")
    print(f"✓ Results saved to: {output_file}")

    # Print key findings
    print(f"\n{'=' * 80}")
    print("KEY FINDINGS")
    print(f"{'=' * 80}")

    if jsrt_results:
        print("\nJSRT:")
        for metric_name, metric_data in jsrt_results.get("metrics", {}).items():
            trend = metric_data.get("trend", {})
            r2 = trend.get("r_squared", np.nan)
            if r2 >= 0.70:
                print(f"  - {metric_name}: Strong linear trend (R² = {r2:.3f})")

    if montgomery_results:
        print("\nMontgomery:")
        for metric_name, metric_data in montgomery_results.get("metrics", {}).items():
            trend = metric_data.get("trend", {})
            if isinstance(trend, dict) and "r_squared" in trend:
                r2 = trend.get("r_squared", np.nan)
                if r2 >= 0.70:
                    print(f"  - {metric_name}: Strong linear trend (R² = {r2:.3f})")

    print()


if __name__ == "__main__":
    main()
