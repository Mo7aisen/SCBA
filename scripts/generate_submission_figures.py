"""
Generate high-resolution PDF figures for SCBA submissions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scba.metrics.statistical_tests import compare_all_methods


METHOD_LABELS = {
    "multi_layer_cam": "Multi Layer CAM",
    "layer_cam": "LayerCAM",
    "hires_cam": "HiResCAM",
    "grad_cam_pp": "Grad CAM++",
}


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
            "legend.fontsize": 8,
            "figure.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    tiff_path = output_path.with_suffix(".tiff")
    fig.savefig(tiff_path, dpi=300, bbox_inches="tight")


def _load_json(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)


def _method_label(name: str) -> str:
    return METHOD_LABELS.get(name, name)


def _collect_cf_metric(detailed_results: Dict, metric_key: str) -> Dict[str, np.ndarray]:
    method_values: Dict[str, List[float]] = {}
    for sample in detailed_results.values():
        for method_name, method_results in sample.items():
            for cf_key, cf_results in method_results.items():
                if cf_key == "baseline_metrics":
                    continue
                if metric_key == "perceptual_distance":
                    value = cf_results.get("perceptual_distance")
                else:
                    value = cf_results.get("metrics", {}).get(metric_key)
                    if value is None:
                        value = cf_results.get(metric_key)
                if value is None:
                    continue
                method_values.setdefault(method_name, []).append(value)
    return {k: np.array(v, dtype=float) for k, v in method_values.items() if v}


def _collect_patient_cf_metric(
    detailed_results: Dict,
    metric_key: str,
    *,
    expected_cf_keys: List[str] | None = None,
) -> Dict[str, np.ndarray]:
    method_values: Dict[str, List[float]] = {}
    for sample in detailed_results.values():
        for method_name, method_results in sample.items():
            if expected_cf_keys and not all(k in method_results for k in expected_cf_keys):
                continue
            values = []
            for cf_key, cf_results in method_results.items():
                if cf_key == "baseline_metrics":
                    continue
                if metric_key == "perceptual_distance":
                    value = cf_results.get("perceptual_distance")
                else:
                    value = cf_results.get("metrics", {}).get(metric_key)
                    if value is None:
                        value = cf_results.get(metric_key)
                if value is None:
                    continue
                values.append(value)
            if values:
                method_values.setdefault(method_name, []).append(float(np.mean(values)))
    return {k: np.array(v, dtype=float) for k, v in method_values.items() if v}


def _collect_baseline_metric(detailed_results: Dict, metric_key: str) -> Dict[str, np.ndarray]:
    method_values: Dict[str, List[float]] = {}
    for sample in detailed_results.values():
        for method_name, method_results in sample.items():
            baseline = method_results.get("baseline_metrics", {})
            value = baseline.get(metric_key)
            if value is None:
                continue
            method_values.setdefault(method_name, []).append(value)
    return {k: np.array(v, dtype=float) for k, v in method_values.items() if v}


def plot_main_metrics(results_path: Path, output_path: Path, title: str) -> None:
    data = _load_json(results_path)
    detailed = data.get("detailed_results", {})
    stats = data.get("statistics", {})
    config = data.get("config", {})
    seed = config.get("seed", 42)
    method_order = config.get("xai_methods", [])
    method_labels = config.get("xai_method_labels", {})

    cf_configs = config.get("cf_configs", [])
    expected_cf_keys = [f"{c['operation']}_r{c['radius_px']}" for c in cf_configs] if cf_configs else None
    metric_sets = {
        "delta_am_roi": _collect_patient_cf_metric(
            detailed, "delta_am_roi", expected_cf_keys=expected_cf_keys
        ),
        "shift_distance": _collect_patient_cf_metric(
            detailed, "shift_distance", expected_cf_keys=expected_cf_keys
        ),
        "directional_consistency": _collect_patient_cf_metric(
            detailed, "directional_consistency", expected_cf_keys=expected_cf_keys
        ),
    }

    comparisons = {}
    if stats:
        for key in metric_sets.keys():
            stat_obj = stats.get(key)
            if not stat_obj:
                continue
            summary_by_key = {}
            for method_key in method_order:
                label = method_labels.get(method_key, method_key)
                if label in stat_obj.get("summary", {}):
                    summary_by_key[method_key] = stat_obj["summary"][label]
            if summary_by_key:
                comparisons[key] = {"summary": summary_by_key}

    if not comparisons:
        comparisons = {
            key: compare_all_methods(values, metric_name=key, seed=seed)
            for key, values in metric_sets.items()
            if values
        }

    if not comparisons:
        raise RuntimeError(f"No metrics found in {results_path}")

    if not method_order:
        method_order = list(next(iter(metric_sets.values())).keys())
    labels = [_method_label(name) for name in method_order]
    y_positions = np.arange(len(method_order))

    metrics = [
        ("Delta AM ROI", "delta_am_roi", "steelblue"),
        ("CoA Shift (px)", "shift_distance", "forestgreen"),
        ("Directional Consistency", "directional_consistency", "coral"),
    ]

    fig, axes = plt.subplots(
        len(metrics), 1, figsize=(5.4, 6.6), sharex=False, constrained_layout=True
    )

    for ax, (label, key, color) in zip(axes, metrics):
        summary = comparisons[key]["summary"]
        values = []
        lower_err = []
        upper_err = []
        medians = []
        for method in method_order:
            stats = summary[method]
            values.append(stats["mean"])
            lower_err.append(stats["mean"] - stats["ci_lower"])
            upper_err.append(stats["ci_upper"] - stats["mean"])
            medians.append(stats["median"])

        ax.barh(
            y_positions,
            values,
            color=color,
            alpha=0.85,
            xerr=np.array([lower_err, upper_err]),
            error_kw={"elinewidth": 1.5, "capthick": 1.5},
            capsize=4,
        )
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(label, fontsize=9)
        ax.set_title(label, fontsize=10, loc="left", pad=4)
        ax.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.8)
        if key == "delta_am_roi":
            ax.axvline(0, color="black", linewidth=0.8)
        if key == "directional_consistency":
            ax.set_xlim(0, 1)
        for y, median in zip(y_positions, medians):
            ax.plot(median, y, marker="D", color="#d62728", markersize=4)
        ax.invert_yaxis()

    fig.suptitle(title, fontsize=12, fontweight="bold")
    save_figure(fig, output_path)
    plt.close(fig)


def plot_perceptual_distributions(results_path: Path, output_path: Path, title: str) -> None:
    data = _load_json(results_path)
    detailed = data.get("detailed_results", {})
    config = data.get("config", {})
    cf_configs = config.get("cf_configs", [])
    expected_cf_keys = [f"{c['operation']}_r{c['radius_px']}" for c in cf_configs] if cf_configs else None
    perceptual = _collect_patient_cf_metric(detailed, "perceptual_distance", expected_cf_keys=expected_cf_keys)
    if not perceptual:
        raise RuntimeError(f"No perceptual distances found in {results_path}")

    method_order = list(perceptual.keys())
    labels = [_method_label(name) for name in method_order]
    values = [perceptual[name] for name in method_order]

    fig, ax = plt.subplots(figsize=(5.4, 3.4), constrained_layout=True)
    parts = ax.violinplot(values, showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_alpha(0.7)
        body.set_facecolor("#4c72b0")

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Perceptual Distance", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    save_figure(fig, output_path)
    plt.close(fig)


def plot_ablation(results_path: Path, output_path: Path, title: str) -> None:
    data = _load_json(results_path)
    ablations = data.get("ablation_results", {})
    if not ablations:
        raise RuntimeError(f"No ablation results found in {results_path}")

    config_order = list(ablations.keys())
    label_map = {
        "placeholder_warp": "Baseline",
        "tps_base": "TPS-Base",
        "tps_low_smooth": "Smooth-L",
        "tps_high_smooth": "Smooth-H",
        "tps_low_ssim": "SSIM-L",
        "tps_high_ssim": "SSIM-H",
        "tps_low_points": "Pts-L",
    }
    labels = [label_map.get(cfg, cfg.replace("_", " ")) for cfg in config_order]
    summaries = [ablations[cfg]["summary"] for cfg in config_order]

    metrics = [
        ("Delta AM ROI", "mean_delta_am_roi"),
        ("CoA Shift (px)", "mean_shift_distance"),
        ("Directional Consistency", "mean_dc"),
        ("Perceptual Distance", "mean_perceptual_distance"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.2))
    axes = axes.ravel()

    for idx, (ax, (label, key)) in enumerate(zip(axes, metrics)):
        values = [s.get(key, np.nan) for s in summaries]
        ax.bar(np.arange(len(values)), values, color="#4c72b0", alpha=0.85)
        ax.set_title(label, fontsize=10)
        ax.set_xticks(np.arange(len(labels)))
        if idx < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=8)
            ax.tick_params(axis="x", pad=4)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        if key == "mean_dc":
            ax.set_ylim(0, 1)
        ax.margins(x=0.02)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.94])
    save_figure(fig, output_path)
    plt.close(fig)


def plot_sanity(results_path: Path, output_path: Path, title: str) -> None:
    data = _load_json(results_path)
    stages = data.get("randomization_order", [])
    stage_metrics = data.get("stages", {})
    if not stages:
        raise RuntimeError(f"No stages found in {results_path}")

    corr = [stage_metrics[s]["mean_pearson_corr"] for s in stages]
    ssim = [stage_metrics[s]["mean_ssim"] for s in stages]

    fig, axes = plt.subplots(2, 1, figsize=(6.2, 4.8), sharex=True, constrained_layout=True)
    axes[0].plot(stages, corr, marker="o", color="#d62728")
    axes[0].set_ylabel("Mean Pearson r", fontsize=9)
    axes[0].grid(axis="y", alpha=0.3, linestyle="--")

    axes[1].plot(stages, ssim, marker="o", color="#2ca02c")
    axes[1].set_ylabel("Mean SSIM", fontsize=9)
    axes[1].grid(axis="y", alpha=0.3, linestyle="--")
    axes[1].set_xticks(range(len(stages)))
    axes[1].set_xticklabels(stages, rotation=35, ha="right", fontsize=8)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    save_figure(fig, output_path)
    plt.close(fig)


def plot_inter_method_correlation(
    results_path: Path, output_path: Path, title: str, *, metric_key: str = "delta_am_roi"
) -> None:
    data = _load_json(results_path)
    detailed = data.get("detailed_results", {})
    config = data.get("config", {})
    cf_configs = config.get("cf_configs", [])
    expected_cf_keys = [f"{c['operation']}_r{c['radius_px']}" for c in cf_configs] if cf_configs else None

    per_method = _collect_patient_cf_metric(
        detailed, metric_key, expected_cf_keys=expected_cf_keys
    )
    if not per_method:
        raise RuntimeError(f"No per-method data found in {results_path}")

    method_order = config.get("xai_methods", list(per_method.keys()))
    method_order = [m for m in method_order if m in per_method]
    labels = [_method_label(name) for name in method_order]

    data_mat = np.vstack([per_method[m] for m in method_order])
    corr = np.corrcoef(data_mat)

    fig, ax = plt.subplots(figsize=(5.0, 4.2), constrained_layout=True)
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr[i, j]
            color = "white" if abs(value) > 0.5 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color=color)

    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.78, label="Pearson r")
    save_figure(fig, output_path)
    plt.close(fig)


def plot_old_method_correlation(results_path: Path, output_path: Path, title: str) -> None:
    data = _load_json(results_path)
    corr = data.get("correlation_matrix", {})
    methods = corr.get("methods")
    matrix = corr.get("matrix")
    if not methods or not matrix:
        raise RuntimeError(f"No correlation_matrix in {results_path}")

    labels = [_method_label(name) for name in methods]
    matrix = np.array(matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(5.0, 4.2), constrained_layout=True)
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = matrix[i, j]
            color = "white" if abs(value) > 0.5 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color=color)

    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.78, label="Pearson r")
    save_figure(fig, output_path)
    plt.close(fig)


def plot_warp_reconciliation(jsrt_path: Path, mont_path: Path, output_path: Path) -> None:
    jsrt = _load_json(jsrt_path)
    mont = _load_json(mont_path)

    def _extract(data: Dict) -> Dict[str, Dict]:
        ablations = data.get("ablation_results", {})
        base = ablations.get("placeholder_warp", {}).get("summary", {})
        tps = ablations.get("tps_base", {}).get("summary", {})
        if not base or not tps:
            raise RuntimeError("Missing placeholder_warp or tps_base in ablation results.")
        return {"base": base, "tps": tps}

    jsrt_vals = _extract(jsrt)
    mont_vals = _extract(mont)

    metrics = [
        ("Delta AM ROI", "mean_delta_am_roi"),
        ("Directional Consistency", "mean_dc"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), constrained_layout=True)
    for ax, title, values in [
        (axes[0], "JSRT", jsrt_vals),
        (axes[1], "Montgomery", mont_vals),
    ]:
        base = values["base"]
        tps = values["tps"]
        base_vals = [base.get(key, np.nan) for _, key in metrics]
        tps_vals = [tps.get(key, np.nan) for _, key in metrics]
        x = np.arange(len(metrics))
        width = 0.36
        ax.bar(x - width / 2, base_vals, width, label="OLD (Morph)", color="#4c72b0", alpha=0.85)
        ax.bar(x + width / 2, tps_vals, width, label="NEW (TPS)", color="#dd8452", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([m[0] for m in metrics], rotation=20, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.margins(x=0.05)
        ax.set_ylim(0, max(0.6, max(base_vals + tps_vals) * 1.2))

    fig.suptitle("OLD vs NEW Reconciliation (Warp Ablation)", fontsize=12, fontweight="bold")
    axes[0].legend(loc="upper left", fontsize=8)
    save_figure(fig, output_path)
    plt.close(fig)


def plot_sobel_correlation(
    jsrt_path: Path,
    montgomery_path: Path,
    output_path: Path,
    shenzhen_path: Path | None = None,
) -> None:
    datasets = [
        ("JSRT", _load_json(jsrt_path)),
        ("Montgomery", _load_json(montgomery_path)),
    ]
    if shenzhen_path is not None and shenzhen_path.exists():
        datasets.append(("Shenzhen", _load_json(shenzhen_path)))

    methods = list(METHOD_LABELS.keys())
    labels = [_method_label(m) for m in methods]
    x = np.arange(len(methods))

    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(3.8 * len(datasets), 3.2),
        constrained_layout=True,
        sharey=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, (title, data) in zip(axes, datasets):
        means = [data["summary"][m]["mean_corr"] for m in methods]
        stds = [data["summary"][m]["std_corr"] for m in methods]
        ax.bar(
            x,
            means,
            yerr=stds,
            color="#4c72b0",
            alpha=0.85,
            error_kw={"elinewidth": 1.5, "capthick": 1.5},
            capsize=4,
        )
        ax.axhline(0.30, color="#d62728", linestyle="--", linewidth=1.5)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=8)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_ylim(0, max(0.35, max(means) + max(stds) + 0.05))

    axes[0].set_ylabel("Pearson r with Sobel edges", fontsize=9)
    axes[-1].text(
        0.02,
        0.92,
        "Adebayo et al. r≈0.30",
        transform=axes[-1].transAxes,
        fontsize=8,
        color="#d62728",
        va="top",
    )

    fig.suptitle("Sobel Edge Correlation Sanity Check", fontsize=12, fontweight="bold")
    save_figure(fig, output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SCBA submission figures.")
    parser.add_argument("--output-dir", type=str, default="submissions/journal/figures")
    parser.add_argument("--jsrt-results", type=str, default="experiments/results/scba_publication/scba_publication_results.json")
    parser.add_argument("--montgomery-results", type=str, default="experiments/results/scba_publication_montgomery/scba_publication_montgomery_results.json")
    parser.add_argument("--shenzhen-results", type=str, default="")
    parser.add_argument("--cross-jsrt-to-mont", type=str, default="experiments/results/scba_cross_jsrt_to_montgomery/scba_publication_montgomery_results.json")
    parser.add_argument("--cross-mont-to-jsrt", type=str, default="experiments/results/scba_cross_montgomery_to_jsrt/scba_publication_results.json")
    parser.add_argument("--ablation-jsrt", type=str, default="experiments/results/scba_ablation/scba_ablation_jsrt.json")
    parser.add_argument("--ablation-mont", type=str, default="experiments/results/scba_ablation/scba_ablation_montgomery.json")
    parser.add_argument("--ablation-shenzhen", type=str, default="")
    parser.add_argument("--sanity-jsrt", type=str, default="experiments/results/scba_sanity_randomization/scba_randomization_jsrt.json")
    parser.add_argument("--sanity-mont", type=str, default="experiments/results/scba_sanity_randomization/scba_randomization_montgomery.json")
    parser.add_argument("--sanity-shenzhen", type=str, default="")
    parser.add_argument("--sobel-jsrt", type=str, default="experiments/results/scba_sobel_correlation/sobel_corr_jsrt.json")
    parser.add_argument("--sobel-mont", type=str, default="experiments/results/scba_sobel_correlation/sobel_corr_montgomery.json")
    parser.add_argument("--sobel-shenzhen", type=str, default="")
    parser.add_argument("--old-jsrt", type=str, default="")
    parser.add_argument("--old-mont", type=str, default="")
    return parser.parse_args()


def main() -> None:
    configure_matplotlib()
    args = parse_args()
    output_dir = Path(args.output_dir)

    jsrt_results = Path(args.jsrt_results)
    mont_results = Path(args.montgomery_results)
    shenzhen_results = Path(args.shenzhen_results) if args.shenzhen_results else None
    cross_jsrt_to_mont = Path(args.cross_jsrt_to_mont)
    cross_mont_to_jsrt = Path(args.cross_mont_to_jsrt)
    ablation_jsrt = Path(args.ablation_jsrt)
    ablation_mont = Path(args.ablation_mont)
    ablation_shenzhen = Path(args.ablation_shenzhen) if args.ablation_shenzhen else None
    sanity_jsrt = Path(args.sanity_jsrt)
    sanity_mont = Path(args.sanity_mont)
    sanity_shenzhen = Path(args.sanity_shenzhen) if args.sanity_shenzhen else None
    sobel_jsrt = Path(args.sobel_jsrt)
    sobel_mont = Path(args.sobel_mont)
    sobel_shenzhen = Path(args.sobel_shenzhen) if args.sobel_shenzhen else None
    old_jsrt = Path(args.old_jsrt) if args.old_jsrt else None
    old_mont = Path(args.old_mont) if args.old_mont else None

    if jsrt_results.exists():
        plot_main_metrics(jsrt_results, output_dir / "jsrt_main_metrics.pdf", "SCBA Metrics for JSRT")
        plot_perceptual_distributions(
            jsrt_results, output_dir / "jsrt_perceptual_distribution.pdf", "Perceptual Distance for JSRT"
        )
        plot_inter_method_correlation(
            jsrt_results,
            output_dir / "jsrt_inter_method_correlation.pdf",
            "Inter-Method Correlation (ΔAM-ROI) - JSRT",
        )
    else:
        print(f"⚠ Missing JSRT results: {jsrt_results}")

    if mont_results.exists():
        plot_main_metrics(
            mont_results, output_dir / "montgomery_main_metrics.pdf", "SCBA Metrics for Montgomery"
        )
        plot_perceptual_distributions(
            mont_results,
            output_dir / "montgomery_perceptual_distribution.pdf",
            "Perceptual Distance for Montgomery",
        )
        plot_inter_method_correlation(
            mont_results,
            output_dir / "montgomery_inter_method_correlation.pdf",
            "Inter-Method Correlation (ΔAM-ROI) - Montgomery",
        )
    else:
        print(f"⚠ Missing Montgomery results: {mont_results}")

    if shenzhen_results and shenzhen_results.exists():
        plot_main_metrics(
            shenzhen_results, output_dir / "shenzhen_main_metrics.pdf", "SCBA Metrics for Shenzhen"
        )
        plot_perceptual_distributions(
            shenzhen_results,
            output_dir / "shenzhen_perceptual_distribution.pdf",
            "Perceptual Distance for Shenzhen",
        )
        plot_inter_method_correlation(
            shenzhen_results,
            output_dir / "shenzhen_inter_method_correlation.pdf",
            "Inter-Method Correlation (ΔAM-ROI) - Shenzhen",
        )
    elif shenzhen_results:
        print(f"⚠ Missing Shenzhen results: {shenzhen_results}")

    if cross_jsrt_to_mont.exists():
        plot_main_metrics(
            cross_jsrt_to_mont,
            output_dir / "cross_jsrt_to_mont_metrics.pdf",
            "Cross Dataset JSRT to Montgomery",
        )
    else:
        print(f"⚠ Missing cross JSRT→Montgomery results: {cross_jsrt_to_mont}")

    if cross_mont_to_jsrt.exists():
        plot_main_metrics(
            cross_mont_to_jsrt,
            output_dir / "cross_mont_to_jsrt_metrics.pdf",
            "Cross Dataset Montgomery to JSRT",
        )
    else:
        print(f"⚠ Missing cross Montgomery→JSRT results: {cross_mont_to_jsrt}")

    if ablation_jsrt.exists():
        plot_ablation(ablation_jsrt, output_dir / "jsrt_ablation.pdf", "Ablation Summary for JSRT")
    else:
        print(f"⚠ Missing JSRT ablation results: {ablation_jsrt}")

    if ablation_mont.exists():
        plot_ablation(
            ablation_mont, output_dir / "montgomery_ablation.pdf", "Ablation Summary for Montgomery"
        )
    else:
        print(f"⚠ Missing Montgomery ablation results: {ablation_mont}")

    if ablation_shenzhen and ablation_shenzhen.exists():
        plot_ablation(
            ablation_shenzhen, output_dir / "shenzhen_ablation.pdf", "Ablation Summary for Shenzhen"
        )
    elif ablation_shenzhen:
        print(f"⚠ Missing Shenzhen ablation results: {ablation_shenzhen}")

    if ablation_jsrt.exists() and ablation_mont.exists():
        plot_warp_reconciliation(
            ablation_jsrt,
            ablation_mont,
            output_dir / "reconciliation_old_vs_new.pdf",
        )

    if sanity_jsrt.exists():
        plot_sanity(
            sanity_jsrt, output_dir / "jsrt_sanity_randomization.pdf", "Sanity Check for JSRT"
        )
    else:
        print(f"⚠ Missing JSRT sanity results: {sanity_jsrt}")

    if sanity_mont.exists():
        plot_sanity(
            sanity_mont,
            output_dir / "montgomery_sanity_randomization.pdf",
            "Sanity Check for Montgomery",
        )
    else:
        print(f"⚠ Missing Montgomery sanity results: {sanity_mont}")

    if sanity_shenzhen and sanity_shenzhen.exists():
        plot_sanity(
            sanity_shenzhen,
            output_dir / "shenzhen_sanity_randomization.pdf",
            "Sanity Check for Shenzhen",
        )
    elif sanity_shenzhen:
        print(f"⚠ Missing Shenzhen sanity results: {sanity_shenzhen}")

    if sobel_jsrt.exists() and sobel_mont.exists():
        plot_sobel_correlation(
            sobel_jsrt,
            sobel_mont,
            output_dir / "sobel_correlation.pdf",
            shenzhen_path=sobel_shenzhen if sobel_shenzhen and sobel_shenzhen.exists() else None,
        )
    else:
        print(f"⚠ Missing Sobel correlation results: {sobel_jsrt} / {sobel_mont}")

    if old_jsrt and old_jsrt.exists():
        plot_old_method_correlation(
            old_jsrt,
            output_dir / "old_method_correlation_jsrt.pdf",
            "Old Method Inter-Method Correlation (JSRT)",
        )
    elif old_jsrt:
        print(f"⚠ Missing old-methodology JSRT results: {old_jsrt}")

    if old_mont and old_mont.exists():
        plot_old_method_correlation(
            old_mont,
            output_dir / "old_method_correlation_montgomery.pdf",
            "Old Method Inter-Method Correlation (Montgomery)",
        )
    elif old_mont:
        print(f"⚠ Missing old-methodology Montgomery results: {old_mont}")


if __name__ == "__main__":
    main()
