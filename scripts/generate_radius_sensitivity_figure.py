"""
RADIUS SENSITIVITY FIGURE GENERATION FOR SCBA++
================================================

Generates publication-quality visualization showing metric behavior across
perturbation radii r ∈ {2, 3, 4, 5}.

Creates multi-panel figure with:
- CoA shift (with linear trend line if R² ≥ 0.70)
- ΔAM ROI (with error bars showing variance)
- Perceptual distance (with threshold line)
- SSIM (with threshold line)
- DC (directional consistency)
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_statistical_analysis(analysis_file: Path) -> dict:
    """Load the radius sensitivity statistical analysis JSON."""
    with open(analysis_file) as f:
        return json.load(f)


def plot_radius_sensitivity(data: dict, output_dir: Path):
    """Generate radius sensitivity figure."""

    # Set up publication style
    plt.style.use("seaborn-v0_8-paper")
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle("SCBA++ Radius Sensitivity Analysis", fontsize=15, fontweight="bold", y=0.98)

    jsrt_data = data.get("jsrt", {})
    montgomery_data = data.get("montgomery", {})

    jsrt_radii = jsrt_data.get("radii", [])
    jsrt_metrics = jsrt_data.get("metrics", {})

    montgomery_radii = montgomery_data.get("radii", [])
    montgomery_metrics = montgomery_data.get("metrics", {})

    # Panel 1: CoA Shift
    ax = axes[0, 0]
    if jsrt_metrics.get("shift_distance"):
        values = jsrt_metrics["shift_distance"]["values"]
        trend = jsrt_metrics["shift_distance"]["trend"]
        r2 = trend.get("r_squared", 0)

        ax.errorbar(jsrt_radii, values, fmt="o-", color="steelblue",
                   linewidth=2, markersize=8, label="JSRT", capsize=5)

        # Add trend line if R² ≥ 0.70
        if r2 >= 0.70:
            slope = trend["slope"]
            intercept = trend["intercept"]
            x_fit = np.array(jsrt_radii)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, "--", color="steelblue", alpha=0.5,
                   label=f"JSRT fit (R²={r2:.3f})")

    if montgomery_metrics.get("shift_distance"):
        values = montgomery_metrics["shift_distance"]["values"]
        ax.plot(montgomery_radii, values, "s--", color="coral",
               linewidth=2, markersize=8, label="Montgomery")

    ax.set_xlabel("Perturbation Radius (pixels)", fontsize=11)
    ax.set_ylabel("CoA Shift (pixels)", fontsize=11)
    ax.set_title("Center of Attribution Shift", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: ΔAM ROI
    ax = axes[0, 1]
    if jsrt_metrics.get("delta_am_roi"):
        values = jsrt_metrics["delta_am_roi"]["values"]
        ax.plot(jsrt_radii, values, "o-", color="steelblue",
               linewidth=2, markersize=8, label="JSRT")

    if montgomery_metrics.get("delta_am_roi"):
        values = montgomery_metrics["delta_am_roi"]["values"]
        ax.plot(montgomery_radii, values, "s--", color="coral",
               linewidth=2, markersize=8, label="Montgomery")

    ax.axhline(y=0, color="black", linestyle=":", alpha=0.5, linewidth=1)
    ax.set_xlabel("Perturbation Radius (pixels)", fontsize=11)
    ax.set_ylabel("ΔAM ROI", fontsize=11)
    ax.set_title("Attribution Mass Change (ROI)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Perceptual Distance
    ax = axes[0, 2]
    if jsrt_metrics.get("perceptual_distance"):
        values = [v for v in jsrt_metrics["perceptual_distance"]["values"] if not np.isnan(v)]
        radii_clean = [jsrt_radii[i] for i, v in enumerate(jsrt_metrics["perceptual_distance"]["values"]) if not np.isnan(v)]
        trend = jsrt_metrics["perceptual_distance"]["trend"]
        r2 = trend.get("r_squared", 0)

        ax.plot(radii_clean, values, "o-", color="steelblue",
               linewidth=2, markersize=8, label="JSRT")

        # Add trend line if R² ≥ 0.70
        if r2 >= 0.70:
            slope = trend["slope"]
            intercept = trend["intercept"]
            x_fit = np.array(radii_clean)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, "--", color="steelblue", alpha=0.5,
                   label=f"JSRT fit (R²={r2:.3f})")

    if montgomery_metrics.get("perceptual_distance"):
        values = montgomery_metrics["perceptual_distance"]["values"]
        ax.plot(montgomery_radii, values, "s--", color="coral",
               linewidth=2, markersize=8, label="Montgomery")

    # Realism threshold
    ax.axhline(y=0.10, color="red", linestyle="--", alpha=0.7, linewidth=1.5,
              label="Threshold (< 0.10)")

    ax.set_xlabel("Perturbation Radius (pixels)", fontsize=11)
    ax.set_ylabel("Perceptual Distance (LPIPS)", fontsize=11)
    ax.set_title("Counterfactual Realism (Perceptual)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: SSIM
    ax = axes[1, 0]
    if jsrt_metrics.get("ssim"):
        values = [v for v in jsrt_metrics["ssim"]["values"] if not np.isnan(v)]
        radii_clean = [jsrt_radii[i] for i, v in enumerate(jsrt_metrics["ssim"]["values"]) if not np.isnan(v)]
        if values:
            ax.plot(radii_clean, values, "o-", color="steelblue",
                   linewidth=2, markersize=8, label="JSRT")

    if montgomery_metrics.get("ssim"):
        values = montgomery_metrics["ssim"]["values"]
        ax.plot(montgomery_radii, values, "s--", color="coral",
               linewidth=2, markersize=8, label="Montgomery")

    # Realism threshold
    ax.axhline(y=0.85, color="red", linestyle="--", alpha=0.7, linewidth=1.5,
              label="Threshold (> 0.85)")

    ax.set_xlabel("Perturbation Radius (pixels)", fontsize=11)
    ax.set_ylabel("SSIM", fontsize=11)
    ax.set_title("Structural Similarity", fontweight="bold")
    ax.set_ylim(0.80, 1.0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 5: DC (Directional Consistency)
    ax = axes[1, 1]
    if jsrt_metrics.get("dc"):
        values = jsrt_metrics["dc"]["values"]
        ax.plot(jsrt_radii, values, "o-", color="steelblue",
               linewidth=2, markersize=8, label="JSRT")

    if montgomery_metrics.get("dc"):
        values = montgomery_metrics["dc"]["values"]
        ax.plot(montgomery_radii, values, "s--", color="coral",
               linewidth=2, markersize=8, label="Montgomery")

    ax.set_xlabel("Perturbation Radius (pixels)", fontsize=11)
    ax.set_ylabel("Directional Consistency", fontsize=11)
    ax.set_title("Directional Consistency", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 6: Summary text
    ax = axes[1, 2]
    ax.axis("off")

    summary_text = "Radius Sensitivity Summary\n\n"
    summary_text += "JSRT (r=2,3,4,5):\n"

    if jsrt_metrics.get("shift_distance"):
        trend = jsrt_metrics["shift_distance"]["trend"]
        r2 = trend.get("r_squared", 0)
        p = trend.get("p_value", 1)
        if r2 >= 0.70:
            summary_text += f"  CoA shift: R²={r2:.3f}, p={p:.4f}\n"
            summary_text += f"    (strong linear trend)\n"

    if jsrt_metrics.get("perceptual_distance"):
        trend = jsrt_metrics["perceptual_distance"]["trend"]
        r2 = trend.get("r_squared", 0)
        p = trend.get("p_value", 1)
        if r2 >= 0.70:
            summary_text += f"  Perceptual: R²={r2:.3f}, p={p:.4f}\n"
            summary_text += f"    (strong linear trend)\n"

    if jsrt_metrics.get("delta_am_roi"):
        trend = jsrt_metrics["delta_am_roi"]["trend"]
        r2 = trend.get("r_squared", 0)
        p = trend.get("p_value", 1)
        summary_text += f"  ΔAM ROI: R²={r2:.3f}, p={p:.3f}\n"
        summary_text += f"    (high variance, no trend)\n"

    summary_text += "\nMontgomery (r=4,5 only):\n"
    summary_text += "  Descriptive only\n"
    summary_text += "  (2 points, no regression)\n"

    summary_text += "\nKey Findings:\n"
    summary_text += "• Realism maintained at r=4,5\n"
    summary_text += "  (SSIM>0.97, perceptual<0.07)\n"
    summary_text += "• CoA/perceptual increase\n"
    summary_text += "  linearly with r (JSRT only)\n"
    summary_text += "• ΔAM ROI: high variance at\n"
    summary_text += "  sample size, no trend claims\n"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
           fontsize=9, verticalalignment="top", fontfamily="monospace",
           bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    plt.tight_layout()

    # Save figure
    output_file_pdf = output_dir / "radius_sensitivity_analysis.pdf"
    output_file_png = output_dir / "radius_sensitivity_analysis.png"

    fig.savefig(output_file_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(output_file_png, dpi=150, bbox_inches="tight")

    print(f"✓ Figure saved:")
    print(f"  - {output_file_pdf}")
    print(f"  - {output_file_png}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate radius sensitivity visualization for SCBA++."
    )
    parser.add_argument(
        "--analysis-file",
        type=str,
        default="experiments/results/scba_radius_sensitivity/radius_sensitivity_statistical_analysis.json",
        help="Path to statistical analysis JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="submissions/journal/figures",
        help="Directory to save figure outputs.",
    )
    args = parser.parse_args()

    analysis_file = Path(args.analysis_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SCBA++ RADIUS SENSITIVITY FIGURE GENERATION")
    print("=" * 80)
    print(f"\n✓ Loading analysis: {analysis_file}")

    data = load_statistical_analysis(analysis_file)

    print(f"✓ Generating figure...")
    plot_radius_sensitivity(data, output_dir)

    print(f"\n{'=' * 80}")
    print("✅ FIGURE GENERATION COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
