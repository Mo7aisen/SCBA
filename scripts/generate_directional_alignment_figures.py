#!/usr/bin/env python3
"""
Generate publication-grade figures for auxiliary Directional Alignment (DA).

Outputs:
- submissions/journal/figures/directional_alignment_comparison.pdf (+.tiff)
- submissions/conference/figures/directional_alignment_comparison.pdf (+.tiff)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
    fig.savefig(output_path.with_suffix(".tiff"), dpi=300, bbox_inches="tight")


def _load(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)


def _extract(summary: Dict, key_mean: str, key_lo: str, key_hi: str) -> Tuple[np.ndarray, np.ndarray]:
    means = []
    errs = []
    for method in METHOD_LABELS:
        s = summary[method]
        mean = float(s[key_mean])
        lo = float(s[key_lo])
        hi = float(s[key_hi])
        means.append(mean)
        errs.append([mean - lo, hi - mean])
    return np.array(means, dtype=float), np.array(errs, dtype=float).T


def plot_dc_vs_da(jsrt_path: Path, mont_path: Path, output_path: Path) -> None:
    jsrt = _load(jsrt_path)
    mont = _load(mont_path)

    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.6), constrained_layout=True, sharey="row")
    datasets = [("JSRT", jsrt), ("Montgomery", mont)]

    x = np.arange(len(METHOD_LABELS))
    labels = [METHOD_LABELS[m] for m in METHOD_LABELS]

    for row, (ds_name, ds) in enumerate(datasets):
        dc_means, dc_err = _extract(ds["summary"], "mean_dc", "ci_lower_dc", "ci_upper_dc")
        da_means, da_err = _extract(ds["summary"], "mean_da", "ci_lower_da", "ci_upper_da")

        ax_dc = axes[row, 0]
        ax_da = axes[row, 1]

        ax_dc.bar(
            x,
            dc_means,
            yerr=dc_err,
            color="#4c72b0",
            alpha=0.85,
            capsize=4,
            error_kw={"elinewidth": 1.5, "capthick": 1.5},
        )
        ax_da.bar(
            x,
            da_means,
            yerr=da_err,
            color="#55a868",
            alpha=0.85,
            capsize=4,
            error_kw={"elinewidth": 1.5, "capthick": 1.5},
        )

        ax_dc.set_title(f"{ds_name}: DC (binary)", fontsize=10)
        ax_da.set_title(f"{ds_name}: DA (continuous)", fontsize=10)

        for ax in (ax_dc, ax_da):
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=8)
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            ax.set_ylim(0, 1)

        ax_da.axhline(0.5, color="#444444", linestyle=":", linewidth=1.2)
        ax_da.text(
            0.02,
            0.92,
            "0.5 = perpendicular",
            transform=ax_da.transAxes,
            fontsize=8,
            color="#444444",
            va="top",
        )

    axes[0, 0].set_ylabel("Score", fontsize=9)
    axes[1, 0].set_ylabel("Score", fontsize=9)

    fig.suptitle("Directional Consistency (DC) vs. Auxiliary Directional Alignment (DA)", fontsize=12, fontweight="bold")
    save_figure(fig, output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate DA comparison figures.")
    p.add_argument(
        "--jsrt-da",
        type=str,
        default="experiments/results/scba_publication/directional_alignment_jsrt.json",
    )
    p.add_argument(
        "--mont-da",
        type=str,
        default="experiments/results/scba_publication/directional_alignment_montgomery.json",
    )
    p.add_argument("--output", type=str, default="submissions/journal/figures/directional_alignment_comparison.pdf")
    return p.parse_args()


def main() -> None:
    configure_matplotlib()
    args = parse_args()
    plot_dc_vs_da(Path(args.jsrt_da), Path(args.mont_da), Path(args.output))


if __name__ == "__main__":
    main()
