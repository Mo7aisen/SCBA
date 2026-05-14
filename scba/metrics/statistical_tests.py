"""
Statistical Testing Module for SCBA Results.

Implements bootstrap confidence intervals, paired tests, and effect sizes
for rigorous statistical comparison of XAI methods.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
import hashlib


def bootstrap_confidence_interval(
    data: np.ndarray,
    n_iterations: int = 10000,
    confidence_level: float = 0.95,
    statistic_func: callable = np.mean,
    seed: Optional[int] = 42,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        data: 1D array of observations
        n_iterations: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        statistic_func: Function to compute statistic (default: mean)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (lower_bound, point_estimate, upper_bound)
    """
    if len(data) == 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)

    n = len(data)
    bootstrap_stats = []

    for _ in range(n_iterations):
        # Resample with replacement
        bootstrap_sample = rng.choice(data, size=n, replace=True)
        stat = statistic_func(bootstrap_sample)
        bootstrap_stats.append(stat)

    bootstrap_stats = np.array(bootstrap_stats)

    # Compute percentile confidence interval
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
    point_estimate = statistic_func(data)

    return lower, point_estimate, upper


def aggregate_by_cluster(
    values: np.ndarray,
    clusters: np.ndarray,
    reducer: callable = np.mean,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate repeated measurements to a cluster-level summary.

    Scientific justification: when multiple perturbations/CFs exist per patient,
    treating each as independent causes pseudo-replication and inflated
    significance. Aggregating within patient restores the correct unit of
    inference.

    Args:
        values: 1D array of per-observation values (e.g., per CF).
        clusters: 1D array of cluster IDs (e.g., patient_id) aligned with values.
        reducer: function applied within each cluster (default: mean).

    Returns:
        (unique_clusters, aggregated_values) in a stable, sorted cluster order.
    """
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    if values.shape[0] != clusters.shape[0]:
        raise ValueError("values and clusters must have the same length")
    unique_clusters = np.unique(clusters)
    aggregated = np.array([reducer(values[clusters == c]) for c in unique_clusters], dtype=float)
    return unique_clusters, aggregated


def cluster_bootstrap_confidence_interval(
    values: np.ndarray,
    clusters: np.ndarray,
    n_iterations: int = 10000,
    confidence_level: float = 0.95,
    statistic_func: callable = np.mean,
    reducer: callable = np.mean,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Cluster bootstrap CI where resampling occurs at the cluster level.

    Scientific justification: for correlated repeated measures within patients,
    the bootstrap must resample patients (clusters), not individual CF rows.

    Args:
        values: per-observation values.
        clusters: per-observation cluster IDs (same length as values).
        n_iterations: number of bootstrap replicates.
        confidence_level: e.g., 0.95.
        statistic_func: statistic over cluster-aggregated values (default: mean).
        reducer: within-cluster reducer to obtain patient-level values (default: mean).
        seed: RNG seed for determinism.

    Returns:
        (lower, point_estimate, upper) for statistic_func over cluster-level values.
    """
    _cluster_ids, aggregated = aggregate_by_cluster(values, clusters, reducer=reducer)
    if aggregated.size == 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n = int(aggregated.size)
    boots = np.empty(n_iterations, dtype=float)
    for i in range(n_iterations):
        sample = rng.choice(aggregated, size=n, replace=True)
        boots[i] = float(statistic_func(sample))

    alpha = 1.0 - float(confidence_level)
    lower = float(np.percentile(boots, alpha / 2 * 100))
    upper = float(np.percentile(boots, (1 - alpha / 2) * 100))
    point_estimate = float(statistic_func(aggregated))
    return lower, point_estimate, upper


def paired_t_test(
    data1: np.ndarray,
    data2: np.ndarray,
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """
    Paired t-test for comparing two methods.

    Args:
        data1: Observations from method 1
        data2: Observations from method 2 (paired)
        alternative: "two-sided", "greater", or "less"

    Returns:
        Dict with t_statistic, p_value, degrees_of_freedom, mean_diff
    """
    if len(data1) != len(data2):
        raise ValueError("Data arrays must have same length for paired test")

    data1 = np.asarray(data1, dtype=float)
    data2 = np.asarray(data2, dtype=float)
    differences = data1 - data2
    if np.allclose(differences, 0.0, rtol=0.0, atol=1e-12):
        return {
            "t_statistic": 0.0,
            "p_value": 1.0,
            "degrees_of_freedom": len(data1) - 1,
            "mean_difference": 0.0,
            "std_difference": 0.0,
        }

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(data1, data2, alternative=alternative)

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "degrees_of_freedom": len(data1) - 1,
        "mean_difference": float(np.mean(differences)),
        "std_difference": float(np.std(differences, ddof=1)),
    }


def wilcoxon_signed_rank_test(
    data1: np.ndarray,
    data2: np.ndarray,
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """
    Wilcoxon signed-rank test (non-parametric paired test).

    More robust than t-test when normality assumption violated.

    Args:
        data1: Observations from method 1
        data2: Observations from method 2 (paired)
        alternative: "two-sided", "greater", or "less"

    Returns:
        Dict with statistic, p_value, effect_size
    """
    if len(data1) != len(data2):
        raise ValueError("Data arrays must have same length for paired test")

    data1 = np.asarray(data1, dtype=float)
    data2 = np.asarray(data2, dtype=float)
    differences = data1 - data2
    if np.allclose(differences, 0.0, rtol=0.0, atol=1e-12):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "n_pairs": len(data1),
            "effect_size_r": 0.0,
        }

    # Wilcoxon signed-rank test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = stats.wilcoxon(data1, data2, alternative=alternative, zero_method='wilcox')

    # Compute effect size using rank-biserial correlation
    # Scientific justification: proper effect size for Wilcoxon test
    # Formula: r = 1 - (2*W) / (n*(n+1)) where W is the smaller rank sum
    n = len(data1)
    # Wilcoxon result.statistic is W+ (sum of positive ranks)
    # Total sum of ranks = n*(n+1)/2, so W- = n*(n+1)/2 - W+
    w_plus = float(result.statistic)
    w_minus = n * (n + 1) / 2 - w_plus
    w_smaller = min(w_plus, w_minus)
    effect_size = 1.0 - (2.0 * w_smaller) / (n * (n + 1))
    # Handle potential numerical issues
    if not np.isfinite(effect_size):
        effect_size = 0.0

    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_pairs": n,
        "effect_size_r": float(effect_size),
    }


def cohens_d_paired(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Compute paired-samples Cohen's d (a.k.a. d_z) for paired data.

    Interpretation:
        - Small: d ≈ 0.2
        - Medium: d ≈ 0.5
        - Large: d ≈ 0.8

    Args:
        data1: Observations from method 1
        data2: Observations from method 2

    Scientific justification:
        SCBA comparisons are repeated-measures: each method is evaluated on the
        same (patient, perturbation) instances. The appropriate standardized
        effect size is therefore computed on paired differences:
            d_z = mean(data1 - data2) / std(data1 - data2, ddof=1)

    Returns:
        Paired Cohen's d (d_z)
    """
    if len(data1) != len(data2):
        raise ValueError("Data arrays must have same length for paired Cohen's d")
    differences = data1 - data2
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)

    if std_diff == 0:
        return 0.0

    return mean_diff / std_diff


def cohens_d(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Backward-compatible alias for `cohens_d_paired`.

    Scientific justification: historical code used `cohens_d` while computing
    the paired-samples variant (d_z); retaining the name avoids breaking
    downstream scripts while ensuring the definition is explicit in the paper.
    """
    return cohens_d_paired(data1, data2)


def _method_seed(method_name: str, base_seed: Optional[int]) -> Optional[int]:
    """
    Generate method-specific seed using platform-independent hash.

    Scientific justification: ensures reproducible bootstrap CIs across platforms.
    Uses hashlib.md5 instead of zlib.adler32 for cross-platform consistency.
    """
    if base_seed is None:
        return None
    # Use MD5 hash for platform-independent determinism
    digest = int(hashlib.md5(method_name.encode("utf-8")).hexdigest()[:8], 16)
    return int((int(base_seed) + digest) % (2 ** 32))


def compare_all_methods(
    method_results: Dict[str, np.ndarray],
    metric_name: str = "metric",
    alpha: float = 0.05,
    seed: Optional[int] = 42,
) -> Dict:
    """
    Comprehensive pairwise comparison of all methods.

    Args:
        method_results: Dict mapping method names to arrays of observations
        metric_name: Name of metric being compared (for reporting)
        alpha: Significance level for tests
        seed: RNG seed used for bootstrap CIs (method-specific offsets applied)

    Returns:
        Dict with summary table and pairwise comparisons
    """
    methods = list(method_results.keys())
    n_methods = len(methods)

    # Summary statistics for each method
    summary = {}
    for method, data in method_results.items():
        # Bootstrap 95% CI
        lower, mean, upper = bootstrap_confidence_interval(
            data, seed=_method_seed(method, seed)
        )

        summary[method] = {
            "n": len(data),
            "mean": float(mean),
            "std": float(np.std(data, ddof=1)),
            "median": float(np.median(data)),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "ci_width": float(upper - lower),
        }

    # Pairwise comparisons
    pairwise_comparisons = []

    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            method1, method2 = methods[i], methods[j]
            data1, data2 = method_results[method1], method_results[method2]

            # Paired t-test
            t_test = paired_t_test(data1, data2, alternative="two-sided")

            # Wilcoxon signed-rank test
            wilcoxon = wilcoxon_signed_rank_test(data1, data2, alternative="two-sided")

            # Effect size
            d = cohens_d_paired(data1, data2)

            # Determine if significantly different
            is_significant_parametric = t_test["p_value"] < alpha
            is_significant_nonparametric = wilcoxon["p_value"] < alpha

            pairwise_comparisons.append({
                "method_1": method1,
                "method_2": method2,
                "mean_diff": t_test["mean_difference"],
                "t_statistic": t_test["t_statistic"],
                "t_p_value": t_test["p_value"],
                "wilcoxon_statistic": wilcoxon["statistic"],
                "wilcoxon_p_value": wilcoxon["p_value"],
                "cohens_d": float(d),
                "significant_parametric": is_significant_parametric,
                "significant_nonparametric": is_significant_nonparametric,
            })

    # Bonferroni correction for multiple comparisons
    n_comparisons = len(pairwise_comparisons)
    bonferroni_alpha = alpha / n_comparisons

    for comparison in pairwise_comparisons:
        comparison["bonferroni_alpha"] = bonferroni_alpha
        comparison["significant_bonferroni_t"] = comparison["t_p_value"] < bonferroni_alpha
        comparison["significant_bonferroni_w"] = comparison["wilcoxon_p_value"] < bonferroni_alpha

    return {
        "metric_name": metric_name,
        "alpha": alpha,
        "bonferroni_alpha": bonferroni_alpha,
        "n_methods": n_methods,
        "n_comparisons": n_comparisons,
        "summary": summary,
        "pairwise_comparisons": pairwise_comparisons,
    }


def friedman_test(method_results: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Friedman test for multiple related samples (non-parametric ANOVA).

    Tests if there are any differences among multiple methods.

    Args:
        method_results: Dict mapping method names to arrays of observations

    Returns:
        Dict with test statistic and p-value
    """
    # Stack data into matrix (samples × methods)
    data_matrix = np.column_stack([method_results[m] for m in method_results.keys()])

    # Friedman test
    statistic, p_value = stats.friedmanchisquare(*data_matrix.T)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n_methods": len(method_results),
        "n_observations": len(next(iter(method_results.values()))),
    }


def format_ci(lower: float, upper: float, decimals: int = 4) -> str:
    """Format confidence interval for printing."""
    return f"[{lower:.{decimals}f}, {upper:.{decimals}f}]"


def format_p_value(p: float) -> str:
    """Format p-value for printing with appropriate precision."""
    if not np.isfinite(p):
        return "p = n/a"
    if p < 0.001:
        return "p < 0.001 ***"
    elif p < 0.01:
        return f"p = {p:.3f} **"
    elif p < 0.05:
        return f"p = {p:.3f} *"
    else:
        return f"p = {p:.3f}"


def generate_comparison_table(comparison_results: Dict) -> str:
    """
    Generate formatted comparison table for publication.

    Args:
        comparison_results: Output from compare_all_methods()

    Returns:
        Formatted string table
    """
    summary = comparison_results["summary"]
    metric = comparison_results["metric_name"]

    lines = []
    lines.append("=" * 100)
    lines.append(f"Statistical Comparison: {metric}")
    lines.append("=" * 100)
    lines.append(f"{'Method':<25} {'Mean±SD':<20} {'Median':<12} {'95% CI':<25} {'N':<5}")
    lines.append("-" * 100)

    for method, stats in summary.items():
        mean_std = f"{stats['mean']:.4f}±{stats['std']:.4f}"
        median = f"{stats['median']:.4f}"
        ci = format_ci(stats['ci_lower'], stats['ci_upper'])
        n = stats['n']
        lines.append(f"{method:<25} {mean_std:<20} {median:<12} {ci:<25} {n:<5}")

    lines.append("=" * 100)

    # Pairwise comparisons
    lines.append("\nPairwise Comparisons (with Bonferroni correction):")
    lines.append("-" * 100)
    lines.append(f"{'Comparison':<40} {'Mean Diff':<12} {'t-test':<20} {'Wilcoxon':<20} {'Cohen d':<10}")
    lines.append("-" * 100)

    for comp in comparison_results["pairwise_comparisons"]:
        comparison_name = f"{comp['method_1']} vs {comp['method_2']}"
        mean_diff = f"{comp['mean_diff']:.4f}"
        t_test_str = format_p_value(comp['t_p_value'])
        wilcoxon_str = format_p_value(comp['wilcoxon_p_value'])
        cohens_d = f"{comp['cohens_d']:.3f}"

        sig_marker = ""
        if comp['significant_bonferroni_t'] and comp['significant_bonferroni_w']:
            sig_marker = " ✓✓"
        elif comp['significant_bonferroni_t'] or comp['significant_bonferroni_w']:
            sig_marker = " ✓"

        lines.append(f"{comparison_name:<40} {mean_diff:<12} {t_test_str:<20} {wilcoxon_str:<20} {cohens_d:<10}{sig_marker}")

    lines.append("=" * 100)
    lines.append("Significance: *** p<0.001, ** p<0.01, * p<0.05")
    lines.append(f"Bonferroni-corrected α = {comparison_results['bonferroni_alpha']:.4f}")
    lines.append("✓✓ = significant with both tests, ✓ = significant with one test")
    lines.append("=" * 100)

    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Simulate data for 3 methods
    method_a = np.random.normal(0.85, 0.05, 30)
    method_b = np.random.normal(0.83, 0.05, 30)
    method_c = np.random.normal(0.80, 0.06, 30)

    method_results = {
        "Method A": method_a,
        "Method B": method_b,
        "Method C": method_c,
    }

    # Run comparison
    results = compare_all_methods(method_results, metric_name="Dice Coefficient")

    # Print table
    print(generate_comparison_table(results))

    # Friedman test
    friedman = friedman_test(method_results)
    print(f"\nFriedman Test: χ²={friedman['statistic']:.2f}, {format_p_value(friedman['p_value'])}")
