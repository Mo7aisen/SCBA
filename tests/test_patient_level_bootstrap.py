import numpy as np

from scba.metrics.statistical_tests import (
    bootstrap_confidence_interval,
    cluster_bootstrap_confidence_interval,
)


def test_cluster_bootstrap_resamples_patients_not_pseudoreplicates() -> None:
    # Scientific justification (M1): repeated CFs per patient are correlated.
    # A naive bootstrap over CF rows underestimates uncertainty; a cluster
    # bootstrap over patients should yield a wider CI in this correlated setting.
    rng = np.random.default_rng(42)
    patient_effects = rng.normal(loc=0.0, scale=1.0, size=10)

    values = np.repeat(patient_effects, repeats=3).astype(float)
    clusters = np.repeat(np.arange(patient_effects.size), repeats=3)

    lo_naive, mean_naive, hi_naive = bootstrap_confidence_interval(
        values, n_iterations=3000, seed=42
    )
    lo_cluster, mean_cluster, hi_cluster = cluster_bootstrap_confidence_interval(
        values, clusters, n_iterations=3000, seed=42
    )

    assert np.isclose(mean_naive, mean_cluster)
    assert (hi_cluster - lo_cluster) > (hi_naive - lo_naive)

