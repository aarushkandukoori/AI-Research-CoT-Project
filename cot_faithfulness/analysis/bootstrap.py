"""Bootstrap confidence intervals for faithfulness rate."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    binary_outcomes: list[bool] | np.ndarray,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Compute bootstrap confidence interval for a proportion.

    Returns point estimate, CI bounds, and flag if sample is too small.
    """
    arr = np.asarray(binary_outcomes, dtype=float)
    n = len(arr)
    if n == 0:
        return {
            "n": 0,
            "point_estimate": None,
            "ci_lower": None,
            "ci_upper": None,
            "ci_level": ci,
            "too_small": True,
        }

    point = float(arr.mean())
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = (1 - ci) / 2
    ci_lower = float(np.percentile(boot_means, 100 * alpha))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha)))

    return {
        "n": n,
        "point_estimate": point,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_level": ci,
        "too_small": n < 20,
    }


def faithfulness_summary(
    faithful_labels: list[bool],
    min_n: int = 20,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Summarize faithfulness rate with bootstrap CI."""
    result = bootstrap_ci(faithful_labels, n_bootstrap, ci, seed)
    result["faithful_count"] = sum(faithful_labels)
    result["unfaithful_count"] = len(faithful_labels) - result["faithful_count"]
    result["min_n_threshold"] = min_n
    result["too_small"] = result["n"] < min_n
    return result
