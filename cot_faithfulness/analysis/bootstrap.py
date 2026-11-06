"""Bootstrap and Wilson confidence intervals for faithfulness rate."""

from __future__ import annotations

import math

import numpy as np


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval; retains coverage at the boundary (D5)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((center - margin) / denom, (center + margin) / denom)


def bootstrap_ci(
    binary_outcomes: list[bool] | np.ndarray,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Compute CI for a proportion.

    Uses percentile bootstrap in the interior; switches to Wilson when the
    cell is saturated (all 0 or all 1) so the interval does not collapse (D5).
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
            "method": None,
        }

    point = float(arr.mean())
    k = int(arr.sum())
    saturated = point in (0.0, 1.0)

    if saturated:
        lo, hi = wilson_ci(k, n)
        return {
            "n": n,
            "point_estimate": point,
            "ci_lower": lo,
            "ci_upper": hi,
            "ci_level": ci,
            "too_small": n < 20,
            "method": "wilson",
        }

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
        "method": "bootstrap",
    }


def faithfulness_summary(
    faithful_labels: list[bool],
    min_n: int = 20,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Summarize faithfulness rate with CI (bootstrap or Wilson)."""
    result = bootstrap_ci(faithful_labels, n_bootstrap, ci, seed)
    result["faithful_count"] = sum(faithful_labels)
    result["unfaithful_count"] = len(faithful_labels) - result["faithful_count"]
    result["min_n_threshold"] = min_n
    result["too_small"] = result["n"] < min_n
    return result


def faithfulness_bracket(labels: list[str]) -> dict:
    """
    Primary F (AMBIG as unfaithful) plus F- (drop AMBIG) and F+ (AMBIG as faithful).
    """
    primary = [lab == "FAITHFUL" for lab in labels]
    excl = [lab for lab in labels if lab != "AMBIGUOUS"]
    f_minus = [lab == "FAITHFUL" for lab in excl]
    f_plus = [lab in ("FAITHFUL", "AMBIGUOUS") for lab in labels]
    return {
        "F": faithfulness_summary(primary),
        "F_minus": faithfulness_summary(f_minus),
        "F_plus": faithfulness_summary(f_plus),
        "n_ambiguous": sum(1 for lab in labels if lab == "AMBIGUOUS"),
    }
