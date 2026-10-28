"""
Paired uncertainty and supporting significance tests (spec §8.2).

Every comparison is paired: both retrievers are scored on the same queries, and each
bootstrap resample uses the same query indices for both. Deltas are candidate − baseline
(B1 semantic − Stage A).
"""
from typing import Dict, Optional, Sequence, Union

import numpy as np
from scipy import stats as scipy_stats

import config


def paired_bootstrap(
    baseline: Sequence[float],
    candidate: Sequence[float],
    n_resamples: int = config.BOOTSTRAP_RESAMPLES,
    seed: int = config.SEED,
    ci: Sequence[float] = config.BOOTSTRAP_CI,
) -> Optional[Dict[str, float]]:
    """
    Paired percentile bootstrap of mean(candidate) − mean(baseline).

    Resamples query indices with replacement (default_rng(seed)); the CI is the
    (2.5, 97.5) percentiles of the resampled deltas. Returns None for n == 0.
    """
    a = np.asarray(baseline, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Paired arrays must have the same shape")
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("Filter to the metric's denominator before bootstrapping")
    n = len(a)
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    deltas = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    low, high = np.percentile(deltas, ci)
    return {
        "n": int(n),
        "baseline": float(a.mean()),
        "candidate": float(b.mean()),
        "delta": float(b.mean() - a.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "resamples": int(n_resamples),
        "seed": int(seed),
        "method": "paired percentile bootstrap",
    }


def paired_wilcoxon(baseline: Sequence[float], candidate: Sequence[float]) -> Dict[str, Union[int, float, str]]:
    """
    Two-sided Wilcoxon signed-rank on per-query differences (candidate − baseline).

    zero_method="wilcox": zero-difference pairs are discarded before ranking; n_zero
    reports how many. If every difference is zero the test is not computed ("n/a").
    Supporting evidence only — never sufficient on its own (spec §13).
    """
    a = np.asarray(baseline, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Paired arrays must have the same shape")
    d = b - a
    n_zero = int(np.sum(d == 0))
    n_nonzero = int(len(d) - n_zero)
    result: Dict[str, Union[int, float, str]] = {
        "n_total": int(len(d)),
        "n_zero": n_zero,
        "n_nonzero": n_nonzero,
        "alternative": config.WILCOXON["alternative"],
        "zero_method": config.WILCOXON["zero_method"],
    }
    if n_nonzero == 0:
        result.update({"statistic": "n/a", "p_value": "n/a"})
        return result
    res = scipy_stats.wilcoxon(
        b, a, alternative=config.WILCOXON["alternative"], zero_method=config.WILCOXON["zero_method"]
    )
    result.update({"statistic": float(res.statistic), "p_value": float(res.pvalue)})
    return result
