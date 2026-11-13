"""CEE-Gate and its bootstrap bound (SPEC 5.8)."""
from __future__ import annotations

import numpy as np

N_BOOT = 10_000     # SPEC 15: percentile bootstrap with 10,000 resamples
_CHUNK = 256        # resamples drawn at a time, to bound memory


def gate_allows(kl_inc, eps_prior, eps_wm):
    """CEE-Gate (SPEC 5.8): apply the correction only if KL_inc > eps_prior + eps_wm.

    The two bounds are upper 90% bootstrap bounds, by test type, of the absolute prior and
    world-model terms of SPEC 5.4, estimated on calibration folds that exclude the scored case.
    """
    return bool(kl_inc > eps_prior + eps_wm)


def bootstrap_upper_bound(values, groups, rng, level=0.90, n_boot=N_BOOT, absolute=False):
    """One-sided upper percentile-bootstrap bound of a mean, resampling groups (SPEC 5.8, 15).

    values holds one number per row (a decision point); groups names the admission or patient each
    row belongs to. The resampling unit is the group, never the row: each resample draws as many
    groups as there are, with replacement, and takes the mean over all their rows. With absolute
    True the bound is on the absolute value of that mean, which is what the gate thresholds need.
    rng is a numpy Generator supplied by the caller, so the seed is recorded by purpose.
    """
    values, groups = np.asarray(values, dtype=float), np.asarray(groups)
    if values.ndim != 1 or groups.shape != values.shape or values.size == 0:
        raise ValueError("values and groups must be non-empty and one entry per row")
    _, index = np.unique(groups, return_inverse=True)
    sums, rows = np.bincount(index, weights=values), np.bincount(index)
    means = np.empty(n_boot)
    for start in range(0, n_boot, _CHUNK):
        draws = rng.integers(0, sums.size, size=(min(_CHUNK, n_boot - start), sums.size))
        means[start : start + draws.shape[0]] = sums[draws].sum(1) / rows[draws].sum(1)
    if absolute:
        means = np.abs(means)
    return float(np.quantile(means, level))
