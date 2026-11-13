"""Index bookkeeping, resampling and percentile intervals shared by every bootstrap in this package."""
from __future__ import annotations

import numpy as np


def check(rng, n_boot, level=0.95) -> None:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator; this package never reads global random state")
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or n_boot < 2:
        raise ValueError("n_boot (or n_sims) must be an integer >= 2")
    if not 0 < level < 1:
        raise ValueError("level must lie strictly between 0 and 1")


def numbers(values, name) -> np.ndarray:
    """A one-dimensional float array. A missing or non-finite value is refused, never dropped."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty one-dimensional sequence of finite numbers")
    return array


def index(ids, name, size):
    """(distinct ids in sorted order, each row's position among them)."""
    ids = np.asarray(ids)
    if ids.ndim != 1 or len(ids) != size:
        raise ValueError(f"{name} must hold one id per row")
    return np.unique(ids, return_inverse=True)


def strata(unit_index, n_units, within) -> list:
    """Unit positions grouped by stratum, strata in sorted order; one group when `within` is None."""
    if within is None:
        return [np.arange(n_units)]
    labels, row_stratum = index(within, "within", len(unit_index))
    unit_stratum = np.empty(n_units, dtype=np.intp)
    unit_stratum[unit_index] = row_stratum
    if (unit_stratum[unit_index] != row_stratum).any():
        raise ValueError("a unit may lie in one stratum only; make unit ids distinct across strata")
    return [np.flatnonzero(unit_stratum == s) for s in range(len(labels))]


def draw_units(groups, rng) -> np.ndarray:
    """One resample: each stratum keeps its size and its units are drawn with replacement inside it.

    Strata are drawn in the order given, one `rng.integers` call each. For the four classes in
    class order this consumes the generator exactly as replication.stats.paired_bootstrap does.
    """
    return np.concatenate([group[rng.integers(0, len(group), size=len(group))] for group in groups])


def unit_sum_samples(x, groups, n_boot, rng) -> np.ndarray:
    """Bootstrap statistics for a quantity that is a sum over units of x[policy, unit].

    Each resample draws units within strata and then, when x has several policy rows, as many
    policies as there are, with replacement. Its statistic is the mean over the drawn policies of
    each policy's sum over the drawn units, so units and policies are resampled together.
    """
    n_policies = x.shape[0]
    samples = np.empty(n_boot)
    for b in range(n_boot):
        sums = x[:, draw_units(groups, rng)].sum(axis=1)
        if n_policies > 1:
            sums = sums[rng.integers(0, n_policies, size=n_policies)]
        samples[b] = sums.mean()
    return samples


class RowsByUnit:
    """The rows of each unit, so that a drawn unit brings every one of its rows."""

    def __init__(self, unit_index, n_units):
        self.order = np.argsort(unit_index, kind="stable")
        self.counts = np.bincount(unit_index, minlength=n_units)
        self.starts = np.cumsum(self.counts) - self.counts

    def take(self, units):
        """(rows of the drawn units, draw by draw; each row's position in `units`)."""
        counts = self.counts[units]
        position = np.repeat(np.arange(len(units)), counts)
        first = np.cumsum(counts) - counts
        rows = self.order[self.starts[units][position] + np.arange(len(position)) - first[position]]
        return rows, position


def percentile(samples, level) -> list:
    low = round((1 - level) / 2, 12)  # 0.025 exactly at level 0.95, the quantiles the project uses
    return [float(v) for v in np.quantile(samples, [low, 1 - low])]


def result(estimate, samples, level, method, **extra) -> dict:
    """A JSON-ready interval. An undefined resample makes it unassessable; draws are never dropped."""
    valid = np.isfinite(samples)
    assessed = bool(np.isfinite(estimate) and valid.all())
    return {"estimate": float(estimate) if np.isfinite(estimate) else None,
            "ci": percentile(samples, level) if assessed else None, "level": level,
            "n_boot": len(samples), "valid_resamples": int(valid.sum()),
            "status": "assessed" if assessed else "unassessable", "method": method, **extra}
