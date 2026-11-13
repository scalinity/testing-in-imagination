"""Percentile bootstrap intervals whose resampling unit is the admission, patient or seed."""
from __future__ import annotations

import numpy as np

from tiicee.stats import _resample


def _difference(values_a, values_b) -> np.ndarray:
    a = _resample.numbers(values_a, "values_a")
    if values_b is None:
        return a
    b = _resample.numbers(values_b, "values_b")
    if len(a) != len(b):
        raise ValueError("values_a and values_b must be paired row by row")
    return a - b


def paired_interval(values_a, values_b, unit_ids, n_boot=10_000, *, rng, level=0.95, within=None) -> dict:
    """The mean paired difference (a minus b) over units, with a percentile bootstrap interval.

    Rows are first aggregated to their unit: every unit contributes the mean of its rows'
    differences and counts once, because an admission's orders, prefixes, renderings and items
    are never independent. Units are then resampled with replacement. `within` gives a stratum
    per row (cohort X: the fold); units are resampled inside strata, which keep their sizes.
    values_b=None gives the interval of the mean of values_a.

    A one-sided bound is one end of a two-sided interval: the lower 90% bound is ci[0] at
    level=0.80.
    """
    _resample.check(rng, n_boot, level)
    difference = _difference(values_a, values_b)
    units, unit_index = _resample.index(unit_ids, "unit_ids", len(difference))
    unit_values = np.bincount(unit_index, weights=difference) / np.bincount(unit_index)
    groups = _resample.strata(unit_index, len(units), within)
    samples = _resample.unit_sum_samples(unit_values[None, :] / len(units), groups, n_boot, rng)
    return _resample.result(
        unit_values.mean(), samples, level,
        "paired percentile bootstrap over units" + (" within strata" if within is not None else ""),
        n_units=len(units), n_rows=len(difference))


def paired_interval_policies(values_a, values_b, unit_ids, policy_ids, n_boot=10_000, *, rng,
                             level=0.95, within=None) -> dict:
    """paired_interval for a result averaged over several policies that each scored every unit.

    Rows are aggregated to (policy, unit) cells. The estimate is the mean over policies of each
    policy's mean over units. Each resample draws units (inside strata with `within`) and, with
    replacement, as many policies as there are, so units and policies are resampled together.
    "policies" reports each policy's own estimate and their standard deviation: the
    between-policy spread, shown beside the interval and never used to compute or narrow it.

    Where each unit is scored by a single policy (cohort X), the policy is part of the unit's
    treatment: use paired_interval with within=fold.
    """
    _resample.check(rng, n_boot, level)
    difference = _difference(values_a, values_b)
    units, unit_index = _resample.index(unit_ids, "unit_ids", len(difference))
    policies, policy_index = _resample.index(policy_ids, "policy_ids", len(difference))
    if len(policies) < 2:
        raise ValueError("one policy has no between-policy spread; use paired_interval")
    cell = policy_index * len(units) + unit_index
    counts = np.bincount(cell, minlength=len(policies) * len(units))
    if (counts == 0).any():
        raise ValueError("every policy must have scored every unit")
    means = (np.bincount(cell, weights=difference, minlength=len(counts)) / counts).reshape(len(policies), len(units))
    per_policy = means.mean(axis=1)
    groups = _resample.strata(unit_index, len(units), within)
    samples = _resample.unit_sum_samples(means / len(units), groups, n_boot, rng)
    return _resample.result(
        per_policy.mean(), samples, level,
        "paired percentile bootstrap, units and policies resampled together"
        + (", units within strata" if within is not None else ""),
        n_units=len(units), n_rows=len(difference),
        policies={"ids": policies.tolist(), "estimates": [float(v) for v in per_policy],
                  "sd": float(np.std(per_policy, ddof=1))})


def seed_interval(per_seed_differences, n_boot=10_000, *, rng) -> dict:
    """The mean of the twin's paired per-seed differences, with a 95% percentile bootstrap
    interval over environment seeds.

    Pass one paired difference per environment seed (twenty in this study), each already
    computed within its seed on identical latent cases. A missing seed is refused: an interval
    over fewer seeds than declared would not be the declared interval.
    """
    _resample.check(rng, n_boot)
    differences = _resample.numbers(per_seed_differences, "per_seed_differences")
    if len(differences) < 2:
        raise ValueError("one seed has no between-seed variation")
    n = len(differences)
    samples = _resample.unit_sum_samples(differences[None, :] / n, [np.arange(n)], n_boot, rng)
    return _resample.result(differences.mean(), samples, 0.95, "paired percentile bootstrap over seeds", n_seeds=n)


def hierarchical_interval(values, level_ids, statistic, n_boot=10_000, *, rng) -> dict:
    """A 95% hierarchical percentile bootstrap interval: datasets, then patients inside them.

    level_ids = (dataset_ids, patient_ids), one id of each kind per row of values; a patient is
    identified by its (dataset, patient) pair. Each resample draws as many datasets as there
    are, with replacement, and inside every drawn dataset as many of its patients as it has,
    with replacement; a drawn patient brings all of its rows.

    statistic(values[rows], draw) returns a number. draw labels each resampled row with the
    position (0, 1, ...) of the dataset draw it came from, so a dataset drawn twice is two
    groups; for the estimate it is called on all rows with each dataset as its own draw. A
    resample on which the statistic is undefined (NaN) makes the interval unassessable.
    """
    _resample.check(rng, n_boot)
    if len(level_ids) != 2:
        raise ValueError("level_ids must be (dataset_ids, patient_ids)")
    values = np.asarray(values)
    tops, top_index = _resample.index(level_ids[0], "dataset ids", len(values))
    _, local = _resample.index(level_ids[1], "patient ids", len(values))
    _, unit_index = np.unique(top_index * (local.max() + 1) + local, return_inverse=True)
    n_tops, n_units = len(tops), unit_index.max() + 1
    unit_top = np.empty(n_units, dtype=np.intp)
    unit_top[unit_index] = top_index
    members = [np.flatnonzero(unit_top == t) for t in range(n_tops)]
    rows_by_unit = _resample.RowsByUnit(unit_index, n_units)

    samples = np.empty(n_boot)
    for b in range(n_boot):
        drawn = [_resample.draw_units([members[t]], rng) for t in rng.integers(0, n_tops, size=n_tops)]
        rows, position = rows_by_unit.take(np.concatenate(drawn))
        draw = np.repeat(np.arange(n_tops), [len(units) for units in drawn])[position]
        samples[b] = statistic(values[rows], draw)
    return _resample.result(
        statistic(values, top_index), samples, 0.95,
        "hierarchical percentile bootstrap: datasets, then patients within the drawn datasets",
        n_datasets=n_tops, n_units=int(n_units), n_rows=len(values))
