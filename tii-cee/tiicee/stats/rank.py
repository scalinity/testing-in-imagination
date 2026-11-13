"""Spearman rank correlation and its bootstrap intervals (E1 part C, E7)."""
from __future__ import annotations

import numpy as np

from tiicee.stats import _resample
from tiicee.stats.intervals import hierarchical_interval


def _ranks(x) -> np.ndarray:
    """Ranks from 1, tied values sharing the average of the ranks they span."""
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inverse]


def _centered_ranks(x) -> np.ndarray:
    ranks = _ranks(x)
    return ranks - ranks.mean()


def _correlation(rx, ry) -> float:
    scale = np.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / scale) if scale > 0 else float("nan")


def _spearman(x, y) -> float:
    return _correlation(_centered_ranks(x), _centered_ranks(y))


def spearman(x, y) -> float:
    """Spearman's rho: the Pearson correlation of average ranks. NaN when either input is constant."""
    x, y = _resample.numbers(x, "x"), _resample.numbers(y, "y")
    if len(x) != len(y):
        raise ValueError("x and y must be paired")
    return _spearman(x, y)


def spearman_difference_interval(target, x_a, x_b, unit_ids, n_boot=10_000, *, rng, level=0.95,
                                 within=None) -> dict:
    """rho(target, x_a) - rho(target, x_b) on the same rows, with a percentile bootstrap interval.

    The correlations are across rows (E1 part C: histories). Rows of one unit are never
    independent, so units are resampled with replacement and a drawn unit brings all of its
    rows; `within` resamples units inside strata (patients within seeds). Both correlations are
    recomputed on every resample from the same draw, which keeps the difference paired.
    """
    _resample.check(rng, n_boot, level)
    target, x_a, x_b = (_resample.numbers(v, name) for v, name in ((target, "target"), (x_a, "x_a"), (x_b, "x_b")))
    if not len(target) == len(x_a) == len(x_b):
        raise ValueError("target, x_a and x_b must be paired row by row")
    units, unit_index = _resample.index(unit_ids, "unit_ids", len(target))
    groups = _resample.strata(unit_index, len(units), within)
    rows_by_unit = _resample.RowsByUnit(unit_index, len(units))

    def difference(rows):
        ranks = _centered_ranks(target[rows])
        return _correlation(ranks, _centered_ranks(x_a[rows])) - _correlation(ranks, _centered_ranks(x_b[rows]))

    samples = np.empty(n_boot)
    for b in range(n_boot):
        samples[b] = difference(rows_by_unit.take(_resample.draw_units(groups, rng))[0])
    return _resample.result(
        difference(slice(None)), samples, level,
        "percentile bootstrap of a difference of Spearman correlations, units resampled with all their rows"
        + (" within strata" if within is not None else ""),
        rho_a=_spearman(target, x_a), rho_b=_spearman(target, x_b), n_units=len(units), n_rows=len(target))


def cell_spearman_interval(predicted, realized, cell_ids, dataset_ids, patient_ids, n_boot=10_000, *, rng) -> dict:
    """Spearman correlation of predicted with realized gain across cells, with a 95% interval
    under the hierarchical bootstrap (datasets, then patients).

    A cell is one cell_id (E7: a reader and pairing) in one dataset. Rows are first aggregated
    to their patient within the cell, so a patient counts once however many steps and tests it
    contributes; a cell's value is the mean over its patients. On a resample a dataset drawn
    twice gives two sets of cells, each from its own draw of patients.
    """
    predicted, realized = _resample.numbers(predicted, "predicted"), _resample.numbers(realized, "realized")
    if len(predicted) != len(realized):
        raise ValueError("predicted and realized must be paired row by row")
    n = len(predicted)
    cells, cell = _resample.index(cell_ids, "cell_ids", n)
    _, dataset = _resample.index(dataset_ids, "dataset_ids", n)
    _, patient = _resample.index(patient_ids, "patient_ids", n)
    n_cells, n_patients = len(cells), patient.max() + 1
    keys, group = np.unique((dataset * n_patients + patient) * n_cells + cell, return_inverse=True)
    size = np.bincount(group)
    predicted, realized = np.bincount(group, weights=predicted) / size, np.bincount(group, weights=realized) / size
    cell, patient, dataset = keys % n_cells, keys // n_cells % n_patients, keys // n_cells // n_patients

    def across_cells(rows, draw):
        _, which = np.unique(draw * n_cells + cell[rows], return_inverse=True)
        count = np.bincount(which)
        return _spearman(np.bincount(which, weights=predicted[rows]) / count,
                         np.bincount(which, weights=realized[rows]) / count)

    out = hierarchical_interval(np.arange(len(keys)), (dataset, patient), across_cells, n_boot, rng=rng)
    out["n_cells"] = int(len(np.unique(dataset * n_cells + cell)))
    return out
