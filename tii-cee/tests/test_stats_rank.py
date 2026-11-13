"""Spearman correlation, the interval of a difference of two correlations (E1 part C), and the
rank correlation across cells under the hierarchical bootstrap (E7)."""
import math

import numpy as np
import pytest

from tiicee import stats


def reference_spearman(x, y):
    """Average ranks by counting, then Pearson's formula; no sorting shared with the module."""
    def ranks(values):
        return [sum(v < u for v in values) + (sum(v == u for v in values) + 1) / 2 for u in values]
    rx, ry = ranks(list(x)), ranks(list(y))
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
    sxx, syy = sum((p - mx) ** 2 for p in rx), sum((q - my) ** 2 for q in ry)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


# ---- spearman ---------------------------------------------------------------------------------

def test_spearman_by_hand():
    # No ties: rank differences (-1, 1, -1, 1, 0), so rho = 1 - 6 * 4 / (5 * 24) = 0.8.
    assert stats.spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]) == pytest.approx(0.8, abs=1e-12)
    # Ties: y ranks (1.5, 1.5, 3, 4), centered (-1, -1, 0.5, 1.5); x centered (-1.5, -0.5, 0.5, 1.5).
    # Cross products sum to 4.5, squares to 5 and 4.5: 4.5 / sqrt(22.5) = 3 / sqrt(10).
    assert stats.spearman([1, 2, 3, 4], [1, 1, 2, 3]) == pytest.approx(3 / math.sqrt(10), abs=1e-12)
    assert stats.spearman([3, 1, 2], [30, 10, 20]) == pytest.approx(1.0, abs=1e-12)
    assert stats.spearman([3, 1, 2], [-3, -1, -2]) == pytest.approx(-1.0, abs=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_spearman_matches_a_counting_reference_with_ties(seed):
    rng = np.random.default_rng(seed)
    x, y = rng.integers(0, 6, size=40).astype(float), rng.normal(size=40).round(1)
    assert stats.spearman(x, y) == pytest.approx(reference_spearman(x, y), abs=1e-12)
    assert stats.spearman(np.exp(x), y ** 3) == pytest.approx(stats.spearman(x, y), abs=1e-12)  # monotone maps


def test_spearman_is_undefined_for_a_constant_and_refuses_bad_input():
    assert math.isnan(stats.spearman([1, 2, 3], [5, 5, 5]))
    with pytest.raises(ValueError, match="paired"):
        stats.spearman([1, 2, 3], [1, 2])
    with pytest.raises(ValueError, match="finite"):
        stats.spearman([1, 2, float("nan")], [1, 2, 3])


# ---- difference of two correlations on the same units ----------------------------------------

def histories(seed, n_units=40, rows_per_unit=3):
    """A target that tracks x_a closely and x_b loosely, a few rows per unit."""
    rng = np.random.default_rng(seed)
    unit = np.repeat(np.arange(n_units), rows_per_unit)
    x_a = rng.normal(size=len(unit))
    target = x_a + rng.normal(0, 0.5, size=len(unit))
    x_b = target + rng.normal(0, 3.0, size=len(unit))
    return target, x_a, x_b, unit


def test_difference_estimate_and_its_two_correlations():
    target, x_a, x_b, unit = histories(1)
    result = stats.spearman_difference_interval(target, x_a, x_b, unit, n_boot=300, rng=np.random.default_rng(0))
    assert result["rho_a"] == pytest.approx(reference_spearman(target, x_a), abs=1e-12)
    assert result["rho_b"] == pytest.approx(reference_spearman(target, x_b), abs=1e-12)
    assert result["estimate"] == pytest.approx(result["rho_a"] - result["rho_b"], abs=1e-12)
    assert result["ci"][0] > 0.2 and result["ci"][0] < result["estimate"] < result["ci"][1]
    assert result["n_units"] == 40 and result["n_rows"] == 120 and result["status"] == "assessed"
    assert result == stats.spearman_difference_interval(target, x_a, x_b, unit, n_boot=300, rng=np.random.default_rng(0))


def test_difference_restated_independently():
    target, x_a, x_b, unit = histories(2, n_units=12, rows_per_unit=2)
    seed_of_unit = np.arange(12) % 2  # patients within seeds
    rows_of = [np.flatnonzero(unit == u) for u in range(12)]
    members = [np.flatnonzero(seed_of_unit == s) for s in (0, 1)]
    rng, differences = np.random.default_rng(23), []
    for _ in range(120):
        drawn = np.concatenate([m[rng.integers(0, len(m), size=len(m))] for m in members])
        rows = np.concatenate([rows_of[u] for u in drawn])
        differences.append(reference_spearman(target[rows], x_a[rows]) - reference_spearman(target[rows], x_b[rows]))
    result = stats.spearman_difference_interval(target, x_a, x_b, unit, n_boot=120, rng=np.random.default_rng(23),
                                                within=seed_of_unit[unit])
    assert result["ci"] == pytest.approx(list(np.quantile(differences, [0.025, 0.975])), abs=1e-9)


def test_the_two_correlations_share_every_resample():
    target, x_a, _, unit = histories(3)
    same = stats.spearman_difference_interval(target, x_a, x_a.copy(), unit, n_boot=200, rng=np.random.default_rng(0))
    assert same["estimate"] == 0.0 and same["ci"] == [0.0, 0.0]


def test_units_are_resampled_not_rows():
    rng = np.random.default_rng(4)
    unit = np.repeat(np.arange(15), 20)  # twenty identical rows per unit
    x_a, x_b = rng.normal(size=15), rng.normal(size=15)
    target = x_a + rng.normal(0, 1.0, size=15)
    by_unit = stats.spearman_difference_interval(target[unit], x_a[unit], x_b[unit], unit, n_boot=600, rng=np.random.default_rng(1))
    by_row = stats.spearman_difference_interval(target[unit], x_a[unit], x_b[unit], np.arange(300), n_boot=600, rng=np.random.default_rng(1))
    assert by_unit["estimate"] == pytest.approx(by_row["estimate"], abs=1e-12)
    assert by_unit["ci"][1] - by_unit["ci"][0] > 3 * (by_row["ci"][1] - by_row["ci"][0])


def test_difference_refusals():
    target, x_a, x_b, unit = histories(5, n_units=6, rows_per_unit=2)
    with pytest.raises(ValueError, match="one stratum"):
        stats.spearman_difference_interval(target, x_a, x_b, unit, rng=np.random.default_rng(0), within=np.arange(12) % 2)
    with pytest.raises(ValueError, match="paired"):
        stats.spearman_difference_interval(target, x_a, x_b[:-1], unit, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="level"):
        stats.spearman_difference_interval(target, x_a, x_b, unit, rng=np.random.default_rng(0), level=1.0)


# ---- E7: rank correlation across cells, hierarchical bootstrap --------------------------------

def cell_rows(seed, patients=(7, 5, 9), cells=("r1", "r2", "r3", "r4")):
    """One row per patient and cell in each dataset; realized gain follows predicted gain loosely."""
    rng = np.random.default_rng(seed)
    rows = [(d, p, c) for d, n in enumerate(patients) for p in range(n) for c in cells]
    dataset, patient, cell = (np.array(column) for column in zip(*rows))
    level = rng.normal(size=(len(patients), len(cells)))
    predicted = level[dataset, np.searchsorted(cells, cell)] + rng.normal(0, 0.3, size=len(rows))
    realized = predicted + rng.normal(0, 0.6, size=len(rows))
    return predicted, realized, cell, dataset, patient


def test_cell_correlation_estimate():
    predicted, realized, cell, dataset, patient = cell_rows(1)
    result = stats.cell_spearman_interval(predicted, realized, cell, dataset, patient, n_boot=200, rng=np.random.default_rng(0))
    keys = sorted(set(zip(dataset.tolist(), cell.tolist())))
    mean = lambda v, k: float(np.mean(v[(dataset == k[0]) & (cell == k[1])]))
    expected = reference_spearman([mean(predicted, k) for k in keys], [mean(realized, k) for k in keys])
    assert result["estimate"] == pytest.approx(expected, abs=1e-12)
    assert result["n_cells"] == 12 and result["n_datasets"] == 3 and result["n_units"] == 21
    assert -1 <= result["ci"][0] <= result["ci"][1] <= 1 and result["level"] == 0.95
    assert result == stats.cell_spearman_interval(predicted, realized, cell, dataset, patient, n_boot=200, rng=np.random.default_rng(0))


def test_cell_correlation_restated_independently():
    predicted, realized, cell, dataset, patient = cell_rows(2, patients=(4, 6), cells=("r1", "r2", "r3"))
    names = ["r1", "r2", "r3"]
    rng, correlations = np.random.default_rng(29), []
    for _ in range(150):
        p_means, r_means = [], []
        for d in rng.integers(0, 2, size=2):
            n = (4, 6)[d]
            drawn = rng.integers(0, n, size=n)
            for name in names:  # the drawn dataset's cells, from this draw's patients
                rows = [np.flatnonzero((dataset == d) & (patient == p) & (cell == name))[0] for p in drawn]
                p_means.append(predicted[rows].mean())
                r_means.append(realized[rows].mean())
        correlations.append(reference_spearman(p_means, r_means))
    result = stats.cell_spearman_interval(predicted, realized, cell, dataset, patient, n_boot=150, rng=np.random.default_rng(29))
    assert result["ci"] == pytest.approx(list(np.quantile(correlations, [0.025, 0.975])), abs=1e-9)


def test_a_patients_rows_count_once_in_a_cell():
    predicted, realized, cell, dataset, patient = cell_rows(3)
    base = stats.cell_spearman_interval(predicted, realized, cell, dataset, patient, n_boot=150, rng=np.random.default_rng(5))
    repeat = np.r_[np.arange(len(cell)), np.zeros(30, dtype=int)]  # one patient's row in one cell, thirty more times
    again = stats.cell_spearman_interval(predicted[repeat], realized[repeat], cell[repeat], dataset[repeat],
                                         patient[repeat], n_boot=150, rng=np.random.default_rng(5))
    assert again["estimate"] == pytest.approx(base["estimate"], abs=1e-12)
    assert again["ci"] == pytest.approx(base["ci"], abs=1e-12)


def test_a_perfect_relation_stays_perfect_on_every_resample():
    predicted, _, cell, dataset, patient = cell_rows(4)
    predicted = np.round(10 * predicted)  # whole numbers: every cell mean is computed without rounding error
    result = stats.cell_spearman_interval(predicted, 2 * predicted + 1, cell, dataset, patient, n_boot=300,
                                          rng=np.random.default_rng(0))
    assert result["estimate"] == pytest.approx(1.0, abs=1e-12) and result["ci"] == pytest.approx([1.0, 1.0], abs=1e-12)
