"""Unit, seed, several-policy and hierarchical bootstrap intervals.

Each resampling procedure is restated here in plain loops, with the generator consumed in the
documented order, and compared with the module; the unit rule is tested by its consequences.
"""
import math

import numpy as np
import pytest

from tiicee import stats


def clustered(seed, n_units, rows_per_unit):
    """Paired rows whose difference is mostly a unit effect, as an admission's prefixes are."""
    rng = np.random.default_rng(seed)
    unit = np.repeat(np.arange(n_units), rows_per_unit)
    effect = rng.normal(0.3, 1.0, size=n_units)[unit]
    b = rng.normal(size=len(unit))
    return b + effect + rng.normal(0, 0.05, size=len(unit)), b, unit


def width(result):
    return result["ci"][1] - result["ci"][0]


# ---- paired_interval --------------------------------------------------------------------------

def test_estimate_counts_every_unit_once():
    a = [3.0, 5.0, 4.0, 1.0, 1.0, 7.0]
    b = [2.0, 2.0, 0.0, 1.0, 1.0, 1.0]
    unit = ["u1", "u1", "u2", "u3", "u3", "u3"]
    # Unit means of a - b: (1 + 3) / 2 = 2, 4, (0 + 0 + 6) / 3 = 2. The row mean would be 14 / 6.
    result = stats.paired_interval(a, b, unit, n_boot=100, rng=np.random.default_rng(0))
    assert result["estimate"] == pytest.approx(8 / 3, abs=1e-12)
    assert result["n_units"] == 3 and result["n_rows"] == 6 and result["status"] == "assessed"


def test_interval_restated_independently():
    a, b, unit = clustered(1, 25, 4)
    values = np.array([np.mean((a - b)[unit == u]) for u in range(25)])
    rng = np.random.default_rng(42)
    means = [values[rng.integers(0, 25, size=25)].mean() for _ in range(500)]
    result = stats.paired_interval(a, b, unit, n_boot=500, rng=np.random.default_rng(42))
    assert result["estimate"] == pytest.approx(values.mean(), abs=1e-12)
    assert result["ci"] == pytest.approx(list(np.quantile(means, [0.025, 0.975])), abs=1e-12)
    inner = stats.paired_interval(a, b, unit, n_boot=500, rng=np.random.default_rng(42), level=0.80)
    assert inner["ci"] == pytest.approx(list(np.quantile(means, [0.10, 0.90])), abs=1e-12) and inner["level"] == 0.80
    assert result["ci"][0] < inner["ci"][0] < inner["ci"][1] < result["ci"][1]  # ci[0] at 0.80 is the lower 90% bound


def test_duplicating_an_admissions_rows_does_not_narrow_the_interval():
    rng = np.random.default_rng(2)
    a, b, unit = rng.normal(0.4, 1, size=30), rng.normal(size=30), np.arange(30)
    base = stats.paired_interval(a, b, unit, n_boot=800, rng=np.random.default_rng(7))
    repeat = np.r_[np.arange(30), np.zeros(49, dtype=int)]  # admission 0 now has fifty identical rows
    again = stats.paired_interval(a[repeat], b[repeat], unit[repeat], n_boot=800, rng=np.random.default_rng(7))
    assert again["estimate"] == pytest.approx(base["estimate"], abs=1e-12)
    assert again["ci"] == pytest.approx(base["ci"], abs=1e-12)
    assert again["n_units"] == 30 and again["n_rows"] == 79


def test_rows_of_one_unit_are_not_treated_as_independent():
    a, b, unit = clustered(3, 20, 25)
    by_unit = stats.paired_interval(a, b, unit, n_boot=1500, rng=np.random.default_rng(1))
    by_row = stats.paired_interval(a, b, np.arange(len(unit)), n_boot=1500, rng=np.random.default_rng(1))
    assert width(by_unit) > 3 * width(by_row)  # 25 near-copies per unit: about five times narrower if rows were units


def test_within_keeps_every_stratum_at_its_size():
    """Units of stratum A are worth 1 and those of B 0: resampling inside strata cannot move the mean."""
    values = [1.0] * 6 + [0.0] * 14
    stratum = ["A"] * 6 + ["B"] * 14
    inside = stats.paired_interval(values, None, range(20), n_boot=400, rng=np.random.default_rng(3), within=stratum)
    assert inside["estimate"] == pytest.approx(0.3, abs=1e-12)
    assert inside["ci"] == pytest.approx([0.3, 0.3], abs=1e-12) and "within strata" in inside["method"]
    pooled = stats.paired_interval(values, None, range(20), n_boot=400, rng=np.random.default_rng(3))
    assert width(pooled) > 0.2


def test_within_restated_independently():
    a, b, unit = clustered(4, 18, 3)
    fold = (unit % 3)  # six units per fold
    values = np.array([np.mean((a - b)[unit == u]) for u in range(18)])
    members = [np.flatnonzero(np.arange(18) % 3 == f) for f in range(3)]
    rng, means = np.random.default_rng(9), []
    for _ in range(300):
        means.append(np.concatenate([values[m[rng.integers(0, 6, size=6)]] for m in members]).mean())
    result = stats.paired_interval(a, b, unit, n_boot=300, rng=np.random.default_rng(9), within=fold)
    assert result["ci"] == pytest.approx(list(np.quantile(means, [0.025, 0.975])), abs=1e-12)


def test_single_arm_interval_and_refusals():
    values = np.random.default_rng(5).normal(1, 1, size=40)
    single = stats.paired_interval(values, None, np.arange(40), n_boot=300, rng=np.random.default_rng(1))
    paired = stats.paired_interval(values, np.zeros(40), np.arange(40), n_boot=300, rng=np.random.default_rng(1))
    assert single == paired and single["ci"][0] < single["estimate"] < single["ci"][1]
    with pytest.raises(ValueError, match="one stratum"):
        stats.paired_interval([1.0, 2.0, 3.0], None, ["u", "u", "v"], rng=np.random.default_rng(0), within=["f1", "f2", "f1"])
    with pytest.raises(ValueError, match="finite"):
        stats.paired_interval([1.0, float("nan")], None, [0, 1], rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="paired"):
        stats.paired_interval([1.0, 2.0], [1.0], [0, 1], rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="one id per row"):
        stats.paired_interval([1.0, 2.0], None, [0], rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="level"):
        stats.paired_interval([1.0, 2.0], None, [0, 1], rng=np.random.default_rng(0), level=95)
    with pytest.raises(TypeError, match="Generator"):
        stats.paired_interval([1.0, 2.0], None, [0, 1], rng=np.random.RandomState(0))


# ---- several policies -------------------------------------------------------------------------

def test_policies_estimate_and_spread_by_hand():
    #            policy p: units 1..3      policy q: units 1..3 (unit 3 has two rows)
    a = np.array([1.0, 2.0, 6.0, 0.0, 1.0, 1.0, 3.0])
    unit = ["u1", "u2", "u3", "u1", "u2", "u3", "u3"]
    policy = ["p", "p", "p", "q", "q", "q", "q"]
    result = stats.paired_interval_policies(a, None, unit, policy, n_boot=200, rng=np.random.default_rng(0))
    # Policy means: p = 3, q = (0 + 1 + 2) / 3 = 1. Their mean is 2 and their standard deviation sqrt(2).
    assert result["policies"] == {"ids": ["p", "q"], "estimates": pytest.approx([3.0, 1.0], abs=1e-12),
                                  "sd": pytest.approx(math.sqrt(2), abs=1e-12)}
    assert result["estimate"] == pytest.approx(2.0, abs=1e-12) and result["n_units"] == 3


def test_units_and_policies_are_resampled_together():
    """Units add no spread; the policies are worth 0 and 1. A resample is the mean of two policy
    draws: 0, 1/2 or 1 with chances 1/4, 1/2, 1/4, so the 95% percentile interval is [0, 1]."""
    values = [0.0] * 5 + [1.0] * 5
    result = stats.paired_interval_policies(values, None, list(range(5)) * 2, [0] * 5 + [1] * 5, n_boot=2000,
                                            rng=np.random.default_rng(1))
    assert result["estimate"] == pytest.approx(0.5, abs=1e-12)
    assert result["ci"] == pytest.approx([0.0, 1.0], abs=1e-12)
    assert "policies resampled together" in result["method"]


def test_policies_restated_independently():
    rng = np.random.default_rng(6)
    cells = rng.normal(size=(4, 12)) + rng.normal(size=(4, 1))  # policies x units
    unit, policy = np.tile(np.arange(12), 4), np.repeat(np.arange(4), 12)
    rng, means = np.random.default_rng(13), []
    for _ in range(400):
        columns = rng.integers(0, 12, size=12)
        rows = rng.integers(0, 4, size=4)
        means.append(cells[:, columns].mean(axis=1)[rows].mean())
    result = stats.paired_interval_policies(cells.ravel(), None, unit, policy, n_boot=400, rng=np.random.default_rng(13))
    assert result["ci"] == pytest.approx(list(np.quantile(means, [0.025, 0.975])), abs=1e-12)


def test_between_policy_spread_never_narrows_the_interval():
    """With policies that differ, the interval is wider than resampling units alone would give;
    the spread sits beside it and the interval does not depend on how it is summarized."""
    rng = np.random.default_rng(7)
    cells = rng.normal(0, 0.2, size=(5, 60)) + np.array([[-1.0], [-0.5], [0.0], [0.5], [1.0]])
    unit, policy = np.tile(np.arange(60), 5), np.repeat(np.arange(5), 60)
    together = stats.paired_interval_policies(cells.ravel(), None, unit, policy, n_boot=2000, rng=np.random.default_rng(2))
    units_only = stats.paired_interval(cells.mean(axis=0), None, np.arange(60), n_boot=2000, rng=np.random.default_rng(2))
    assert together["estimate"] == pytest.approx(units_only["estimate"], abs=1e-12)
    assert width(together) > 5 * width(units_only)
    assert together["policies"]["sd"] == pytest.approx(np.std(cells.mean(axis=1), ddof=1), abs=1e-12)


def test_policies_refusals_and_cell_aggregation():
    with pytest.raises(ValueError, match="every unit"):
        stats.paired_interval_policies([1.0, 2.0, 3.0], None, [0, 1, 0], [0, 0, 1], rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="use paired_interval"):
        stats.paired_interval_policies([1.0, 2.0], None, [0, 1], [0, 0], rng=np.random.default_rng(0))
    values, unit, policy = np.array([1.0, 4.0, 2.0, 8.0]), np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])
    base = stats.paired_interval_policies(values, None, unit, policy, n_boot=100, rng=np.random.default_rng(3))
    repeat = [0, 0, 0, 1, 2, 3]  # one cell's row three times over
    again = stats.paired_interval_policies(values[repeat], None, unit[repeat], policy[repeat], n_boot=100,
                                           rng=np.random.default_rng(3))
    assert again["ci"] == pytest.approx(base["ci"], abs=1e-12) and again["estimate"] == pytest.approx(base["estimate"], abs=1e-12)


# ---- seed_interval ----------------------------------------------------------------------------

def test_seed_interval_restated_independently():
    differences = np.random.default_rng(8).normal(0.02, 0.05, size=20)
    rng = np.random.default_rng(31)
    means = [differences[rng.integers(0, 20, size=20)].mean() for _ in range(1000)]
    result = stats.seed_interval(differences, n_boot=1000, rng=np.random.default_rng(31))
    assert result["estimate"] == pytest.approx(differences.mean(), abs=1e-15)
    assert result["ci"] == pytest.approx(list(np.quantile(means, [0.025, 0.975])), abs=1e-12)
    assert result["n_seeds"] == 20 and result["level"] == 0.95 and result["n_boot"] == 1000
    assert result == stats.seed_interval(differences, n_boot=1000, rng=np.random.default_rng(31))
    assert result["ci"] != stats.seed_interval(differences, n_boot=1000, rng=np.random.default_rng(32))["ci"]


def test_seed_interval_edge_cases():
    constant = stats.seed_interval([0.25] * 20, n_boot=50, rng=np.random.default_rng(0))
    assert constant["estimate"] == pytest.approx(0.25) and constant["ci"] == pytest.approx([0.25, 0.25], abs=1e-12)
    with pytest.raises(ValueError, match="finite"):
        stats.seed_interval([0.1] * 19 + [float("nan")], rng=np.random.default_rng(0))  # a missing seed
    with pytest.raises(ValueError, match="finite"):
        stats.seed_interval([0.1] * 19 + [None], rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="one seed"):
        stats.seed_interval([0.1], rng=np.random.default_rng(0))


# ---- hierarchical_interval --------------------------------------------------------------------

def row_mean(values, draw):
    return float(np.mean(values))


def test_datasets_are_resampled_first():
    """Two datasets worth 0 and 1 with no spread inside: the draws are (0,0), (0,1), (1,0), (1,1)."""
    values = np.r_[np.zeros(6), np.ones(6)]
    dataset = ["A"] * 6 + ["B"] * 6
    patient = list(range(6)) * 2  # the same patient ids in both datasets: twelve different patients
    result = stats.hierarchical_interval(values, (dataset, patient), row_mean, n_boot=2000, rng=np.random.default_rng(0))
    assert result["estimate"] == pytest.approx(0.5) and result["ci"] == pytest.approx([0.0, 1.0], abs=1e-12)
    assert result["n_datasets"] == 2 and result["n_units"] == 12 and result["n_rows"] == 12


def test_patients_are_resampled_inside_the_drawn_datasets_with_all_their_rows():
    # Dataset 0: 3 patients with 2 rows each. Dataset 1: 5 patients with 3 rows each.
    dataset = np.r_[np.zeros(6, dtype=int), np.ones(15, dtype=int)]
    patient = np.r_[np.repeat([0, 1, 2], 2), np.repeat([0, 1, 2, 3, 4], 3)]
    table = np.column_stack([dataset, patient, np.arange(21)])
    seen = []

    def record(values, draw):
        seen.append((values.copy(), draw.copy()))
        return 0.0

    stats.hierarchical_interval(table, (dataset, patient), record, n_boot=60, rng=np.random.default_rng(1))
    assert len(seen) == 61  # sixty resamples and the estimate
    for values, draw in seen[:60]:
        assert sorted(set(draw.tolist())) == [0, 1]  # as many dataset draws as there are datasets
        for j in (0, 1):
            rows = values[draw == j]
            assert len(set(rows[:, 0].tolist())) == 1  # one draw, one dataset
            d = int(rows[0, 0])
            assert len(rows) == (6 if d == 0 else 15)  # as many patients as the dataset has, whole
            for p in set(rows[:, 1].tolist()):
                copies = np.bincount(rows[rows[:, 1] == p][:, 2], minlength=21)
                own = np.flatnonzero((dataset == d) & (patient == p))
                assert len(set(copies[own].tolist())) == 1 and copies.sum() == copies[own].sum()
    values, draw = seen[60]
    assert (values == table).all() and (draw == dataset).all()
    kinds = {tuple(sorted(int(values[draw == j][0, 0]) for j in (0, 1))) for values, draw in seen[:60]}
    assert kinds == {(0, 0), (0, 1), (1, 1)}  # a dataset drawn twice is two draws, each with its own patients


def test_hierarchical_restated_independently():
    rng = np.random.default_rng(2)
    dataset = np.repeat([0, 1, 2], [8, 5, 11])
    patient = np.r_[np.arange(8), np.arange(5), np.arange(11)]
    values = rng.normal(size=24) + dataset
    members = [np.flatnonzero(dataset == d) for d in range(3)]
    rng, means = np.random.default_rng(17), []
    for _ in range(400):
        rows = []
        for d in rng.integers(0, 3, size=3):
            rows.extend(members[d][rng.integers(0, len(members[d]), size=len(members[d]))])
        means.append(values[rows].mean())
    result = stats.hierarchical_interval(values, (dataset, patient), row_mean, n_boot=400, rng=np.random.default_rng(17))
    assert result["ci"] == pytest.approx(list(np.quantile(means, [0.025, 0.975])), abs=1e-12)
    assert result == stats.hierarchical_interval(values, (dataset, patient), row_mean, n_boot=400, rng=np.random.default_rng(17))


def test_an_undefined_resample_makes_the_interval_unassessable():
    values, dataset, patient = np.arange(8.0), [0] * 4 + [1] * 4, list(range(8))
    one_dataset_only = lambda rows, draw: float("nan") if rows.min() >= 4 or rows.max() < 4 else float(rows.mean())
    result = stats.hierarchical_interval(values, (dataset, patient), one_dataset_only, n_boot=200, rng=np.random.default_rng(0))
    assert result["status"] == "unassessable" and result["ci"] is None
    assert 0 < result["valid_resamples"] < 200 and result["estimate"] == pytest.approx(3.5)
    with pytest.raises(ValueError, match="dataset_ids, patient_ids"):
        stats.hierarchical_interval(values, (dataset,), row_mean, rng=np.random.default_rng(0))
