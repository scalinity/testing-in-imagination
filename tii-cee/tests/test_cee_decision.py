"""E0 fixtures, part 5: coherent value of information, CEE-Select, CEE-Commit and CEE-Gate (SPEC 5.6 to 5.8).

Hand anchor A with the twin's utility 1[correct] (see test_cee_core.py for the projection).
    phi(b) = max(1/2, 1/2) = 1/2
    raw score      = (1/2) max(0.8, 0.2) + (1/2) max(0.6, 0.4) - 1/2 = 0.7 - 0.5 = 0.2
    coherent VOI   = (1/2)(1.6 - 0.4 sqrt6) + (1/2)(1.6 - 0.4 sqrt6) - 1/2 = 1.1 - 0.4 sqrt6 = 0.120204
The raw score overstates the value because part of the imagined movement is drift.

Hand anchor D (pure drift): b = (0.9, 0.1), both outcomes give q = (0.6, 0.4).
    raw score      = 0.6 - 0.9 = -0.3      (negative: the reader expects to know less after the test)
    coherent VOI   = 0                      (both corrected posteriors equal b)

Recall-aligned utility. Class frequencies pi = (1/2, 1/4, 1/4), C = 3: U(d, y) = 1[d = y] / (3 pi_y),
that is diag(2/3, 4/3, 4/3). For a rule with P(d | y) rows (0.9, 0.1, 0), (0.2, 0.6, 0.2),
(0.1, 0.1, 0.8) the expected utility is sum_y pi_y P(d = y | y) / (3 pi_y) = (0.9 + 0.6 + 0.8)/3,
the mean class recall 0.766667.

Bootstrap by enumeration. Three groups with (sum, rows) = (0, 2), (1, 1), (6, 3). The 27 equally
likely ordered resamples give the ratio sum/rows:
    0 (1 way), 0.2 (3), 0.5 (3), 6/7 (3), 1 (1), 7/6 (6), 1.5 (3), 1.6 (3), 13/7 (3), 2 (1)
with cumulative probabilities 1/27, 4/27, 7/27, 10/27, 11/27, 17/27, 20/27, 23/27, 26/27, 1.
The 0.90 quantile lies inside the mass at 13/7 (cumulative 0.852 to 0.963), the 0.80 quantile
inside the mass at 1.6 (0.741 to 0.852), the 0.50 quantile inside the mass at 7/6 (0.407 to 0.630).
"""
import itertools
import math

import numpy as np
import pytest

from tiicee.cee import (
    bootstrap_upper_bound,
    coherent_voi,
    commit,
    expected_charged_cost,
    gate_allows,
    identity_utility,
    net_value,
    phi,
    project,
    raw_voi_score,
    recall_aligned_utility,
    select,
)
from tiicee.interfaces import UNAVAILABLE

SQRT6 = math.sqrt(6.0)
TIGHT = dict(tol=1e-13, max_iter=20000)


def _floor(p):
    p = np.maximum(np.asarray(p, dtype=float), 1e-6)
    return p / p.sum(-1, keepdims=True)


def _random_cases(seed, n):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        C = int(rng.integers(2, 6))
        K = int(rng.integers(2, 8))
        yield rng, rng.dirichlet(np.ones(C)), _floor(rng.dirichlet(np.ones(K))), rng.dirichlet(np.ones(C), size=K)


# ------------------------------------------------------------------------------------------------
# phi and the utility tables
# ------------------------------------------------------------------------------------------------
def test_phi_by_hand():
    assert phi([0.7, 0.3], np.eye(2)) == pytest.approx(0.7)
    U = [[1.0, 0.0], [0.0, 3.0]]                                   # diagnosis 1 is worth three times as much when right
    assert phi([0.7, 0.3], U) == pytest.approx(0.9)                # max(0.7, 0.9)
    np.testing.assert_allclose(phi([[0.7, 0.3], [0.9, 0.1]], U), [0.9, 0.9])     # row-wise: max(0.9, 0.3) for the second
    three = [[1.0, 0.0], [0.0, 1.0], [0.6, 0.6]]                   # a third diagnosis that hedges
    assert phi([0.5, 0.5], three) == pytest.approx(0.6)


def test_identity_utility_is_one_when_correct():
    np.testing.assert_array_equal(identity_utility(4), np.eye(4))


def test_recall_aligned_utility_by_hand():
    U = recall_aligned_utility([0.5, 0.25, 0.25])
    np.testing.assert_allclose(U, np.diag([2 / 3, 4 / 3, 4 / 3]), atol=1e-15)
    np.testing.assert_allclose(recall_aligned_utility([2, 1, 1]), U, atol=1e-15)      # counts give the same table
    pi = np.array([0.5, 0.25, 0.25])
    rule = np.array([[0.9, 0.1, 0.0], [0.2, 0.6, 0.2], [0.1, 0.1, 0.8]])              # P(d | y), rows y
    expected_utility = sum(pi[y] * rule[y, d] * U[d, y] for y in range(3) for d in range(3))
    assert expected_utility == pytest.approx((0.9 + 0.6 + 0.8) / 3, abs=1e-15)          # mean class recall
    with pytest.raises(ValueError):
        recall_aligned_utility([0.5, 0.5, 0.0])


# ------------------------------------------------------------------------------------------------
# Coherent value of information and the raw score
# ------------------------------------------------------------------------------------------------
def test_anchor_a_voi_by_hand():
    res = project([0.5, 0.5], [0.5, 0.5], [[0.8, 0.2], [0.6, 0.4]], tol=1e-14, max_iter=10000)
    U = identity_utility(2)
    assert coherent_voi(res.prior, res.weights, res.corrected, U) == pytest.approx(1.1 - 0.4 * SQRT6, abs=1e-11)
    assert 1.1 - 0.4 * SQRT6 == pytest.approx(0.120204, abs=5e-7)
    assert raw_voi_score([0.5, 0.5], [0.5, 0.5], [[0.8, 0.2], [0.6, 0.4]], U) == pytest.approx(0.2, abs=1e-15)


def test_anchor_d_raw_score_is_negative_and_coherent_voi_is_zero():
    b, w, Q = [0.9, 0.1], [0.5, 0.5], [[0.6, 0.4], [0.6, 0.4]]
    U = identity_utility(2)
    assert raw_voi_score(b, w, Q, U) == pytest.approx(-0.3, abs=1e-15)
    res = project(b, w, Q)
    assert coherent_voi(res.prior, res.weights, res.corrected, U) == pytest.approx(0.0, abs=1e-12)


def test_coherent_voi_is_non_negative_for_random_utilities_and_the_raw_score_is_not():
    min_voi, n_raw_negative, n = np.inf, 0, 0
    for rng, b, w, Q in _random_cases(20261113, 600):
        res = project(b, w, Q, **TIGHT)
        if not res.converged:
            continue
        n += 1
        U = rng.normal(size=(int(rng.integers(2, 6)), len(b)))                     # any utility table
        value = coherent_voi(res.prior, res.weights, res.corrected, U)
        by_definition = float(res.weights @ (res.corrected @ U.T).max(1) - (U @ res.prior).max())
        assert value == pytest.approx(by_definition, abs=1e-13)
        min_voi = min(min_voi, value)
        raw = raw_voi_score(b, w, Q, U)
        assert raw == pytest.approx(float(w @ (_floor(Q) @ U.T).max(1) - (U @ _floor(b)).max()), abs=1e-13)
        n_raw_negative += raw < -1e-12
    assert n >= 594
    assert min_voi > -1e-9                    # Jensen, given the marginal constraint
    assert n_raw_negative > 0                 # the raw pair does produce negative scores


def test_coherent_voi_at_the_default_tolerance_degrades_only_by_the_residual():
    # PROOF section 5: with residual m - b the bound degrades by at most max|U| x ||m - b||_1.
    for rng, b, w, Q in _random_cases(7, 100):
        res = project(b, w, Q)
        if not res.converged:
            continue
        U = rng.normal(size=(3, len(b)))
        slack = np.abs(U).max() * len(b) * res.max_violation
        assert coherent_voi(res.prior, res.weights, res.corrected, U) >= -slack - 1e-12


# ------------------------------------------------------------------------------------------------
# Cost and net value
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "outcomes, weights",
    [
        ((UNAVAILABLE,), [1.0]),                                                  # p_available = 0
        (("low", "normal", "high"), [0.2, 0.5, 0.3]),                             # p_available = 1
        (("low", "normal", "high", UNAVAILABLE), [0.2, 0.3, 0.1, 0.4]),           # p_available = 0.6
        (("positive", UNAVAILABLE), [0.07, 0.93]),                                # p_available = 0.07
    ],
)
def test_expected_charged_cost_against_enumeration(outcomes, weights):
    # REVIEW-006: unavailable results are free, so the expected charge is the sum over outcomes of
    # the world-model weight times what that outcome costs.
    cost = 0.35
    enumerated = sum(w * (0.0 if o == UNAVAILABLE else cost) for o, w in zip(outcomes, weights))
    p_available = sum(w for o, w in zip(outcomes, weights) if o != UNAVAILABLE)
    assert expected_charged_cost(p_available, cost) == pytest.approx(enumerated, abs=1e-15)
    assert expected_charged_cost(p_available, cost) == pytest.approx(p_available * cost, abs=1e-15)


def test_expected_charged_cost_end_points_and_bad_probability():
    assert expected_charged_cost(0.0, 5.0) == 0.0
    assert expected_charged_cost(1.0, 5.0) == 5.0
    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            expected_charged_cost(bad, 1.0)


def test_net_value_by_hand():
    assert net_value(0.3, 0.1) == pytest.approx(0.2)                 # primary lambda is 1
    assert net_value(0.3, 0.1, lam=0.0) == pytest.approx(0.3)
    assert net_value(0.3, 0.1, lam=0.5) == pytest.approx(0.25)
    assert net_value(0.3, 0.1, lam=2.0) == pytest.approx(0.1)
    assert net_value(0.05, 0.1) == pytest.approx(-0.05)


# ------------------------------------------------------------------------------------------------
# CEE-Select and CEE-Commit
# ------------------------------------------------------------------------------------------------
def test_select_takes_the_highest_positive_net_value():
    assert select([0.1, 0.4, 0.2]) == 1
    assert select([0.1, 0.3, 0.3, -0.2]) == 1                        # tie: lowest index
    assert select([-0.1, 0.0, -0.5]) is None                         # zero is not positive
    assert select([]) is None
    assert select([0.1, 0.4, 0.2], eligible=[True, False, True]) == 2
    assert select([0.1, 0.4, 0.2], eligible=[False, False, False]) is None
    assert select([-0.1, 0.4], eligible=[True, False]) is None       # the only positive test is not eligible
    assert isinstance(select([0.1, 0.4]), int)
    with pytest.raises(ValueError):
        select([0.1, 0.4], eligible=[True])


def test_commit_when_no_eligible_test_has_positive_net_value():
    U = identity_utility(3)
    assert commit([0.2, 0.5, 0.3], U, [-0.1, 0.0]) == (True, 1)
    assert commit([0.2, 0.5, 0.3], U, []) == (True, 1)                                # nothing left to request
    assert commit([0.2, 0.5, 0.3], U, [0.4, -0.1], eligible=[False, True]) == (True, 1)
    assert commit([0.2, 0.5, 0.3], U, [0.4, -0.1]) == (False, None)                   # a test is still worth requesting


def test_commit_when_the_budget_is_reached_even_if_a_test_is_worth_requesting():
    assert commit([0.2, 0.5, 0.3], identity_utility(3), [0.4, 0.2], budget_reached=True) == (True, 1)


def test_commit_maximizes_expected_utility_not_probability():
    # pi = (3/4, 1/4), C = 2: U = diag(1/(2 x 0.75), 1/(2 x 0.25)) = diag(2/3, 2). Belief (0.6, 0.4)
    # gives expected utilities 0.4 and 0.8, so the rarer class is the diagnosis.
    U = recall_aligned_utility([0.75, 0.25])
    np.testing.assert_allclose(U, np.diag([2 / 3, 2.0]), atol=1e-15)
    assert commit([0.6, 0.4], U, []) == (True, 1)
    assert commit([0.6, 0.4], identity_utility(2), []) == (True, 0)
    assert commit([0.5, 0.5], identity_utility(2), []) == (True, 0)   # tie: lowest index
    flag, diagnosis = commit([0.6, 0.4], U, [])
    assert isinstance(flag, bool) and isinstance(diagnosis, int)


# ------------------------------------------------------------------------------------------------
# CEE-Gate
# ------------------------------------------------------------------------------------------------
def test_gate_allows_only_above_the_sum_of_the_two_bounds():
    assert gate_allows(0.05, 0.02, 0.02) is True
    assert gate_allows(0.04, 0.02, 0.02) is False                    # strict inequality
    assert gate_allows(0.03, 0.02, 0.02) is False
    assert gate_allows(0.0, 0.0, 0.0) is False                       # a coherent pair is never corrected


def test_bootstrap_of_a_single_group_is_its_mean():
    rng = np.random.default_rng(0)
    assert bootstrap_upper_bound([1.0, 2.0, 6.0], ["a", "a", "a"], rng, n_boot=200) == pytest.approx(3.0, abs=1e-15)


def test_bootstrap_resamples_groups_never_rows():
    # 500 rows of 0 from one admission and 500 rows of 1 from another. Resampling the two
    # admissions gives a mean of 0, 0.5 or 1 with probabilities 1/4, 1/2, 1/4, so the upper 90%
    # bound is 1. Resampling the 1000 rows would give about 0.5 + 1.28 x 0.5/sqrt(1000) = 0.52.
    values = np.r_[np.zeros(500), np.ones(500)]
    groups = np.r_[np.zeros(500, dtype=int), np.ones(500, dtype=int)]
    rng = np.random.default_rng(1)
    assert bootstrap_upper_bound(values, groups, rng, n_boot=4000) == 1.0
    assert bootstrap_upper_bound(values, groups, np.random.default_rng(1), level=0.5, n_boot=4000) == 0.5
    assert bootstrap_upper_bound(values, groups, np.random.default_rng(1), level=0.2, n_boot=4000) == 0.0


def test_bootstrap_against_exact_enumeration():
    values = [0.0, 0.0, 1.0, 2.0, 2.0, 2.0]
    groups = ["p1", "p1", "p2", "p3", "p3", "p3"]
    sums, rows = [0.0, 1.0, 6.0], [2, 1, 3]
    stats = sorted(sum(sums[g] for g in draw) / sum(rows[g] for g in draw) for draw in itertools.product(range(3), repeat=3))
    assert len(stats) == 27

    def exact_quantile(level, margin=0.03):
        value = next(v for i, v in enumerate(stats) if (i + 1) / 27 >= level)
        below = sum(s < value - 1e-12 for s in stats) / 27
        upto = sum(s <= value + 1e-12 for s in stats) / 27
        assert below + margin < level < upto - margin          # well inside one mass point
        return value

    assert exact_quantile(0.90) == pytest.approx(13 / 7) and exact_quantile(0.80) == pytest.approx(1.6)
    assert exact_quantile(0.50) == pytest.approx(7 / 6)
    for level in (0.90, 0.80, 0.50):
        got = bootstrap_upper_bound(values, groups, np.random.default_rng(2), level=level, n_boot=20000)
        assert got == pytest.approx(exact_quantile(level), abs=1e-12)


def test_bootstrap_of_an_absolute_mean():
    # Admission a: 400 rows of -1. Admission b: 400 rows of -0.5. Resampled means are -1, -0.75,
    # -0.5 with probabilities 1/4, 1/2, 1/4. Upper 90% bound of the mean: -0.5. Of its absolute
    # value: 1. The gate needs the second (SPEC 5.8: bounds of the absolute terms).
    values = np.r_[np.full(400, -1.0), np.full(400, -0.5)]
    groups = np.r_[np.full(400, "a"), np.full(400, "b")]
    assert bootstrap_upper_bound(values, groups, np.random.default_rng(3), n_boot=4000) == -0.5
    assert bootstrap_upper_bound(values, groups, np.random.default_rng(3), n_boot=4000, absolute=True) == 1.0


def test_bootstrap_is_reproducible_and_checks_its_input():
    values, groups = [0.1, 0.4, -0.2, 0.3, 0.0], [1, 1, 2, 3, 3]
    first = bootstrap_upper_bound(values, groups, np.random.default_rng(5))
    assert first == bootstrap_upper_bound(values, groups, np.random.default_rng(5))
    assert min(0.25, -0.2, 0.15) <= first <= max(0.25, -0.2, 0.15)       # between the smallest and largest group mean
    lower = bootstrap_upper_bound(values, groups, np.random.default_rng(5), level=0.5)
    assert lower <= first
    with pytest.raises(ValueError):
        bootstrap_upper_bound([0.1, 0.2], [1], np.random.default_rng(0))
    with pytest.raises(ValueError):
        bootstrap_upper_bound([], [], np.random.default_rng(0))
