"""E0 fixtures, part 4: the tempered family, tau_B, CEE-Scale and tau_ratio (SPEC 5.5, 5.11; PROOF section 6).

Hand anchor S (a symmetric power distortion, r' in closed form). The coherent joint
[[0.4, 0.1], [0.1, 0.4]] has P_hat = b = (1/2, 1/2), posteriors q0_1 = (0.8, 0.2), q0_2 = (0.2, 0.8)
and mutual information log 2 - H(0.8, 0.2) = 0.693147 - 0.500402 = 0.192745 nats.

An over-reactive reader squares the increments (gamma = 2): q_1 is proportional to
b (q0_1 / b)^2 = (1/2)(1.6^2, 0.4^2), that is q_1 = (16/17, 1/17), and q_2 is its mirror image.
Tempering at tau gives q_1(tau) proportional to (16^(1/tau), 1), so its first entry is

    p(tau) = 16^(1/tau) / (1 + 16^(1/tau)),

and q_2(tau) is the mirror. By symmetry the imagined mean is (1/2, 1/2) = b for every tau: the
tempered pair is always coherent, the projection leaves it alone (KL_inc = 0), and

    r'(tau) = log 2 - H(p(tau), 1 - p(tau)).

    r'(1) = log 2 - H(16/17, 1/17) = 0.693147 - 0.223718 = 0.469429
    r'(2) = log 2 - H(4/5, 1/5)    = 0.192745        (16^(1/2) = 4)
    r'(4) = log 2 - H(2/3, 1/3)    = 0.693147 - 0.636514 = 0.056633        (16^(1/4) = 2)
    r'(16) = log 2 - H(0.543214, 0.456786) = 0.693147 - 0.689408 = 0.003739  (16^(1/16) = 1.189207)

So with budget B = 0.192745 the solution of r'(tau) = B is tau_B = 2 exactly, and the repaired
posteriors are q0. An under-reactive reader with gamma = 1/2 has q_1 = (2/3, 1/3); the same budget
gives tau_B = 1/2, which the shrink-only interval [1, 16] clips to 1.

Analytic solver checks. The search is also run on a stand-in r'(tau) = 1/tau (monotone), where
r' = B has the solution 1/B, and on 1/tau plus a bump of height 1 on [7, 10] (not monotone), where
the largest tau with r' >= 0.7 is 10 while the first crossing is 1/0.7 = 1.43.
"""
import math

import numpy as np
import pytest

import tiicee.cee.scaling as scaling
from tiicee.cee import (
    TAU_RATIO_GRID,
    VARIANT_BOUNDS,
    ScaleResult,
    TauBudget,
    TauRatio,
    apply_scale_to_real,
    project,
    projected_responsiveness,
    scale,
    tau_budget,
    tau_ratio,
    temper,
)

TIGHT = dict(tol=1e-13, max_iter=200000)
TWO_SIDED = (0.125, 16.0)
SHRINK_ONLY = (1.0, 16.0)

S_B = np.array([0.5, 0.5])
S_P = np.array([0.5, 0.5])
S_Q0 = np.array([[0.8, 0.2], [0.2, 0.8]])
S_OVER = np.array([[16 / 17, 1 / 17], [1 / 17, 16 / 17]])        # gamma = 2
S_UNDER = np.array([[2 / 3, 1 / 3], [1 / 3, 2 / 3]])              # gamma = 1/2
S_BUDGET = math.log(2.0) + 0.8 * math.log(0.8) + 0.2 * math.log(0.2)


def _h2(p):
    return -p * math.log(p) - (1 - p) * math.log(1 - p)


def _s_r_prime(tau):
    """r'(tau) of hand anchor S for the gamma = 2 reader."""
    x = 16.0 ** (1.0 / tau)
    return math.log(2.0) - _h2(x / (1.0 + x))


def _floor(p):
    p = np.maximum(np.asarray(p, dtype=float), 1e-6)
    return p / p.sum(-1, keepdims=True)


def _kl(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    m = p > 0
    return float((p[m] * np.log(p[m] / q[m])).sum())


def _coherent_pair(rng, spread):
    """A random coherent pair: b, P_hat, posteriors q0, and the joint they come from."""
    C, K = int(rng.integers(2, 6)), int(rng.integers(2, 8))
    J = np.exp(rng.normal(scale=spread, size=(K, C)))
    J = J / J.sum()
    P, b = J.sum(1), J.sum(0)
    return b, P, J / P[:, None], J


def _distort(b, q0, gamma):
    q = b * (q0 / b) ** gamma
    return q / q.sum(1, keepdims=True)


def _mutual_information(J):
    return float((J * np.log(J / np.outer(J.sum(1), J.sum(0)))).sum())


def _random_cases(seed, n):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        C = int(rng.integers(2, 7))
        K = int(rng.integers(2, 10))
        yield rng.dirichlet(np.ones(C)), _floor(rng.dirichlet(np.ones(K))), rng.dirichlet(np.ones(C) * rng.uniform(0.3, 3.0), size=K)


# ------------------------------------------------------------------------------------------------
# The declared intervals and the tempered family
# ------------------------------------------------------------------------------------------------
def test_variant_intervals_are_the_specifications():
    assert VARIANT_BOUNDS == {"shrink-only": (1.0, 16.0), "two-sided": (0.125, 16.0)}
    assert TAU_RATIO_GRID == (0.5, 0.67, 1, 1.5, 2, 3, 5, 8)


def test_temper_by_hand():
    # b (q/b)^(1/tau), normalized. tau = 2 takes square roots of the increments.
    np.testing.assert_allclose(temper([0.5, 0.5], [0.8, 0.2], 2.0), [2 / 3, 1 / 3], atol=1e-15)       # sqrt(1.6) : sqrt(0.4) = 2 : 1
    np.testing.assert_allclose(temper([0.5, 0.5], [0.8, 0.2], 0.5), [16 / 17, 1 / 17], atol=1e-15)    # squares: 2.56 : 0.16
    np.testing.assert_allclose(temper([0.5, 0.5], [0.8, 0.2], 1.0), [0.8, 0.2], atol=1e-15)           # tau = 1 is the reader
    # Non-uniform prior: 0.2 sqrt(2.5) : 0.8 sqrt(0.625) = 0.316228 : 0.632456 = 1 : 2.
    np.testing.assert_allclose(temper([0.2, 0.8], [0.5, 0.5], 2.0), [1 / 3, 2 / 3], atol=1e-15)
    # Row-wise on a matrix of posteriors, and flat in the limit of a very large tau.
    np.testing.assert_allclose(temper(S_B, S_OVER, 2.0), S_Q0, atol=1e-15)
    np.testing.assert_allclose(temper([0.2, 0.8], [[0.5, 0.5], [0.9, 0.1]], 1e9), [[0.2, 0.8], [0.2, 0.8]], atol=1e-8)


def test_temper_then_inverse_temper_is_the_identity():
    rng = np.random.default_rng(4)
    for _ in range(30):
        b, _, q0, _ = _coherent_pair(rng, spread=0.5)
        for tau in (0.5, 0.8, 3.0):
            tempered = temper(b, q0, tau)
            assert tempered.min() > 1e-5          # above the floor, so the second call does not floor it again
            np.testing.assert_allclose(temper(b, tempered, 1.0 / tau), q0, atol=1e-12)


# ------------------------------------------------------------------------------------------------
# Hand anchor S
# ------------------------------------------------------------------------------------------------
def test_anchor_s_decimals_match_the_docstring():
    assert S_BUDGET == pytest.approx(0.192745, abs=5e-7)
    assert _s_r_prime(1.0) == pytest.approx(0.469429, abs=5e-7)
    assert _s_r_prime(2.0) == pytest.approx(S_BUDGET, abs=1e-15)
    assert _s_r_prime(4.0) == pytest.approx(0.056633, abs=5e-7)
    assert _s_r_prime(16.0) == pytest.approx(0.003739, abs=5e-7)
    np.testing.assert_allclose(_distort(S_B, S_Q0, 2.0), S_OVER, atol=1e-15)
    np.testing.assert_allclose(_distort(S_B, S_Q0, 0.5), S_UNDER, atol=1e-15)


@pytest.mark.parametrize("tau", [0.125, 0.5, 1.0, 2.0, 3.0, 4.0, 16.0])
def test_anchor_s_projected_responsiveness_closed_form(tau):
    assert projected_responsiveness(S_B, S_P, S_OVER, tau) == pytest.approx(_s_r_prime(tau), abs=1e-12)


def test_anchor_s_tau_b_is_two_and_the_coherent_reader_is_recovered():
    for variant in ("two-sided", "shrink-only"):
        res = scale(S_B, S_P, S_OVER, S_BUDGET, variant)
        assert isinstance(res, ScaleResult)
        assert res.path == "solved" and res.monotone and res.converged and res.search_converged
        assert abs(res.tau_b / 2.0 - 1.0) <= 0.01                         # 1% relative tolerance on tau
        np.testing.assert_allclose(res.repaired, S_Q0, atol=5e-3)
        assert abs(res.kl_inc) < 1e-12                                     # the tempered pair is coherent
        exact = scale(S_B, S_P, S_OVER, S_BUDGET, variant, rtol=1e-12)
        assert exact.tau_b == pytest.approx(2.0, abs=1e-9)
        np.testing.assert_allclose(exact.repaired, S_Q0, atol=1e-9)
        np.testing.assert_allclose(exact.u, [0.5, 0.5], atol=1e-12)


def test_anchor_s_under_reactive_reader():
    two_sided = scale(S_B, S_P, S_UNDER, S_BUDGET, "two-sided", rtol=1e-12)
    assert two_sided.path == "solved" and two_sided.tau_b == pytest.approx(0.5, abs=1e-9)
    np.testing.assert_allclose(two_sided.repaired, S_Q0, atol=1e-9)
    shrink = scale(S_B, S_P, S_UNDER, S_BUDGET, "shrink-only")
    assert shrink.path == "clipped_low" and shrink.tau_b == 1.0             # never amplified
    np.testing.assert_allclose(shrink.repaired, S_UNDER, atol=1e-12)        # left as the reader had it


def test_clipping_at_both_ends_of_both_intervals():
    # Budget below r'(16) = 0.003739: even the flattest allowed reader carries more. Clip high.
    for variant in ("two-sided", "shrink-only"):
        res = scale(S_B, S_P, S_OVER, 0.001, variant)
        assert res.path == "clipped_high" and res.tau_b == 16.0
        np.testing.assert_allclose(res.repaired, temper(S_B, S_OVER, 16.0), atol=1e-12)
        zero = scale(S_B, S_P, S_OVER, 0.0, variant)
        assert zero.path == "clipped_high" and zero.tau_b == 16.0
    # Budget above log 2 = H(b): no reader can carry it. Clip low, at 1/8 or at 1.
    res = scale(S_B, S_P, S_OVER, 0.6932, "two-sided")
    assert res.path == "clipped_low" and res.tau_b == 0.125
    res = scale(S_B, S_P, S_OVER, 0.6932, "shrink-only")
    assert res.path == "clipped_low" and res.tau_b == 1.0
    # Budget between r'(1) = 0.469429 and r'(1/8): solved by two-sided below 1, clipped by shrink-only.
    res = scale(S_B, S_P, S_OVER, 0.6, "two-sided")
    assert res.path == "solved" and 0.125 < res.tau_b < 1.0
    assert _s_r_prime(res.tau_b) == pytest.approx(0.6, rel=0.02)
    res = scale(S_B, S_P, S_OVER, 0.6, "shrink-only")
    assert res.path == "clipped_low" and res.tau_b == 1.0


def test_exact_re_exposure_has_zero_budget_and_returns_the_prior():
    # SPEC 5.11: a never-updating agent has r' = 0 and is a solution only when B = 0, the exact
    # re-exposure case. Hand anchor B of test_cee_core.py: all the mass on one outcome.
    b, point, Q = [0.5, 0.3, 0.2], [1.0, 0.0], [[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]]
    for tau in (0.125, 1.0, 16.0):
        assert projected_responsiveness(b, point, Q, tau) == pytest.approx(0.0, abs=1e-12)    # q' = b whatever tau is
    for variant in ("two-sided", "shrink-only"):
        res = scale(b, point, Q, 0.0, variant)
        assert res.path == "clipped_high" and res.tau_b == 16.0 and res.converged
        np.testing.assert_allclose(res.repaired[0], b, atol=1e-9)             # duplicate evidence moves nothing
    for bb, w, QQ in _random_cases(41, 100):
        certain = np.zeros(len(w))
        certain[0] = 1.0
        found = tau_budget(bb, certain, QQ, 0.0, TWO_SIDED)
        assert found.path == "clipped_high" and found.tau == 16.0 and np.all(found.r_grid >= 0.0)


def test_a_reader_that_never_moves_cannot_meet_a_positive_budget():
    b, Q = [0.3, 0.7], [[0.3, 0.7], [0.3, 0.7]]
    two_sided = scale(b, [0.5, 0.5], Q, 0.05, "two-sided")
    assert two_sided.path == "clipped_low" and two_sided.tau_b == 0.125
    np.testing.assert_allclose(two_sided.repaired, Q, atol=1e-15)             # amplifying nothing is still nothing
    shrink = scale(b, [0.5, 0.5], Q, 0.05, "shrink-only")
    assert shrink.path == "clipped_low" and shrink.tau_b == 1.0


def test_tau_budget_reports_the_same_search_with_caller_bounds():
    found = tau_budget(S_B, S_P, S_OVER, S_BUDGET, TWO_SIDED)
    assert isinstance(found, TauBudget)
    assert found.path == "solved" and found.monotone and found.converged
    assert abs(found.tau / 2.0 - 1.0) <= 0.01
    np.testing.assert_allclose(found.grid, [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0], rtol=0, atol=0)
    np.testing.assert_allclose(found.r_grid, [_s_r_prime(t) for t in found.grid], atol=1e-12)
    # Any interval may be asked for: [2.5, 3] lies wholly above the solution, so it clips low.
    narrow = tau_budget(S_B, S_P, S_OVER, S_BUDGET, (2.5, 3.0))
    assert narrow.path == "clipped_low" and narrow.tau == 2.5
    np.testing.assert_allclose(narrow.grid, [2.5, 3.0], rtol=0, atol=0)
    wide = tau_budget(S_B, S_P, S_OVER, S_BUDGET, (1.5, 2.6), rtol=1e-12)
    assert wide.path == "solved" and wide.tau == pytest.approx(2.0, abs=1e-9)


# ------------------------------------------------------------------------------------------------
# The solver against analytic stand-ins for r' (monkeypatched)
# ------------------------------------------------------------------------------------------------
def _patch_r_prime(monkeypatch, r_of_tau):
    real = scaling._projected

    def fake(prior, weights, posteriors, tau, tol, max_iter):
        projection, _ = real(prior, weights, posteriors, tau, tol, max_iter)
        return projection, r_of_tau(tau)

    monkeypatch.setattr(scaling, "_projected", fake)


@pytest.mark.parametrize("budget, expected", [(0.7, 1 / 0.7), (0.3, 1 / 0.3), (3.0, 1 / 3.0), (0.1, 10.0), (0.25, 4.0), (1.0, 1.0)])
def test_solver_on_a_monotone_stand_in(monkeypatch, budget, expected):
    _patch_r_prime(monkeypatch, lambda tau: 1.0 / tau)
    found = tau_budget(S_B, S_P, S_OVER, budget, TWO_SIDED)
    assert found.path == "solved" and found.monotone
    assert abs(found.tau / expected - 1.0) <= 0.01
    tight = tau_budget(S_B, S_P, S_OVER, budget, TWO_SIDED, rtol=1e-10)
    assert tight.tau == pytest.approx(expected, rel=1e-9)


def test_solver_clips_on_a_monotone_stand_in(monkeypatch):
    _patch_r_prime(monkeypatch, lambda tau: 1.0 / tau)
    low = tau_budget(S_B, S_P, S_OVER, 10.0, TWO_SIDED)            # r'(1/8) = 8 < 10
    assert low.path == "clipped_low" and low.tau == 0.125
    high = tau_budget(S_B, S_P, S_OVER, 0.01, TWO_SIDED)           # r'(16) = 0.0625 > 0.01
    assert high.path == "clipped_high" and high.tau == 16.0
    at_end = tau_budget(S_B, S_P, S_OVER, 0.0625, TWO_SIDED)       # r'(16) equals the budget: 16 is a solution
    assert at_end.path == "clipped_high" and at_end.tau == 16.0
    shrink = tau_budget(S_B, S_P, S_OVER, 3.0, SHRINK_ONLY)        # solution 1/3 lies below the interval
    assert shrink.path == "clipped_low" and shrink.tau == 1.0


def test_fallback_takes_the_largest_tau_that_meets_the_budget(monkeypatch):
    _patch_r_prime(monkeypatch, lambda tau: 1.0 / tau + (1.0 if 7.0 <= tau <= 10.0 else 0.0))
    found = tau_budget(S_B, S_P, S_OVER, 0.7, TWO_SIDED)
    assert found.monotone is False and found.path == "fallback"
    assert abs(found.tau / 10.0 - 1.0) <= 0.01                     # not the first crossing at 1.43
    # The record shows why: r' rises from 0.25 at tau = 4 to 1.125 at tau = 8.
    np.testing.assert_allclose(found.r_grid, [8, 4, 2, 1, 0.5, 0.25, 1.125, 0.0625], atol=1e-15)
    res = scale(S_B, S_P, S_OVER, 0.7, "shrink-only")
    assert res.path == "fallback" and res.monotone is False and abs(res.tau_b / 10.0 - 1.0) <= 0.01
    # A non-monotone r' that never meets the budget above the lower end still clips low...
    low = tau_budget(S_B, S_P, S_OVER, 9.0, TWO_SIDED)
    assert low.path == "clipped_low" and low.monotone is False and low.tau == 0.125
    # ...and one that meets it at the upper end still clips high.
    high = tau_budget(S_B, S_P, S_OVER, 0.05, TWO_SIDED)
    assert high.path == "clipped_high" and high.monotone is False and high.tau == 16.0


# ------------------------------------------------------------------------------------------------
# Random fixtures: power distortion, monotonicity, the two variants, the never-updating reader
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("variant, gamma", [("two-sided", 0.5), ("two-sided", 0.8), ("two-sided", 1.7), ("two-sided", 3.0), ("shrink-only", 1.7), ("shrink-only", 3.0)])
def test_power_distortion_is_recovered_by_cee_scale(variant, gamma):
    rng = np.random.default_rng(100 + int(gamma * 10))
    for _ in range(12):
        b, P, q0, J = _coherent_pair(rng, spread=0.5)
        q = _distort(b, q0, gamma)
        assert min(q.min(), q0.min(), b.min()) > 1e-5                 # the floor is inactive in this fixture
        budget = _mutual_information(J)
        res = scale(b, P, q, budget, variant)                         # the specified 1% tolerance
        assert res.path == "solved" and res.converged
        assert abs(res.tau_b / gamma - 1.0) <= 0.01
        np.testing.assert_allclose(res.repaired, q0, atol=0.02)
        exact = scale(b, P, q, budget, variant, rtol=1e-11, **TIGHT)  # exact recovery with the search tightened
        assert exact.tau_b == pytest.approx(gamma, rel=1e-7)
        np.testing.assert_allclose(exact.repaired, q0, atol=1e-7)
        assert abs(exact.kl_inc) < 1e-10                              # the repaired pair is coherent
        assert float(P @ [_kl(row, b) for row in exact.repaired]) == pytest.approx(budget, abs=1e-9)   # and carries B


def test_power_distortion_outside_the_shrink_only_interval_is_clipped():
    rng = np.random.default_rng(9)
    for _ in range(12):
        b, P, q0, J = _coherent_pair(rng, spread=0.5)
        res = scale(b, P, _distort(b, q0, 0.5), _mutual_information(J), "shrink-only")
        assert res.path == "clipped_low" and res.tau_b == 1.0


def test_r_prime_is_monotone_in_tau_on_the_fixture_set():
    taus = np.geomspace(1 / 8, 16, 15)
    worst = -np.inf
    for b, w, Q in _random_cases(20261113, 60):
        r = np.array([projected_responsiveness(b, w, Q, t, **TIGHT) for t in taus])
        worst = max(worst, float(np.diff(r).max()))                   # r' must not increase with tau
        assert np.all(r >= -1e-12)
        found = tau_budget(b, w, Q, 0.5 * r[7], TWO_SIDED, **TIGHT)
        assert found.converged and found.monotone and found.path in ("solved", "clipped_low", "clipped_high")
    rng = np.random.default_rng(6)
    for _ in range(40):
        b, P, q0, _ = _coherent_pair(rng, spread=0.6)
        r = np.array([projected_responsiveness(b, P, _distort(b, q0, 2.0), t, **TIGHT) for t in taus])
        worst = max(worst, float(np.diff(r).max()))
    assert worst < 1e-9


def test_shrink_only_is_two_sided_clipped_at_one():
    for b, w, Q in _random_cases(77, 40):
        base = projected_responsiveness(b, w, Q, 1.0)
        for factor in (0.3, 0.9, 1.5, 4.0):
            two = tau_budget(b, w, Q, factor * base, TWO_SIDED)
            one = tau_budget(b, w, Q, factor * base, SHRINK_ONLY)
            assert one.tau == max(1.0, two.tau)                       # exactly: the two searches share their grid
            assert one.path == (two.path if two.tau >= 1.0 else "clipped_low")


def test_solved_tau_b_carries_the_budget():
    n_solved = 0
    for b, w, Q in _random_cases(88, 40):
        budget = 0.5 * projected_responsiveness(b, w, Q, 1.0)
        res = scale(b, w, Q, budget, "two-sided", rtol=1e-11, **TIGHT)
        if res.path != "solved":
            continue
        n_solved += 1
        carried = float(res.projection.weights @ [_kl(row, res.projection.prior) for row in res.repaired])
        assert carried == pytest.approx(budget, abs=1e-8)
        assert res.tau_b > 1.0                                        # half the reader's own r' means flatten
        np.testing.assert_allclose(res.projection.weights @ res.repaired, res.projection.prior, atol=1e-11)
    assert n_solved >= 35


def test_a_never_updating_reader_is_coherent_and_tau_b_does_not_select_it():
    for b, w, Q in _random_cases(20261115, 60):
        flat = temper(b, Q, 1e7)
        assert abs(project(b, w, flat).kl_inc) < 1e-8                 # coherent: minimizing KL_inc over tau picks this
        assert projected_responsiveness(b, w, Q, 1e7) < 1e-10         # and it carries nothing
        budget = 0.5 * projected_responsiveness(b, w, Q, 1.0)
        if budget <= 1e-9:
            continue
        found = tau_budget(b, w, Q, budget, TWO_SIDED)
        assert found.tau <= 16.0
        assert projected_responsiveness(b, w, Q, found.tau) > 0.5 * budget   # the selected scale still carries information


def test_the_same_tau_and_weights_are_applied_to_the_real_result():
    for b, w, Q in _random_cases(99, 30):
        res = scale(b, w, Q, 0.5 * projected_responsiveness(b, w, Q, 1.0), "two-sided")
        bf, Qf = _floor(b), _floor(Q)
        for o in range(len(w)):
            corrected, log_z = apply_scale_to_real(b, Q[o], res.tau_b, res.u)
            tempered = bf * (Qf[o] / bf) ** (1.0 / res.tau_b)
            tempered = tempered / tempered.sum()
            z = float(tempered @ res.u)
            np.testing.assert_allclose(corrected, res.u * tempered / z, atol=1e-13)
            assert log_z == pytest.approx(math.log(z), abs=1e-12)
            np.testing.assert_allclose(corrected, res.repaired[o], atol=1e-12)     # an imagined result that came true
            assert log_z == pytest.approx(res.log_z[o], abs=1e-12)


# ------------------------------------------------------------------------------------------------
# tau_ratio: a secondary diagnostic
# ------------------------------------------------------------------------------------------------
def test_tau_ratio_table_matches_the_definition():
    for b, w, Q in _random_cases(123, 20):
        found = tau_ratio(b, w, Q)
        assert isinstance(found, TauRatio)
        np.testing.assert_array_equal(found.grid, TAU_RATIO_GRID)
        bf, Qf = _floor(b), _floor(Q)
        for i, tau in enumerate(TAU_RATIO_GRID):
            tempered = bf * (Qf / bf) ** (1.0 / tau)
            tempered = tempered / tempered.sum(1, keepdims=True)
            mean = w @ tempered
            r = sum(w[o] * _kl(tempered[o], mean) for o in range(len(w)))     # responsiveness of the tempered raw pair
            assert found.responsiveness[i] == pytest.approx(r, abs=1e-12)
            if found.converged[i]:
                assert found.ratio[i] == pytest.approx(found.kl_inc[i] / r, rel=1e-9, abs=1e-12)
        assert found.kl_inc[2] == pytest.approx(project(b, w, Q).kl_inc, abs=1e-12)   # tau = 1 is the plain projection
        usable = np.flatnonzero(found.converged & (found.responsiveness > 0))
        assert found.tau == TAU_RATIO_GRID[usable[np.argmin(found.ratio[usable])]]


@pytest.mark.parametrize("gamma", TAU_RATIO_GRID)
def test_tau_ratio_recovers_a_pure_scale_error_on_its_grid(gamma):
    rng = np.random.default_rng(int(gamma * 100))
    for _ in range(8):
        b, P, q0, _ = _coherent_pair(rng, spread=0.15)
        q = _distort(b, q0, gamma)
        assert min(q.min(), b.min()) > 1e-5
        assert tau_ratio(b, P, q, **TIGHT).tau == gamma


def test_tau_ratio_is_undefined_for_a_reader_that_never_moves():
    found = tau_ratio([0.3, 0.7], [0.5, 0.5], [[0.3, 0.7], [0.3, 0.7]])
    assert math.isnan(found.tau) and np.all(np.isnan(found.ratio))


def test_tau_ratio_is_documented_as_a_secondary_diagnostic_and_never_a_repair():
    doc = " ".join(tau_ratio.__doc__.split())
    assert "secondary diagnostic" in doc and "never a repair" in doc


# ------------------------------------------------------------------------------------------------
# Bad arguments
# ------------------------------------------------------------------------------------------------
def test_bad_arguments_are_rejected():
    with pytest.raises(ValueError):
        scale(S_B, S_P, S_OVER, S_BUDGET, "amplify-only")
    with pytest.raises(ValueError):
        scale(S_B, S_P, S_OVER, -0.1, "two-sided")
    with pytest.raises(ValueError):
        tau_budget(S_B, S_P, S_OVER, float("nan"), TWO_SIDED)
    with pytest.raises(ValueError):
        tau_budget(S_B, S_P, S_OVER, S_BUDGET, (4.0, 2.0))
    with pytest.raises(ValueError):
        temper(S_B, S_OVER, 0.0)
    with pytest.raises(ValueError):
        projected_responsiveness(S_B, S_P, S_OVER, -1.0)
