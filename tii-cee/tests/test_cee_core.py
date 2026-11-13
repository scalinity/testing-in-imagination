"""E0 fixtures, part 1: the floor and the projection (SPEC 5 and 5.3; PROOF sections 2, 3 and 5).

Every expected value here is worked by hand below or computed inline from the definitions; none
comes from a library helper.

Hand anchor A. C = 2 classes, K = 2 outcomes, b = (1/2, 1/2), P_hat = (1/2, 1/2),
q_1 = (0.8, 0.2), q_2 = (0.6, 0.4). Only the ratio t = u(1)/u(2) matters. The corrected
first-class probabilities are

    q'_1(1) = 0.8 t / (0.8 t + 0.2) = 4t / (4t + 1),      q'_2(1) = 0.6 t / (0.6 t + 0.4) = 3t / (3t + 2).

Conservation, (1/2) q'_1(1) + (1/2) q'_2(1) = 1/2, reads 4t/(4t+1) + 3t/(3t+2) = 1, so

    4t(3t+2) + 3t(4t+1) = (4t+1)(3t+2)   =>   24 t^2 + 11 t = 12 t^2 + 11 t + 2   =>   t^2 = 1/6.

Hence u(2)/u(1) = sqrt(6), and rationalizing the denominators,

    q'_1 = (4/(4+sqrt6), sqrt6/(4+sqrt6))     = (1.6 - 0.4 sqrt6, 0.4 sqrt6 - 0.6) = (0.620204, 0.379796)
    q'_2 = (3/(3+2 sqrt6), 2 sqrt6/(3+2 sqrt6)) = (0.4 sqrt6 - 0.6, 1.6 - 0.4 sqrt6) = (0.379796, 0.620204).

With the unnormalized weights u = (1, sqrt6): Z_1 = 0.8 + 0.2 sqrt6, Z_2 = 0.6 + 0.4 sqrt6 and
Z_1 Z_2 = 0.48 + 0.32 sqrt6 + 0.12 sqrt6 + 0.48 = 0.96 + 0.44 sqrt6, so

    KL_inc = sum_y b log u - sum_o P_hat log Z = (1/4) log 6 - (1/2) log(0.96 + 0.44 sqrt6) = 0.092010.

Hand anchor B (exact re-exposure). b = (0.5, 0.3, 0.2); the world model puts all its mass on one
outcome whose raw posterior is q = (0.2, 0.3, 0.5). The constraint reads q' = b, so u is
proportional to b/q = (2.5, 1, 0.4), Z = sum_y u q = sum_y b = 1, and

    KL_inc = sum_y b log(b/q) = KL(b || q) = 0.5 log 2.5 + 0.2 log 0.4 = 0.274887.

The zero-weight outcome with raw posterior (0.7, 0.2, 0.1) is reweighted by the same u:
(1.75, 0.2, 0.04) / 1.99.

Hand anchor C (already coherent). The joint [[0.1, 0.2, 0.1], [0.3, 0.1, 0.2]] has outcome
marginal (0.4, 0.6), class marginal (0.4, 0.3, 0.3) and conditionals (0.25, 0.5, 0.25) and
(1/2, 1/6, 1/3). Its imagined mean is its class marginal, so u is constant and KL_inc = 0.

Hand anchor D (pure drift). b = (0.9, 0.1) and both outcomes give q = (0.6, 0.4). Whatever the
weights, u is proportional to b/q = (1.5, 0.25), both corrected posteriors equal b, and
KL_inc = KL(b || q) = 0.9 log 1.5 + 0.1 log 0.25 = 0.364919 - 0.138629 = 0.226289.
"""
import inspect
import math

import numpy as np
import pytest

from tiicee.cee import FLOOR, MAX_ITER, TOL, Projection, apply_to_real, floor_norm, project
from tiicee.cee.core import entropy, kl

SQRT6 = math.sqrt(6.0)
TIGHT = dict(tol=1e-13, max_iter=20000)


def _floor(p):
    """The spec-5 floor restated independently: floor at 1e-6, then renormalize."""
    p = np.maximum(np.asarray(p, dtype=float), 1e-6)
    return p / p.sum(-1, keepdims=True)


def _kl(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    m = p > 0
    return float((p[m] * np.log(p[m] / q[m])).sum())


def _random_cases(seed, n):
    """The case family of cee_check.py part 1: C in 2..6, K in 2..9, raw (unfloored) Dirichlet draws."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        C = int(rng.integers(2, 7))
        K = int(rng.integers(2, 10))
        b = rng.dirichlet(np.ones(C))
        w = _floor(rng.dirichlet(np.ones(K)))
        Q = rng.dirichlet(np.ones(C) * rng.uniform(0.3, 3.0), size=K)
        yield rng, b, w, Q


# ------------------------------------------------------------------------------------------------
# Constants and the floor
# ------------------------------------------------------------------------------------------------
def test_constants_are_the_specifications():
    assert FLOOR == 1e-6 and TOL == 1e-8 and MAX_ITER == 500
    params = inspect.signature(project).parameters
    assert params["tol"].default == 1e-8 and params["max_iter"].default == 500
    assert inspect.signature(floor_norm).parameters["floor"].default == 1e-6


def test_floor_norm_by_hand():
    # (1, 0): the zero is raised to 1e-6 and the vector is divided by 1 + 1e-6.
    np.testing.assert_allclose(floor_norm([1.0, 0.0]), [1.0 / (1.0 + 1e-6), 1e-6 / (1.0 + 1e-6)], rtol=0, atol=1e-18)
    # Nothing below the floor: unchanged.
    np.testing.assert_array_equal(floor_norm([0.5, 0.25, 0.25]), [0.5, 0.25, 0.25])
    # Row-wise on a matrix, and a caller-chosen floor.
    np.testing.assert_allclose(floor_norm([[1.0, 0.0], [0.5, 0.5]]), [[1 / (1 + 1e-6), 1e-6 / (1 + 1e-6)], [0.5, 0.5]], atol=1e-18)
    np.testing.assert_allclose(floor_norm([1.0, 0.0], floor=0.1), [1 / 1.1, 0.1 / 1.1], atol=1e-15)


def test_floor_norm_is_not_idempotent_and_the_second_pass_is_tiny():
    # A floored entry is 1e-6 / (1 + 1e-6) < 1e-6, so a second pass raises it again: relative change
    # of about 1e-6 in that entry, about 1e-12 in absolute terms. This is why reader vectors are
    # floored once, at the entry points, and derived quantities are not floored again.
    once = floor_norm([1.0, 0.0])
    twice = floor_norm(once)
    assert once[1] < 1e-6 < twice[1] * (1 + 1e-6)
    assert 0 < twice[1] - once[1] < 2e-12


def test_kl_and_entropy_by_hand():
    assert kl([0.8, 0.2], [0.5, 0.5]) == pytest.approx(0.8 * math.log(1.6) + 0.2 * math.log(0.4), abs=1e-15)
    assert kl([1.0, 0.0], [0.5, 0.5]) == pytest.approx(math.log(2.0), abs=1e-15)          # 0 log 0 = 0
    assert kl([0.3, 0.7], [0.3, 0.7]) == 0.0
    assert math.isinf(kl([0.5, 0.5], [1.0, 0.0]))
    np.testing.assert_allclose(kl([[0.8, 0.2], [0.5, 0.5]], [0.5, 0.5]), [0.8 * math.log(1.6) + 0.2 * math.log(0.4), 0.0], atol=1e-15)
    assert entropy([0.5, 0.5]) == pytest.approx(math.log(2.0), abs=1e-15)
    assert entropy([1.0, 0.0]) == 0.0


def test_kl_is_never_negative():
    # Gibbs' inequality holds in exact arithmetic; the computed sum is clipped so that rounding
    # cannot turn "r' >= 0" into False for a zero budget.
    rng = np.random.default_rng(0)
    for _ in range(2000):
        p = rng.dirichlet(np.ones(int(rng.integers(2, 7))))
        q = p * (1.0 + 1e-9 * rng.normal(size=p.size))
        assert kl(p, q / q.sum()) >= 0.0
        assert kl(p, p) == 0.0


# ------------------------------------------------------------------------------------------------
# Hand anchors
# ------------------------------------------------------------------------------------------------
ANCHOR_A = dict(prior=[0.5, 0.5], weights=[0.5, 0.5], posteriors=[[0.8, 0.2], [0.6, 0.4]])
ANCHOR_A_CORRECTED = np.array([[1.6 - 0.4 * SQRT6, 0.4 * SQRT6 - 0.6], [0.4 * SQRT6 - 0.6, 1.6 - 0.4 * SQRT6]])
ANCHOR_A_KL_INC = 0.25 * math.log(6.0) - 0.5 * math.log(0.96 + 0.44 * SQRT6)


def test_anchor_a_decimals_match_the_docstring():
    # Guards the hand arithmetic itself: the decimals quoted in the docstring against the closed forms.
    np.testing.assert_allclose(ANCHOR_A_CORRECTED, [[0.620204, 0.379796], [0.379796, 0.620204]], atol=5e-7)
    assert ANCHOR_A_KL_INC == pytest.approx(0.092010, abs=5e-7)
    np.testing.assert_allclose(ANCHOR_A_CORRECTED[0], [4 / (4 + SQRT6), SQRT6 / (4 + SQRT6)], atol=1e-15)
    np.testing.assert_allclose(ANCHOR_A_CORRECTED[1], [3 / (3 + 2 * SQRT6), 2 * SQRT6 / (3 + 2 * SQRT6)], atol=1e-15)


@pytest.mark.parametrize("settings, atol", [({}, 1e-6), (dict(tol=1e-14, max_iter=10000), 1e-11)])
def test_anchor_a_closed_form(settings, atol):
    res = project(**ANCHOR_A, **settings)
    assert isinstance(res, Projection) and res.converged
    assert res.u[1] / res.u[0] == pytest.approx(SQRT6, abs=atol * 10)
    assert res.u.sum() == pytest.approx(1.0, abs=1e-15)                      # rescaled to sum to one
    np.testing.assert_allclose(res.u, [1 / (1 + SQRT6), SQRT6 / (1 + SQRT6)], atol=atol)
    np.testing.assert_allclose(res.corrected, ANCHOR_A_CORRECTED, atol=atol)
    assert res.kl_inc == pytest.approx(ANCHOR_A_KL_INC, abs=atol)
    # log Z for the normalized u is the unnormalized value minus log(1 + sqrt6).
    np.testing.assert_allclose(
        res.log_z,
        [math.log(0.8 + 0.2 * SQRT6) - math.log(1 + SQRT6), math.log(0.6 + 0.4 * SQRT6) - math.log(1 + SQRT6)],
        atol=atol,
    )
    assert res.max_violation <= settings.get("tol", 1e-8)
    assert 0 < res.iterations <= settings.get("max_iter", 500)


def test_anchor_b_point_mass_returns_the_prior():
    b = [0.5, 0.3, 0.2]
    res = project(b, [1.0, 0.0], [[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]])
    assert res.converged
    np.testing.assert_allclose(res.corrected[0], b, atol=1e-12)                # the belief after the certain outcome
    np.testing.assert_allclose(res.u, np.array([2.5, 1.0, 0.4]) / 3.9, atol=1e-12)
    assert res.kl_inc == pytest.approx(0.5 * math.log(2.5) + 0.2 * math.log(0.4), abs=1e-12)
    assert res.kl_inc == pytest.approx(0.274887, abs=5e-7)
    np.testing.assert_allclose(res.corrected[1], np.array([1.75, 0.2, 0.04]) / 1.99, atol=1e-12)


def test_anchor_c_coherent_pair_is_unchanged():
    Q = [[0.25, 0.5, 0.25], [0.5, 1 / 6, 1 / 3]]
    res = project([0.4, 0.3, 0.3], [0.4, 0.6], Q)
    assert res.converged and res.iterations == 0
    np.testing.assert_allclose(res.u, [1 / 3, 1 / 3, 1 / 3], atol=1e-15)        # constant class weights
    np.testing.assert_allclose(res.corrected, Q, atol=1e-15)
    assert abs(res.kl_inc) < 1e-15


@pytest.mark.parametrize("weights", [[0.5, 0.5], [0.9, 0.1], [0.123, 0.877]])
def test_anchor_d_pure_drift(weights):
    res = project([0.9, 0.1], weights, [[0.6, 0.4], [0.6, 0.4]])
    assert res.converged
    np.testing.assert_allclose(res.corrected, [[0.9, 0.1], [0.9, 0.1]], atol=1e-12)
    np.testing.assert_allclose(res.u, np.array([1.5, 0.25]) / 1.75, atol=1e-12)
    assert res.kl_inc == pytest.approx(0.9 * math.log(1.5) + 0.1 * math.log(0.25), abs=1e-12)
    assert res.kl_inc == pytest.approx(0.226289, abs=5e-7)


# ------------------------------------------------------------------------------------------------
# Structure of the result, on random cases
# ------------------------------------------------------------------------------------------------
def test_result_has_the_projection_form_and_records_the_floored_inputs():
    for _, b, w, Q in _random_cases(1, 60):
        res = project(b, w, Q)
        bf, Qf = _floor(b), _floor(Q)
        np.testing.assert_allclose(res.prior, bf, rtol=0, atol=1e-18)           # the floor is part of the method
        np.testing.assert_allclose(res.posteriors, Qf, rtol=0, atol=1e-18)
        np.testing.assert_allclose(res.weights, w, rtol=0, atol=1e-15)
        z = Qf @ res.u
        np.testing.assert_allclose(res.log_z, np.log(z), atol=1e-13)
        np.testing.assert_allclose(res.corrected, Qf * res.u / z[:, None], atol=1e-14)   # q' = u q / Z
        assert res.max_violation == pytest.approx(np.abs(w @ res.corrected - bf).max(), abs=1e-15)
        assert res.converged == (res.max_violation <= 1e-8)
        assert res.kl_inc == pytest.approx(bf @ np.log(res.u) - w @ np.log(z), abs=1e-12)
        assert res.u.sum() == pytest.approx(1.0, abs=1e-12) and np.all(res.u > 0)
        assert res.iterations <= 500


def test_kl_inc_is_the_divergence_of_the_projected_joint_and_is_dually_optimal():
    worst_div, min_kl, worst_dual, n_converged = 0.0, np.inf, -np.inf, 0
    for rng, b, w, Q in _random_cases(20261113, 600):
        res = project(b, w, Q, **TIGHT)
        if not res.converged:
            continue
        n_converged += 1
        bf, Qf = _floor(b), _floor(Q)
        J, Jp = w[:, None] * Qf, w[:, None] * res.corrected
        worst_div = max(worst_div, abs(res.kl_inc - float((Jp * np.log(Jp / J)).sum())))   # KL(J' || J)
        min_kl = min(min_kl, res.kl_inc)
        u2 = rng.uniform(0.1, 10.0, size=bf.size)                                           # any other class weights
        f2 = float(bf @ np.log(u2) - w @ np.log(Qf @ u2))
        worst_dual = max(worst_dual, f2 - res.kl_inc)
    assert n_converged >= 594                      # at most 1% excluded, as in cee_check.py
    assert worst_div < 1e-9
    assert min_kl > -1e-9                          # KL_inc >= 0 once the marginal constraint holds
    assert worst_dual < 1e-9                       # F(random u) <= F(projected u) = KL_inc


def test_default_tolerances_on_random_cases():
    n_failed = 0
    for _, b, w, Q in _random_cases(5, 300):
        res = project(b, w, Q)
        if not res.converged:
            n_failed += 1
            continue
        J, Jp = w[:, None] * _floor(Q), w[:, None] * res.corrected
        assert abs(res.kl_inc - float((Jp * np.log(Jp / J)).sum())) < 1e-6
        assert res.kl_inc > -1e-7
    assert n_failed <= 3


def test_point_mass_predictive_returns_the_prior_on_random_cases():
    for rng, b, w, Q in _random_cases(11, 100):
        k = int(rng.integers(len(w)))
        point = np.zeros(len(w))
        point[k] = 1.0
        res = project(b, point, Q)
        assert res.converged
        np.testing.assert_allclose(res.corrected[k], _floor(b), atol=1e-9)
        assert res.kl_inc == pytest.approx(_kl(_floor(b), _floor(Q)[k]), abs=1e-9)   # KL(b || q) for a certain outcome


def test_bayes_consistent_pair_is_unchanged_on_random_cases():
    rng = np.random.default_rng(3)
    for _ in range(100):
        C, K = int(rng.integers(2, 7)), int(rng.integers(2, 10))
        J = rng.dirichlet(np.ones(K * C)).reshape(K, C) + 1e-3      # every cell far above the floor
        J = J / J.sum()
        w, b = J.sum(1), J.sum(0)
        Q = J / w[:, None]
        res = project(b, w, Q)
        assert res.converged and res.iterations == 0
        np.testing.assert_allclose(res.corrected, Q, atol=1e-12)
        np.testing.assert_allclose(res.u, np.full(C, 1.0 / C), atol=1e-12)
        assert abs(res.kl_inc) < 1e-12


def test_planted_class_reweighting_is_recovered():
    # Take a coherent pair (w, Q0) with class marginal b and divide every posterior by class weights
    # v. Then u = v restores Q0, whose imagined mean is b; by uniqueness the projection must return
    # u proportional to v and the original posteriors.
    rng = np.random.default_rng(17)
    for _ in range(60):
        C, K = int(rng.integers(2, 7)), int(rng.integers(2, 10))
        J = np.exp(rng.normal(scale=0.7, size=(K, C)))
        J = J / J.sum()
        w, b = J.sum(1), J.sum(0)
        Q0 = J / w[:, None]
        v = np.exp(rng.normal(scale=1.0, size=C))
        Q = Q0 / v
        Q = Q / Q.sum(1, keepdims=True)
        assert Q.min() > 1e-5                                         # the floor is inactive in this fixture
        res = project(b, w, Q, **TIGHT)
        assert res.converged
        np.testing.assert_allclose(res.u, v / v.sum(), rtol=0, atol=1e-10)
        np.testing.assert_allclose(res.corrected, Q0, rtol=0, atol=1e-10)
        z = Q @ (v / v.sum())
        assert res.kl_inc == pytest.approx(b @ np.log(v / v.sum()) - w @ np.log(z), abs=1e-10)


# ------------------------------------------------------------------------------------------------
# Floors, non-convergence, the real result, and bad input
# ------------------------------------------------------------------------------------------------
def test_zeros_in_reader_vectors_are_floored_not_fatal():
    prior = [0.7, 0.3, 0.0]
    posteriors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.2, 0.3, 0.5]]
    weights = [0.3, 0.3, 0.4]
    res = project(prior, weights, posteriors, **TIGHT)
    np.testing.assert_allclose(res.prior, _floor(prior), rtol=0, atol=1e-18)
    np.testing.assert_allclose(res.posteriors, _floor(posteriors), rtol=0, atol=1e-18)
    assert res.converged and np.all(np.isfinite(res.u)) and np.all(np.isfinite(res.log_z)) and np.isfinite(res.kl_inc)
    np.testing.assert_allclose(np.asarray(weights) @ res.corrected, _floor(prior), atol=1e-12)   # conservation holds
    assert res.kl_inc > 0


def test_non_convergence_is_reported_and_never_raised():
    res = project(**ANCHOR_A, max_iter=1)
    assert res.converged is False and res.iterations == 1
    assert res.max_violation > 1e-8
    assert res.max_violation == pytest.approx(np.abs(np.array([0.5, 0.5]) @ res.corrected - 0.5).max(), abs=1e-15)
    assert np.all(np.isfinite(res.corrected)) and np.isfinite(res.kl_inc)

    untouched = project(**ANCHOR_A, max_iter=0)             # no update allowed: the raw pair comes back
    assert untouched.converged is False and untouched.iterations == 0
    np.testing.assert_allclose(untouched.u, [0.5, 0.5], atol=1e-15)
    np.testing.assert_allclose(untouched.corrected, ANCHOR_A["posteriors"], atol=1e-15)
    assert untouched.max_violation == pytest.approx(0.2, abs=1e-12)      # the raw drift (0.2, -0.2)

    coherent = project([0.4, 0.3, 0.3], [0.4, 0.6], [[0.25, 0.5, 0.25], [0.5, 1 / 6, 1 / 3]], max_iter=0)
    assert coherent.converged is True                         # a coherent pair needs no iteration


def test_iteration_count_respects_max_iter_exactly():
    full = project(**ANCHOR_A)
    assert full.converged
    short = project(**ANCHOR_A, max_iter=full.iterations - 1)
    assert short.converged is False and short.iterations == full.iterations - 1
    exact = project(**ANCHOR_A, max_iter=full.iterations)
    assert exact.converged is True and exact.iterations == full.iterations


def test_apply_to_real_uses_the_same_class_weights():
    # Anchor A with the unnormalized weights (1, sqrt6): the first outcome arrives.
    corrected, log_z = apply_to_real([1.0, SQRT6], [0.8, 0.2])
    np.testing.assert_allclose(corrected, [4 / (4 + SQRT6), SQRT6 / (4 + SQRT6)], atol=1e-15)
    assert log_z == pytest.approx(math.log(0.8 + 0.2 * SQRT6), abs=1e-15)
    for _, b, w, Q in _random_cases(23, 40):
        res = project(b, w, Q)
        Qf = _floor(Q)
        for o in range(len(w)):
            corrected, log_z = apply_to_real(res.u, Q[o])
            z = float(Qf[o] @ res.u)
            np.testing.assert_allclose(corrected, res.u * Qf[o] / z, atol=1e-15)
            assert log_z == pytest.approx(math.log(z), abs=1e-13)
            np.testing.assert_allclose(corrected, res.corrected[o], atol=1e-13)   # a real result that was imagined
        outside = _floor(np.random.default_rng(0).dirichlet(np.ones(len(b))))     # a real result that was not
        corrected, log_z = apply_to_real(res.u, outside)
        assert corrected.sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize(
    "prior, weights, posteriors",
    [
        ([0.5, -0.1, 0.6], [0.5, 0.5], [[0.2, 0.3, 0.5], [0.3, 0.3, 0.4]]),      # negative probability
        ([0.5, 0.5], [0.5, 0.6], [[0.8, 0.2], [0.6, 0.4]]),                      # weights do not sum to one
        ([0.5, 0.4], [0.5, 0.5], [[0.8, 0.2], [0.6, 0.4]]),                      # prior does not sum to one
        ([0.5, 0.5], [0.5, 0.5], [[0.8, 0.2], [0.6, 0.3]]),                      # a posterior does not sum to one
        ([0.5, 0.5], [0.5, 0.5], [[0.8, 0.2]]),                                  # one posterior, two weights
        ([0.5, 0.5], [0.5, 0.5], [[0.2, 0.3, 0.5], [0.3, 0.3, 0.4]]),            # class counts differ
        ([0.5, float("nan")], [0.5, 0.5], [[0.8, 0.2], [0.6, 0.4]]),             # not finite
    ],
)
def test_malformed_input_is_rejected(prior, weights, posteriors):
    with pytest.raises(ValueError):
        project(prior, weights, posteriors)
