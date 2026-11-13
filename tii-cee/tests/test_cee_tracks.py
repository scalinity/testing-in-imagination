"""E0 fixtures, part 6: the stateful and stateless tracks of SPEC 5.9.

Closed form for C = 2, K = 2. Write a_o = q_o(1) and t = u(1)/u(2). Then
q'_o(1) = t a_o / (t a_o + 1 - a_o), and the constraint P_1 q'_1(1) + P_2 q'_2(1) = b(1), cleared
of denominators, is the quadratic

    t^2 a_1 a_2 (1 - b_1)
      + t [P_1 a_1 (1 - a_2) + P_2 a_2 (1 - a_1) - b_1 (a_1 (1 - a_2) + a_2 (1 - a_1))]
      - b_1 (1 - a_1)(1 - a_2) = 0,

with one positive root.

Hand anchor T. The raw reader holds b_t = (0.2, 0.8) and imagines q_1 = (0.5, 0.5),
q_2 = (3/11, 8/11) under P_hat = (1/2, 1/2). The tracked coherent belief is c_t = (1/2, 1/2).

Stateful step. Increments rho_o = q_o / b_t: rho_1 = (2.5, 0.625), rho_2 = (15/11, 10/11).
Proposals m_o proportional to c_t rho_o: m_1 = (1.25, 0.3125)/1.5625 = (0.8, 0.2) and
m_2 = (15/22, 10/22)/(25/22) = (0.6, 0.4). These are the posteriors of hand anchor A
(test_cee_core.py) with prior (1/2, 1/2): the quadratic reduces to 0.24 t^2 - 0.04 = 0, so
u(2)/u(1) = sqrt6 and
    c_{t+1} = (1.6 - 0.4 sqrt6, 0.4 sqrt6 - 0.6) = (0.620204, 0.379796) if outcome 1 arrives,
    c_{t+1} = (0.4 sqrt6 - 0.6, 1.6 - 0.4 sqrt6) = (0.379796, 0.620204) if outcome 2 arrives.
Their P_hat-average is (1/2, 1/2) = c_t: the track is a martingale under the world model.

Stateless step, same reader. Project (q_1, q_2) against b_t: a_1 = 1/2, a_2 = 3/11, b_1 = 0.2.
    t^2 (3/22)(0.8) + t [2/11 + 0.75/11 - 0.2 (5.5/11)] - 0.2 (1/2)(8/11) = 0
    (2.4/22) t^2 + 0.15 t - 0.8/11 = 0      =>      2.4 t^2 + 3.3 t - 1.6 = 0
    t = (-3.3 + sqrt(10.89 + 15.36)) / 4.8 = (-3.3 + sqrt(26.25)) / 4.8 = 0.379891
    c_{t+1}(1) = 0.5 t / (0.5 t + 0.5) = t / (t + 1) = 0.275305                 if outcome 1 arrives,
    c_{t+1}(1) = (3/11) t / ((3/11) t + 8/11) = 3t / (3t + 8) = 0.124695        if outcome 2 arrives,
and (0.275305 + 0.124695)/2 = 0.2 = b_t(1).
"""
import math

import numpy as np
import pytest

from tiicee.cee import Projection, ScaleResult, Step, project, scale, stateful_step, stateless_step

SQRT6 = math.sqrt(6.0)
TIGHT = dict(tol=1e-13, max_iter=20000)
T_B = [0.2, 0.8]
T_Q = [[0.5, 0.5], [3 / 11, 8 / 11]]
T_P = [0.5, 0.5]
T_C = [0.5, 0.5]
T_ROOT = (-3.3 + math.sqrt(26.25)) / 4.8


def _floor(p):
    p = np.maximum(np.asarray(p, dtype=float), 1e-6)
    return p / p.sum(-1, keepdims=True)


def _random_step(rng):
    """Tracked belief c_t (strictly positive), and raw, unfloored reader vectors b_t and q_o."""
    C, K = int(rng.integers(2, 6)), int(rng.integers(2, 8))
    c = _floor(rng.dirichlet(np.ones(C)))
    w = _floor(rng.dirichlet(np.ones(K)))
    return c, rng.dirichlet(np.ones(C)), w, rng.dirichlet(np.ones(C), size=K)


def _coherent_pair(rng, spread=0.5):
    C, K = int(rng.integers(2, 6)), int(rng.integers(2, 8))
    J = np.exp(rng.normal(scale=spread, size=(K, C)))
    J = J / J.sum()
    P, b = J.sum(1), J.sum(0)
    return b, P, J / P[:, None], J


def test_anchor_t_decimals_match_the_docstring():
    assert T_ROOT == pytest.approx(0.379891, abs=5e-7)
    assert 2.4 * T_ROOT**2 + 3.3 * T_ROOT - 1.6 == pytest.approx(0.0, abs=1e-14)
    assert T_ROOT / (T_ROOT + 1) == pytest.approx(0.275305, abs=5e-7)
    assert 3 * T_ROOT / (3 * T_ROOT + 8) == pytest.approx(0.124695, abs=5e-7)
    assert 0.5 * (T_ROOT / (T_ROOT + 1) + 3 * T_ROOT / (3 * T_ROOT + 8)) == pytest.approx(0.2, abs=1e-15)
    proposals = np.array(T_C) * np.array(T_Q) / np.array(T_B)
    np.testing.assert_allclose(proposals / proposals.sum(1, keepdims=True), [[0.8, 0.2], [0.6, 0.4]], atol=1e-15)


def test_anchor_t_stateful_step_by_hand():
    first = stateful_step(T_C, T_B, T_P, T_Q, 0, **TIGHT)
    second = stateful_step(T_C, T_B, T_P, T_Q, 1, **TIGHT)
    assert isinstance(first, Step) and isinstance(first.projection, Projection)
    assert first.fallback is False and first.scale is None
    np.testing.assert_allclose(first.belief, [1.6 - 0.4 * SQRT6, 0.4 * SQRT6 - 0.6], atol=1e-11)
    np.testing.assert_allclose(second.belief, [0.4 * SQRT6 - 0.6, 1.6 - 0.4 * SQRT6], atol=1e-11)
    np.testing.assert_allclose(0.5 * first.belief + 0.5 * second.belief, T_C, atol=1e-12)    # martingale under P_hat
    # The projection record is the one for the proposals against c_t.
    assert first.projection.u[1] / first.projection.u[0] == pytest.approx(SQRT6, abs=1e-10)
    np.testing.assert_allclose(first.projection.prior, T_C, atol=1e-15)
    np.testing.assert_allclose(first.projection.posteriors, [[0.8, 0.2], [0.6, 0.4]], atol=1e-14)
    assert first.projection.kl_inc == pytest.approx(0.25 * math.log(6.0) - 0.5 * math.log(0.96 + 0.44 * SQRT6), abs=1e-11)


def test_anchor_t_stateless_step_by_hand():
    first = stateless_step(T_B, T_P, T_Q, 0, **TIGHT)
    second = stateless_step(T_B, T_P, T_Q, 1, **TIGHT)
    assert first.fallback is False and first.scale is None
    assert first.projection.u[0] / first.projection.u[1] == pytest.approx(T_ROOT, abs=1e-10)
    np.testing.assert_allclose(first.belief, [T_ROOT / (T_ROOT + 1), 1 / (T_ROOT + 1)], atol=1e-11)
    np.testing.assert_allclose(second.belief, [3 * T_ROOT / (3 * T_ROOT + 8), 8 / (3 * T_ROOT + 8)], atol=1e-11)
    np.testing.assert_allclose(0.5 * first.belief + 0.5 * second.belief, T_B, atol=1e-12)    # conserved against b_t
    # The two tracks differ on the same reader.
    assert abs(first.belief[0] - stateful_step(T_C, T_B, T_P, T_Q, 0, **TIGHT).belief[0]) > 0.3


def test_stateless_step_is_the_projection_at_the_real_outcome():
    rng = np.random.default_rng(1)
    for _ in range(40):
        _, b, w, Q = _random_step(rng)
        res = project(b, w, Q)
        for o in range(len(w)):
            step = stateless_step(b, w, Q, o)
            expected = res.u * _floor(Q)[o]                             # c_{t+1} proportional to u_t x q_real
            np.testing.assert_allclose(step.belief, expected / expected.sum(), atol=1e-13)
            np.testing.assert_allclose(step.projection.u, res.u, atol=1e-15)


def test_stateful_step_formula_and_martingale():
    rng = np.random.default_rng(2)
    for _ in range(40):
        c, b, w, Q = _random_step(rng)
        steps = [stateful_step(c, b, w, Q, o, **TIGHT) for o in range(len(w))]
        assert all(s.projection.converged and not s.fallback for s in steps)
        bf, Qf = _floor(b), _floor(Q)
        for o, step in enumerate(steps):
            expected = step.projection.u * c * (Qf[o] / bf)             # c_{t+1} proportional to u_t x c_t x rho_real
            np.testing.assert_allclose(step.belief, expected / expected.sum(), atol=1e-13)
            proposals = c * Qf / bf
            np.testing.assert_allclose(step.projection.posteriors, proposals / proposals.sum(1, keepdims=True), atol=1e-13)
            np.testing.assert_allclose(step.projection.prior, c, atol=1e-15)   # projected onto mean c_t
        np.testing.assert_allclose(sum(w[o] * steps[o].belief for o in range(len(w))), c, atol=1e-12)


def test_stateful_equals_stateless_while_the_track_agrees_with_the_reader():
    rng = np.random.default_rng(3)
    for _ in range(30):
        _, b, w, Q = _random_step(rng)
        for o in range(len(w)):
            # c_t is the reader's own (floored) belief, so the proposals are the reader's posteriors.
            np.testing.assert_allclose(stateful_step(_floor(b), b, w, Q, o).belief, stateless_step(b, w, Q, o).belief, atol=1e-12)


def test_the_tracked_belief_is_not_floored():
    # c_t has a class at 1e-9, far below the floor. The step must conserve c_t itself; flooring it
    # would move that class to 1e-6 and break the martingale by about 1e-6.
    c = np.array([0.6, 0.4 - 1e-9, 1e-9])
    b = [0.3, 0.3, 0.4]
    w = [0.25, 0.75]
    Q = [[0.5, 0.2, 0.3], [0.2, 0.3, 0.5]]
    steps = [stateful_step(c, b, w, Q, o, **TIGHT) for o in range(2)]
    assert all(s.projection.converged for s in steps)
    np.testing.assert_allclose(0.25 * steps[0].belief + 0.75 * steps[1].belief, c, rtol=0, atol=1e-12)
    assert steps[0].belief[2] < 1e-8


def test_non_convergence_falls_back_to_the_raw_reader():
    # SPEC 5.3: non-convergence is recorded and the raw reader is used.
    stateless = stateless_step(T_B, T_P, T_Q, 1, max_iter=0)
    assert stateless.projection.converged is False and stateless.fallback is True
    np.testing.assert_allclose(stateless.belief, T_Q[1], atol=1e-15)                     # the raw posterior
    stateful = stateful_step(T_C, T_B, T_P, T_Q, 1, max_iter=0)
    assert stateful.projection.converged is False and stateful.fallback is True
    np.testing.assert_allclose(stateful.belief, [0.6, 0.4], atol=1e-15)                  # c_t x rho_real, uncorrected


def test_scale_stage_on_the_stateless_track():
    rng = np.random.default_rng(4)
    for _ in range(10):
        b, P, q0, J = _coherent_pair(rng)
        q = b * (q0 / b) ** 2.0                                          # an over-reactive reader, gamma = 2
        q = q / q.sum(1, keepdims=True)
        budget = float((J * np.log(J / np.outer(P, b))).sum())
        reference = scale(b, P, q, budget, "two-sided")
        for o in range(len(P)):
            step = stateless_step(b, P, q, o, scale=(budget, "two-sided"))
            assert isinstance(step.scale, ScaleResult) and step.fallback is False
            assert step.scale.path == "solved" and abs(step.scale.tau_b / 2.0 - 1.0) <= 0.01
            np.testing.assert_allclose(step.belief, reference.repaired[o], atol=1e-13)
            np.testing.assert_allclose(step.belief, q0[o], atol=0.02)    # the coherent reader, to the 1% tolerance
            assert step.projection is step.scale.projection


def test_scale_stage_on_the_stateful_track():
    # gamma = 1.7, off the search grid: at a grid value r' equals the budget up to rounding and two
    # computations that differ in the last bit may bisect from opposite sides, agreeing only to 1%.
    rng = np.random.default_rng(5)
    for _ in range(10):
        b, P, q0, J = _coherent_pair(rng)
        q = b * (q0 / b) ** 1.7
        q = q / q.sum(1, keepdims=True)
        budget = float((J * np.log(J / np.outer(P, b))).sum())
        # While the track agrees with the reader the two tracks coincide, scale stage included.
        for o in range(len(P)):
            np.testing.assert_allclose(
                stateful_step(b, b, P, q, o, scale=(budget, "two-sided")).belief,
                stateless_step(b, P, q, o, scale=(budget, "two-sided")).belief,
                atol=1e-12,
            )
        # Otherwise the increments are tempered and the proposals projected onto c_t.
        c = _floor(rng.dirichlet(np.ones(len(b)) * 5.0))
        proposals = c * q / b
        proposals = proposals / proposals.sum(1, keepdims=True)
        reference = scale(c, P, proposals, budget, "shrink-only")
        steps = [stateful_step(c, b, P, q, o, scale=(budget, "shrink-only")) for o in range(len(P))]
        for o, step in enumerate(steps):
            assert step.scale.tau_b == pytest.approx(reference.tau_b, rel=1e-12) and step.scale.path == reference.path
            np.testing.assert_allclose(step.belief, reference.repaired[o], atol=1e-12)
            tempered = c * (q[o] / b) ** (1.0 / step.scale.tau_b)        # c_t x rho_real^(1/tau_B), then u_t
            expected = step.scale.u * tempered
            np.testing.assert_allclose(step.belief, expected / expected.sum(), atol=1e-12)
        np.testing.assert_allclose(sum(P[o] * steps[o].belief for o in range(len(P))), c, atol=1e-7)   # still a martingale


def test_scale_stage_falls_back_when_its_projection_does_not_converge():
    step = stateless_step(T_B, T_P, T_Q, 0, scale=(0.01, "two-sided"), max_iter=0)
    assert step.fallback is True and step.scale is not None and step.scale.converged is False
    np.testing.assert_allclose(step.belief, T_Q[0], atol=1e-15)


def test_bad_input_is_rejected():
    with pytest.raises(ValueError):
        stateful_step([0.5, 0.5, 0.0], [0.3, 0.3, 0.4], [0.5, 0.5], [[0.5, 0.2, 0.3], [0.2, 0.3, 0.5]], 0)   # c_t not positive
    with pytest.raises(ValueError):
        stateful_step([0.5, 0.5], [0.3, 0.3, 0.4], [0.5, 0.5], [[0.5, 0.2, 0.3], [0.2, 0.3, 0.5]], 0)        # class counts differ
    with pytest.raises(IndexError):
        stateless_step(T_B, T_P, T_Q, 2)
    with pytest.raises(ValueError):
        stateless_step(T_B, T_P, T_Q, 0, scale=(0.01, "sideways"))
