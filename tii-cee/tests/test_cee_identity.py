"""E0 fixtures, part 2: the gain identity and the realized gain (SPEC 5.4; PROOF section 4).

Hand anchor A, continued (see test_cee_core.py). b = P_hat = (1/2, 1/2), q_1 = (0.8, 0.2),
q_2 = (0.6, 0.4); projected weights u = (1, sqrt6) up to scale, Z_1 = 0.8 + 0.2 sqrt6,
Z_2 = 0.6 + 0.4 sqrt6, KL_inc = (1/4) log 6 - (1/2) log(0.96 + 0.44 sqrt6) = 0.092010.

Per-case realized gain g(o, y) = log u(y) - log Z_o = log q'_o(y) - log q_o(y):

    g(1, 1) = -log Z_1                 = -0.254563        g(1, 2) = (1/2) log 6 - log Z_1 = 0.641317
    g(2, 1) = -log Z_2                 = -0.457296        g(2, 2) = (1/2) log 6 - log Z_2 = 0.438584

Truth 1, calibrated marginals. P_star(o, y) = [[0.4, 0.1], [0.1, 0.4]] has b_star = b and
P_star = P_hat, so the prior and world-model terms vanish and the gain is KL_inc:
    0.4 (-0.254563) + 0.1 (0.641317) + 0.1 (-0.457296) + 0.4 (0.438584) = 0.092010.

Truth 2, miscalibrated. P_star(o, y) = [[0.6, 0.1], [0.1, 0.2]] has b_star = P_star = (0.7, 0.3).
    prior term        = (0.7 - 0.5) log 1 + (0.3 - 0.5) (1/2) log 6          = -0.1 log 6   = -0.179176
    world-model term  = -[(0.7 - 0.5) log Z_1 + (0.3 - 0.5) log Z_2]
                      = 0.2 log(Z_2 / Z_1) = 0.2 log sqrt(1.5)                = 0.1 log 1.5  =  0.040547
      (Z_2 / Z_1 = (0.6 + 0.4 sqrt6) / (0.8 + 0.2 sqrt6) = sqrt6 / 2 = sqrt(1.5).)
    total             = KL_inc - 0.1 log 6 + 0.1 log 1.5 = KL_inc - 0.2 log 2 = -0.046619,
which is also 0.6 (-0.254563) + 0.1 (0.641317) + 0.1 (-0.457296) + 0.2 (0.438584).
The correction is expected to hurt under truth 2 although KL_inc is positive: the identity says why.
"""
import math

import numpy as np
import pytest

from tiicee.cee import GainTerms, gain_terms, project, realized_gain

SQRT6 = math.sqrt(6.0)
LOG_Z1 = math.log(0.8 + 0.2 * SQRT6)
LOG_Z2 = math.log(0.6 + 0.4 * SQRT6)
KL_INC_A = 0.25 * math.log(6.0) - 0.5 * math.log(0.96 + 0.44 * SQRT6)
B, P, Q = np.array([0.5, 0.5]), np.array([0.5, 0.5]), np.array([[0.8, 0.2], [0.6, 0.4]])
G = np.array([[-LOG_Z1, 0.5 * math.log(6.0) - LOG_Z1], [-LOG_Z2, 0.5 * math.log(6.0) - LOG_Z2]])   # g(o, y)
TIGHT = dict(tol=1e-13, max_iter=20000)


def _floor(p):
    p = np.maximum(np.asarray(p, dtype=float), 1e-6)
    return p / p.sum(-1, keepdims=True)


def _rake(J, row, col, tol=1e-13, max_iter=20000):
    """Scale a positive matrix to the given row and column sums (inline, independent of the library)."""
    for _ in range(max_iter):
        J = J * (row / J.sum(1))[:, None]
        J = J * (col / J.sum(0))[None, :]
        if np.abs(J.sum(1) - row).max() < tol:
            break
    return J


def _random_cases(seed, n):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        C = int(rng.integers(2, 7))
        K = int(rng.integers(2, 10))
        b = rng.dirichlet(np.ones(C))
        w = _floor(rng.dirichlet(np.ones(K)))
        Q = rng.dirichlet(np.ones(C) * rng.uniform(0.3, 3.0), size=K)
        truth = rng.dirichlet(np.ones(K * C)).reshape(K, C)          # arbitrary truth over (o, y)
        yield rng, b, w, Q, truth


def test_anchor_decimals_match_the_docstring():
    np.testing.assert_allclose(G, [[-0.254563, 0.641317], [-0.457296, 0.438584]], atol=5e-7)
    assert (0.6 + 0.4 * SQRT6) / (0.8 + 0.2 * SQRT6) == pytest.approx(math.sqrt(1.5), abs=1e-15)
    assert KL_INC_A - 0.2 * math.log(2.0) == pytest.approx(-0.046619, abs=5e-7)
    assert float((np.array([[0.4, 0.1], [0.1, 0.4]]) * G).sum()) == pytest.approx(KL_INC_A, abs=1e-15)
    assert float((np.array([[0.6, 0.1], [0.1, 0.2]]) * G).sum()) == pytest.approx(KL_INC_A - 0.2 * math.log(2.0), abs=1e-15)


def test_realized_gain_by_hand():
    res = project(B, P, Q, tol=1e-14, max_iter=10000)
    for o in range(2):
        for y in range(2):
            assert realized_gain(res.u, res.log_z[o], y) == pytest.approx(G[o, y], abs=1e-11)
    # The definition itself, on numbers that need no projection: log 4 - log 2 = log 2.
    assert realized_gain([1.0, 4.0], math.log(2.0), 1) == pytest.approx(math.log(2.0), abs=1e-15)


def test_calibrated_truth_by_hand():
    res = project(B, P, Q, tol=1e-14, max_iter=10000)
    terms = gain_terms(res.prior, res.weights, res.u, res.log_z, b_star=[0.5, 0.5], p_star=[0.5, 0.5])
    assert isinstance(terms, GainTerms)
    assert terms.incoherence == pytest.approx(KL_INC_A, abs=1e-11)
    assert terms.prior_calibration == pytest.approx(0.0, abs=1e-15)
    assert terms.world_model == pytest.approx(0.0, abs=1e-15)
    assert terms.total == pytest.approx(KL_INC_A, abs=1e-11)


def test_miscalibrated_truth_by_hand():
    res = project(B, P, Q, tol=1e-14, max_iter=10000)
    terms = gain_terms(res.prior, res.weights, res.u, res.log_z, b_star=[0.7, 0.3], p_star=[0.7, 0.3])
    assert terms.incoherence == pytest.approx(KL_INC_A, abs=1e-11)
    assert terms.prior_calibration == pytest.approx(-0.1 * math.log(6.0), abs=1e-11)
    assert terms.world_model == pytest.approx(0.1 * math.log(1.5), abs=1e-11)
    assert terms.total == pytest.approx(KL_INC_A - 0.2 * math.log(2.0), abs=1e-11)
    assert terms.total < 0 < terms.incoherence          # a positive KL_inc does not promise a gain


def test_identity_holds_for_arbitrary_truth_with_projected_weights():
    worst = 0.0
    for _, b, w, Q, truth in _random_cases(20261113, 1000):
        res = project(b, w, Q)
        Qf = _floor(Q)
        direct = float((truth * (np.log(res.corrected) - np.log(Qf))).sum())      # E_star[log q' - log q]
        terms = gain_terms(res.prior, res.weights, res.u, res.log_z, truth.sum(0), truth.sum(1))
        worst = max(worst, abs(direct - terms.total))
        assert terms.total == pytest.approx(terms.incoherence + terms.prior_calibration + terms.world_model, abs=1e-14)
        assert terms.incoherence == pytest.approx(res.kl_inc, abs=1e-12)
    assert worst < 1e-9


def test_identity_is_algebraic_it_holds_for_unconverged_weights_too():
    worst = 0.0
    for _, b, w, Q, truth in _random_cases(8, 100):
        res = project(b, w, Q, max_iter=2)                 # deliberately stopped early
        direct = float((truth * (np.log(res.corrected) - np.log(_floor(Q)))).sum())
        terms = gain_terms(res.prior, res.weights, res.u, res.log_z, truth.sum(0), truth.sum(1))
        worst = max(worst, abs(direct - terms.total))
    assert worst < 1e-9


def test_identity_holds_for_arbitrary_positive_class_weights():
    worst = worst_f = 0.0
    for rng, b, w, Q, truth in _random_cases(20261114, 1000):
        bf, Qf = _floor(b), _floor(Q)
        u = rng.uniform(0.1, 10.0, size=bf.size)           # any positive weights, nothing to do with the projection
        z = Qf @ u
        reweighted = Qf * u / z[:, None]
        direct = float((truth * (np.log(reweighted) - np.log(Qf))).sum())
        terms = gain_terms(bf, w, u, np.log(z), truth.sum(0), truth.sum(1))
        worst = max(worst, abs(direct - terms.total))
        worst_f = max(worst_f, abs(terms.incoherence - float(bf @ np.log(u) - w @ np.log(z))))
        assert terms.prior_calibration == pytest.approx(float((truth.sum(0) - bf) @ np.log(u)), abs=1e-12)
        assert terms.world_model == pytest.approx(-float((truth.sum(1) - w) @ np.log(z)), abs=1e-12)
    assert worst < 1e-9 and worst_f < 1e-12


def test_calibrated_marginals_give_gain_equal_to_kl_inc():
    worst_gain = worst_terms = 0.0
    n = 0
    for rng, b, w, Q, truth in _random_cases(31, 60):
        res = project(b, w, Q, **TIGHT)
        if not res.converged:
            continue
        n += 1
        calibrated = _rake(truth, res.weights, res.prior)                  # truth with marginals (P_hat, b)
        assert np.abs(calibrated.sum(0) - res.prior).max() < 1e-11 and np.abs(calibrated.sum(1) - res.weights).max() < 1e-11
        direct = float((calibrated * (np.log(res.corrected) - np.log(res.posteriors))).sum())
        terms = gain_terms(res.prior, res.weights, res.u, res.log_z, calibrated.sum(0), calibrated.sum(1))
        worst_gain = max(worst_gain, abs(direct - res.kl_inc))
        worst_terms = max(worst_terms, abs(terms.prior_calibration), abs(terms.world_model), abs(terms.total - res.kl_inc))
    assert n >= 58
    assert worst_gain < 1e-9 and worst_terms < 1e-9


def test_mean_realized_gain_under_the_truth_is_the_identity_total():
    # Consequence 3 of SPEC 5.4, by enumeration over every (outcome, label) cell.
    for _, b, w, Q, truth in _random_cases(40, 60):
        res = project(b, w, Q)
        mean_g = sum(
            truth[o, y] * realized_gain(res.u, res.log_z[o], y) for o in range(truth.shape[0]) for y in range(truth.shape[1])
        )
        terms = gain_terms(res.prior, res.weights, res.u, res.log_z, truth.sum(0), truth.sum(1))
        assert mean_g == pytest.approx(terms.total, abs=1e-10)


def test_both_are_invariant_to_rescaling_the_class_weights():
    for _, b, w, Q, truth in _random_cases(52, 40):
        res = project(b, w, Q)
        base = gain_terms(res.prior, res.weights, res.u, res.log_z, truth.sum(0), truth.sum(1))
        for k in (1e-3, 0.37, 41.0):
            u = k * res.u
            log_z = np.log(res.posteriors @ u)                              # Z is recomputed for the rescaled weights
            scaled = gain_terms(res.prior, res.weights, u, log_z, truth.sum(0), truth.sum(1))
            for name in ("incoherence", "prior_calibration", "world_model", "total"):
                assert getattr(scaled, name) == pytest.approx(getattr(base, name), abs=1e-11)
            for o in range(len(w)):
                for y in range(len(b)):
                    assert realized_gain(u, log_z[o], y) == pytest.approx(realized_gain(res.u, res.log_z[o], y), abs=1e-11)


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        gain_terms([0.5, 0.5], [0.5, 0.5], [1.0, 2.0, 3.0], [0.0, 0.0], [0.5, 0.5], [0.5, 0.5])
    with pytest.raises(ValueError):
        gain_terms([0.5, 0.5], [0.5, 0.5], [1.0, 2.0], [0.0, 0.0, 0.0], [0.5, 0.5], [0.5, 0.5])
