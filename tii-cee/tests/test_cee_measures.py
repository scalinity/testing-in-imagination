"""E0 fixtures, part 3: the audit quantities of SPEC 5.2 and the legacy entropy-change surrogate.

Hand anchor A, raw side (see test_cee_core.py). b = P_hat = (1/2, 1/2), q_1 = (0.8, 0.2),
q_2 = (0.6, 0.4).

    imagined mean   b_bar = (1/2)(0.8, 0.2) + (1/2)(0.6, 0.4) = (0.7, 0.3)
    drift           d = b_bar - b = (0.2, -0.2),  L1 norm 0.4,  largest component 0.2
    responsiveness  r = (1/2) KL((0.8, 0.2) || (0.7, 0.3)) + (1/2) KL((0.6, 0.4) || (0.7, 0.3))
                      = 0.4 log(8/7) + 0.1 log(2/3) + 0.3 log(6/7) + 0.2 log(4/3)
                      = 0.053413 - 0.040547 - 0.046245 + 0.057536 = 0.024157

Hand anchor E, the legacy score going negative. b = (0.2, 0.8), P_hat = (1/2, 1/2),
q_1 = (0.5, 0.5), q_2 = (0.1, 0.9); b_bar = (0.3, 0.7), drift (0.1, -0.1).

    H(b)   = -0.2 log 0.2 - 0.8 log 0.8 = 0.321888 + 0.178515 = 0.500402
    H(q_1) = log 2 = 0.693147,   H(q_2) = -0.1 log 0.1 - 0.9 log 0.9 = 0.230259 + 0.094824 = 0.325083
    entropy-change surrogate = 0.500402 - (0.693147 + 0.325083)/2 = -0.008713
    reaction term  = (1/2) KL(q_1 || b) + (1/2) KL(q_2 || b)
                   = (1/2)(0.5 log 2.5 + 0.5 log 0.625) + (1/2)(0.1 log 0.5 + 0.9 log 1.125)
                   = (1/2)(0.223144) + (1/2)(0.036690) = 0.129917
    drift term     = sum_y (b_bar - b)(y) log b(y) = 0.1 log 0.2 - 0.1 log 0.8 = 0.1 log 0.25 = -0.138629
    reaction + drift = -0.008713, the surrogate.

The reaction term is non-negative; the drift term has no sign. A negative value is therefore the
signature of an incoherent pair, not of an uninformative test (PROOF section 5).
"""
import math

import numpy as np
import pytest

from tiicee.cee import (
    EntropyChange,
    drift,
    entropy_change_surrogate,
    imagined_mean,
    reaction_kl,
    responsiveness,
)

A = dict(weights=[0.5, 0.5], posteriors=[[0.8, 0.2], [0.6, 0.4]])
A_RESPONSIVENESS = 0.4 * math.log(8 / 7) + 0.1 * math.log(2 / 3) + 0.3 * math.log(6 / 7) + 0.2 * math.log(4 / 3)
E = dict(prior=[0.2, 0.8], weights=[0.5, 0.5], posteriors=[[0.5, 0.5], [0.1, 0.9]])


def _floor(p):
    p = np.maximum(np.asarray(p, dtype=float), 1e-6)
    return p / p.sum(-1, keepdims=True)


def _kl(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    m = p > 0
    return float((p[m] * np.log(p[m] / q[m])).sum())


def _entropy(p):
    p = np.asarray(p, dtype=float)
    m = p > 0
    return float(-(p[m] * np.log(p[m])).sum())


def _random_cases(seed, n):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        C = int(rng.integers(2, 6))
        K = int(rng.integers(2, 8))
        yield rng.dirichlet(np.ones(C)), _floor(rng.dirichlet(np.ones(K))), rng.dirichlet(np.ones(C), size=K)


def test_anchor_a_raw_quantities_by_hand():
    np.testing.assert_allclose(imagined_mean(**A), [0.7, 0.3], atol=1e-15)
    d = drift([0.5, 0.5], **A)
    np.testing.assert_allclose(d, [0.2, -0.2], atol=1e-15)
    assert np.abs(d).sum() == pytest.approx(0.4, abs=1e-15) and np.abs(d).max() == pytest.approx(0.2, abs=1e-15)
    assert responsiveness(**A) == pytest.approx(A_RESPONSIVENESS, abs=1e-15)
    assert A_RESPONSIVENESS == pytest.approx(0.024157, abs=5e-7)


def test_a_reader_that_never_moves_has_zero_responsiveness_and_drift_only_from_its_prior():
    same = [[0.6, 0.4], [0.6, 0.4], [0.6, 0.4]]
    assert responsiveness([0.2, 0.3, 0.5], same) == pytest.approx(0.0, abs=1e-15)
    np.testing.assert_allclose(drift([0.9, 0.1], [0.2, 0.3, 0.5], same), [-0.3, 0.3], atol=1e-15)


def test_coherent_pair_has_no_drift():
    J = np.array([[0.1, 0.2, 0.1], [0.3, 0.1, 0.2]])
    w, b = J.sum(1), J.sum(0)
    np.testing.assert_allclose(drift(b, w, J / w[:, None]), np.zeros(3), atol=1e-15)


def test_reaction_kl_by_hand():
    # KL(q || b) for one result. Both model-free probes of SPEC 5.2 use it: the reaction to
    # UNAVAILABLE where missingness is uninformative, and the reaction to an exact re-exposure.
    assert reaction_kl([0.5, 0.5], [0.8, 0.2]) == pytest.approx(0.8 * math.log(1.6) + 0.2 * math.log(0.4), abs=1e-15)
    assert reaction_kl([0.5, 0.5], [0.8, 0.2]) == pytest.approx(0.192745, abs=5e-7)
    assert reaction_kl([0.3, 0.7], [0.3, 0.7]) == 0.0                      # a reader that ignores the probe
    assert reaction_kl([0.2, 0.8], [0.5, 0.5]) == pytest.approx(0.5 * math.log(2.5) + 0.5 * math.log(0.625), abs=1e-15)
    # Floored like every reader vector: a zero does not produce an infinite reaction.
    assert reaction_kl([1.0, 0.0], [0.5, 0.5]) == pytest.approx(_kl([0.5, 0.5], _floor([1.0, 0.0])), abs=1e-12)


def test_entropy_change_surrogate_by_hand():
    got = entropy_change_surrogate(**E)
    assert isinstance(got, EntropyChange)
    h_b = -0.2 * math.log(0.2) - 0.8 * math.log(0.8)
    h_q2 = -0.1 * math.log(0.1) - 0.9 * math.log(0.9)
    reaction = 0.5 * (0.5 * math.log(2.5) + 0.5 * math.log(0.625)) + 0.5 * (0.1 * math.log(0.5) + 0.9 * math.log(1.125))
    assert got.score == pytest.approx(h_b - 0.5 * (math.log(2.0) + h_q2), abs=1e-15)
    assert got.reaction_term == pytest.approx(reaction, abs=1e-15)
    assert got.drift_term == pytest.approx(0.1 * math.log(0.25), abs=1e-15)
    assert got.score == pytest.approx(got.reaction_term + got.drift_term, abs=1e-15)
    # The decimals of the docstring.
    assert got.score == pytest.approx(-0.008713, abs=5e-7)
    assert got.reaction_term == pytest.approx(0.129917, abs=5e-7)
    assert got.drift_term == pytest.approx(-0.138629, abs=5e-7)
    assert got.score < 0 <= got.reaction_term


def test_entropy_change_decomposition_on_random_cases():
    n_negative = 0
    for b, w, Q in _random_cases(20261113, 200):
        bf, Qf = _floor(b), _floor(Q)
        got = entropy_change_surrogate(b, w, Q)
        score = _entropy(bf) - sum(w[o] * _entropy(Qf[o]) for o in range(len(w)))
        reaction = sum(w[o] * _kl(Qf[o], bf) for o in range(len(w)))
        drift_term = float((w @ Qf - bf) @ np.log(bf))
        assert got.score == pytest.approx(score, abs=1e-12)
        assert got.reaction_term == pytest.approx(reaction, abs=1e-12)
        assert got.drift_term == pytest.approx(drift_term, abs=1e-12)
        assert got.score == pytest.approx(got.reaction_term + got.drift_term, abs=1e-12)
        assert got.reaction_term >= 0
        n_negative += got.score < -1e-12
        np.testing.assert_allclose(drift(b, w, Q), w @ Qf - bf, atol=1e-15)
        np.testing.assert_allclose(imagined_mean(w, Q), w @ Qf, atol=1e-15)
        assert responsiveness(w, Q) == pytest.approx(sum(w[o] * _kl(Qf[o], w @ Qf) for o in range(len(w))), abs=1e-12)
    assert n_negative > 0                     # the raw pair does produce negative scores


def test_for_a_coherent_pair_the_drift_term_vanishes():
    rng = np.random.default_rng(2)
    for _ in range(50):
        K, C = int(rng.integers(2, 8)), int(rng.integers(2, 6))
        J = rng.dirichlet(np.ones(K * C)).reshape(K, C) + 1e-3
        J = J / J.sum()
        w, b = J.sum(1), J.sum(0)
        got = entropy_change_surrogate(b, w, J / w[:, None])
        assert abs(got.drift_term) < 1e-13
        # Only here does the score coincide with the information the joint carries about the class.
        assert got.score == pytest.approx(float((J * np.log(J / np.outer(w, b))).sum()), abs=1e-12)
        assert got.score >= 0


def test_malformed_input_is_rejected():
    with pytest.raises(ValueError):
        responsiveness([0.5, 0.5], [[0.8, 0.2]])
    with pytest.raises(ValueError):
        drift([0.5, 0.5], [0.5, 0.5], [[0.2, 0.3, 0.5], [0.3, 0.3, 0.4]])
    with pytest.raises(ValueError):
        reaction_kl([0.5, 0.5], [0.2, 0.3, 0.5])
