"""E0 fixtures, part 7: the multi-step statements of SPEC 5.12, by exact enumeration.

Statement (iii), for any belief process with implied joint M(path, y) = P_hat(path) x bel(y; path)
and implied class marginal m_0(y) = sum_path M(path, y):

    E_O[KL(P_star(Y | O) || bel)] = KL(b_star || m_0) + E_Y[KL(P_star(O | Y) || M(O | Y))] - KL(P_star(O) || P_hat(O)).

Hand anchor M (one step, two outcomes, two classes). Truth P_star(o, y) = [[0.4, 0.1], [0.1, 0.4]]:
P_star(o) = b_star = (1/2, 1/2), P_star(Y | o) = (0.8, 0.2) and (0.2, 0.8), likewise P_star(O | y).
Write I = log 2 - H(0.8, 0.2) = 0.192745 for the information the outcome carries.

  M1. Belief = true posterior, P_hat = P_star. Every term is zero and m_0 = (1/2, 1/2).

  M2. Belief never updates: (1/2, 1/2) at both leaves, P_hat = (1/2, 1/2). M = 1/4 in every cell.
      left side   = sum_o (1/2) KL(P_star(Y | o) || (1/2, 1/2)) = I
      prior term  = KL((1/2, 1/2) || (1/2, 1/2)) = 0
      likelihood  = sum_y (1/2) KL(P_star(O | y) || (1/2, 1/2)) = I        (M(O | y) = (1/2, 1/2))
      world model = 0.                                                      I = 0 + I - 0.

  M3. Belief never updates, wrong world model P_hat = (0.7, 0.3). M = [[0.35, 0.35], [0.15, 0.15]],
      m_0 = (1/2, 1/2), M(O | y) = (0.7, 0.3) for both classes.
      left side   = I = 0.192745 (the belief is the same as in M2)
      prior term  = 0
      likelihood  = (1/2)[0.8 log(0.8/0.7) + 0.2 log(0.2/0.3)] + (1/2)[0.2 log(0.2/0.7) + 0.8 log(0.8/0.3)]
                  = (1/2)(0.025732) + (1/2)(0.534111) = 0.279921
      world model = KL((1/2, 1/2) || (0.7, 0.3)) = (1/2) log(5/7) + (1/2) log(5/3) = 0.087177
      0 + 0.279921 - 0.087177 = 0.192745. The world-model error is subtracted: part of what looks
      like likelihood error is the world model's own, and the identity nets it out.

The random fixture is T = 3 steps, K = 3 outcomes, C = 3 classes: an arbitrary truth over
(y, o_1, o_2, o_3), a history-dependent world model, and a reader with arbitrary history-dependent
increments, as in cee_check.py part 3. The corrected beliefs come from stateful_step; every
reference quantity is enumerated inline.
"""
import itertools
import math

import numpy as np
import pytest

from tiicee.cee import MultiStepAccounting, multistep_accounting, stateful_step

T, K, C = 3, 3, 3
PATHS = list(itertools.product(range(K), repeat=T))
PREFIXES = [h for t in range(T) for h in itertools.product(range(K), repeat=t)]
TIGHT = dict(tol=1e-13, max_iter=20000)
INFO = math.log(2.0) + 0.8 * math.log(0.8) + 0.2 * math.log(0.2)
M_TRUTH = np.array([[0.4, 0.1], [0.1, 0.4]])
M_PATHS = [(0,), (1,)]


def _floor(p):
    p = np.maximum(np.asarray(p, dtype=float), 1e-6)
    return p / p.sum(-1, keepdims=True)


def _kl(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    m = p > 0
    return float((p[m] * np.log(p[m] / q[m])).sum())


def _problem(rng, how="real_index"):
    """Truth, world model, and the raw and corrected belief at every node of the outcome tree.

    how names the real outcome to the track by its index among the imagined outcomes, or by the
    reader's posterior after it (SPEC 5.9 needs only that posterior).
    """
    c0 = _floor(rng.dirichlet(np.ones(C)))
    truth = rng.dirichlet(np.ones(C * K**T)).reshape((C,) + (K,) * T)            # P_star(y, o_1, o_2, o_3)
    phat = {h: _floor(rng.dirichlet(np.ones(K))) for h in PREFIXES}              # world model after history h
    rho = {h: np.exp(rng.normal(scale=0.8, size=(K, C))) for h in PREFIXES}      # reader increments after history h
    raw, coh = {(): c0}, {(): c0}
    for h in PREFIXES:
        imagined = raw[h] * rho[h]
        imagined = imagined / imagined.sum(1, keepdims=True)                     # the reader's q_o at history h
        for o in range(K):
            raw[h + (o,)] = imagined[o]
            real = {"real_index": o} if how == "real_index" else {"real_posterior": imagined[o]}
            step = stateful_step(coh[h], raw[h], phat[h], imagined, **real, **TIGHT)
            assert step.projection.converged and not step.fallback
            coh[h + (o,)] = step.belief
    truth_paths = np.array([truth[(slice(None),) + p] for p in PATHS])            # [path, y]
    path_probs = np.array([np.prod([phat[p[:t]][p[t]] for t in range(T)]) for p in PATHS])
    return c0, truth_paths, phat, path_probs, raw, coh


def _enumerated_terms(truth_paths, path_probs, leaf):
    """The four quantities of statement (iii), straight from their definitions."""
    M = path_probs[:, None] * leaf
    implied = M.sum(0)
    p_star, b_star = truth_paths.sum(1), truth_paths.sum(0)
    lhs = sum(p_star[i] * _kl(truth_paths[i] / p_star[i], leaf[i]) for i in range(len(leaf)))
    likelihood = sum(b_star[y] * _kl(truth_paths[:, y] / b_star[y], M[:, y] / implied[y]) for y in range(leaf.shape[1]))
    return implied, lhs, _kl(b_star, implied), likelihood, _kl(p_star, path_probs)


# ------------------------------------------------------------------------------------------------
# Hand anchor M
# ------------------------------------------------------------------------------------------------
def test_anchor_m_decimals_match_the_docstring():
    assert INFO == pytest.approx(0.192745, abs=5e-7)
    likelihood = 0.5 * (0.8 * math.log(0.8 / 0.7) + 0.2 * math.log(0.2 / 0.3)) + 0.5 * (0.2 * math.log(0.2 / 0.7) + 0.8 * math.log(0.8 / 0.3))
    world_model = 0.5 * math.log(5 / 7) + 0.5 * math.log(5 / 3)
    assert likelihood == pytest.approx(0.279921, abs=5e-7) and world_model == pytest.approx(0.087177, abs=5e-7)
    assert likelihood - world_model == pytest.approx(INFO, abs=1e-15)


def test_anchor_m1_true_posterior_and_true_world_model():
    acc = multistep_accounting(M_PATHS, M_TRUTH, [0.5, 0.5], [[0.8, 0.2], [0.2, 0.8]], {(): [0.5, 0.5]})
    assert isinstance(acc, MultiStepAccounting)
    np.testing.assert_allclose(acc.m0, [0.5, 0.5], atol=1e-15)
    for name in ("excess_log_loss", "prior_kl", "likelihood_kl", "world_model_kl", "rhs", "residual", "martingale_residual", "tower_residual"):
        assert getattr(acc, name) == pytest.approx(0.0, abs=1e-15), name


def test_anchor_m2_a_belief_that_never_updates():
    acc = multistep_accounting(M_PATHS, M_TRUTH, [0.5, 0.5], [[0.5, 0.5], [0.5, 0.5]], {(): [0.5, 0.5]})
    assert acc.excess_log_loss == pytest.approx(INFO, abs=1e-15)
    assert acc.prior_kl == pytest.approx(0.0, abs=1e-15)
    assert acc.likelihood_kl == pytest.approx(INFO, abs=1e-15)
    assert acc.world_model_kl == pytest.approx(0.0, abs=1e-15)
    assert acc.rhs == pytest.approx(INFO, abs=1e-15) and acc.residual == pytest.approx(0.0, abs=1e-15)
    assert acc.martingale_residual == pytest.approx(0.0, abs=1e-15)      # a constant belief is trivially a martingale


def test_anchor_m3_a_wrong_world_model_is_netted_out():
    acc = multistep_accounting(M_PATHS, M_TRUTH, [0.7, 0.3], [[0.5, 0.5], [0.5, 0.5]], {(): [0.5, 0.5]})
    likelihood = 0.5 * (0.8 * math.log(0.8 / 0.7) + 0.2 * math.log(0.2 / 0.3)) + 0.5 * (0.2 * math.log(0.2 / 0.7) + 0.8 * math.log(0.8 / 0.3))
    world_model = 0.5 * math.log(5 / 7) + 0.5 * math.log(5 / 3)
    np.testing.assert_allclose(acc.m0, [0.5, 0.5], atol=1e-15)
    assert acc.excess_log_loss == pytest.approx(INFO, abs=1e-15)
    assert acc.prior_kl == pytest.approx(0.0, abs=1e-15)
    assert acc.likelihood_kl == pytest.approx(likelihood, abs=1e-15)
    assert acc.world_model_kl == pytest.approx(world_model, abs=1e-15)
    assert acc.rhs == pytest.approx(likelihood - world_model, abs=1e-15)
    assert acc.residual == pytest.approx(0.0, abs=1e-15)


def test_a_raw_belief_that_drifts_shows_up_in_m0_and_the_residuals():
    # The reader moves to (0.9, 0.1) whatever the outcome, from a prior of (1/2, 1/2).
    acc = multistep_accounting(M_PATHS, M_TRUTH, [0.5, 0.5], [[0.9, 0.1], [0.9, 0.1]], {(): [0.5, 0.5]})
    np.testing.assert_allclose(acc.m0, [0.9, 0.1], atol=1e-15)           # the implied class marginal is the drifted one
    assert acc.prior_kl == pytest.approx(0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1), abs=1e-15)
    assert acc.martingale_residual == pytest.approx(0.4, abs=1e-15)      # |0.9 - 0.5|
    assert acc.tower_residual == pytest.approx(0.4, abs=1e-15)
    assert acc.residual == pytest.approx(0.0, abs=1e-14)                 # the accounting still closes


# ------------------------------------------------------------------------------------------------
# T = 3, K = 3, C = 3 by enumeration
# ------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module", params=["real_index", "real_posterior"])
def problems(request):
    rng = np.random.default_rng(20261113)
    return [_problem(rng, request.param) for _ in range(15)]


def test_both_ways_of_naming_the_real_result_give_the_same_track():
    by_index = [_problem(np.random.default_rng(3), "real_index") for _ in range(3)]
    by_posterior = [_problem(np.random.default_rng(3), "real_posterior") for _ in range(3)]
    for (_, _, _, _, _, a), (_, _, _, _, _, b) in zip(by_index, by_posterior):
        for node in a:
            np.testing.assert_allclose(b[node], a[node], rtol=0, atol=1e-12)


def test_corrected_beliefs_are_a_martingale_under_the_world_model(problems):
    worst = 0.0
    for _, _, phat, _, _, coh in problems:
        for h in PREFIXES:
            expected_next = sum(phat[h][o] * coh[h + (o,)] for o in range(K))     # E_{P_hat}[c_{t+1} | h_t]
            worst = max(worst, float(np.abs(expected_next - coh[h]).max()))
    assert worst < 1e-9


def test_tower_property_with_class_marginal_c0(problems):
    worst_prior = worst_tower = 0.0
    for c0, _, _, path_probs, _, coh in problems:
        M = path_probs[:, None] * np.array([coh[p] for p in PATHS])               # the one joint model
        worst_prior = max(worst_prior, float(np.abs(M.sum(0) - c0).max()))        # its class marginal is c_0
        for h in PREFIXES[1:]:
            idx = [i for i, p in enumerate(PATHS) if p[: len(h)] == h]
            conditional = M[idx].sum(0) / M[idx].sum()                            # M(y | o_1..o_t)
            worst_tower = max(worst_tower, float(np.abs(conditional - coh[h]).max()))
    assert worst_prior < 1e-9 and worst_tower < 1e-9


@pytest.mark.parametrize("which", ["corrected", "raw"])
def test_t_step_identity_by_enumeration(problems, which):
    worst_identity = worst_terms = 0.0
    for c0, truth_paths, _, path_probs, raw, coh in problems:
        beliefs = coh if which == "corrected" else raw
        leaf = np.array([beliefs[p] for p in PATHS])
        implied, lhs, prior_kl, likelihood_kl, world_model_kl = _enumerated_terms(truth_paths, path_probs, leaf)
        worst_identity = max(worst_identity, abs(lhs - (prior_kl + likelihood_kl - world_model_kl)))   # the statement itself
        acc = multistep_accounting(PATHS, truth_paths, path_probs, leaf, {h: beliefs[h] for h in PREFIXES})
        np.testing.assert_allclose(acc.m0, implied, atol=1e-13)
        worst_terms = max(
            worst_terms,
            abs(acc.excess_log_loss - lhs),
            abs(acc.prior_kl - prior_kl),
            abs(acc.likelihood_kl - likelihood_kl),
            abs(acc.world_model_kl - world_model_kl),
            abs(acc.rhs - (prior_kl + likelihood_kl - world_model_kl)),
            abs(acc.residual - (lhs - (prior_kl + likelihood_kl - world_model_kl))),
        )
        assert abs(acc.residual) < 1e-9
        if which == "corrected":
            np.testing.assert_allclose(acc.m0, c0, atol=1e-9)                     # m_0 = c_0 at every horizon
            assert acc.prior_kl == pytest.approx(_kl(truth_paths.sum(0), c0), abs=1e-9)
            assert acc.martingale_residual < 1e-9 and acc.tower_residual < 1e-9
    assert worst_identity < 1e-9 and worst_terms < 1e-12


def test_the_raw_reader_is_not_the_conditional_of_one_joint_model(problems):
    drifts, towers, martingales = [], [], []
    for c0, truth_paths, _, path_probs, raw, _ in problems:
        leaf = np.array([raw[p] for p in PATHS])
        acc = multistep_accounting(PATHS, truth_paths, path_probs, leaf, {h: raw[h] for h in PREFIXES})
        drifts.append(float(np.abs(acc.m0 - c0).sum()))
        towers.append(acc.tower_residual)
        martingales.append(acc.martingale_residual)
    assert np.mean(drifts) > 0.05                    # cee_check.py reports a mean L1 drift of 0.33 after 3 steps
    assert min(towers) > 1e-3 and min(martingales) > 1e-3


def test_horizons_one_and_two_also_close(problems):
    # E1-D reports the terms at every horizon: truncate the tree and marginalize the truth.
    for c0, truth_paths, phat, _, raw, coh in problems[:5]:
        for horizon in (1, 2):
            paths = list(itertools.product(range(K), repeat=horizon))
            truth_h = np.array([truth_paths[[i for i, p in enumerate(PATHS) if p[:horizon] == h]].sum(0) for h in paths])
            probs_h = np.array([np.prod([phat[h[:t]][h[t]] for t in range(horizon)]) for h in paths])
            prefixes = [h for t in range(horizon) for h in itertools.product(range(K), repeat=t)]
            for beliefs in (coh, raw):
                acc = multistep_accounting(paths, truth_h, probs_h, np.array([beliefs[h] for h in paths]), {h: beliefs[h] for h in prefixes})
                assert abs(acc.residual) < 1e-9
            np.testing.assert_allclose(
                multistep_accounting(paths, truth_h, probs_h, np.array([coh[h] for h in paths]), {h: coh[h] for h in prefixes}).m0, c0, atol=1e-9
            )


def test_malformed_problems_are_rejected():
    with pytest.raises(ValueError):                  # a prefix belief is missing
        multistep_accounting([(0, 0), (0, 1), (1, 0), (1, 1)], np.full((4, 2), 0.125), [0.25] * 4, np.full((4, 2), 0.5), {(): [0.5, 0.5]})
    with pytest.raises(ValueError):                  # one path is a prefix of another
        multistep_accounting([(0,), (0, 1)], np.full((2, 2), 0.25), [0.5, 0.5], np.full((2, 2), 0.5), {(): [0.5, 0.5]})
    with pytest.raises(ValueError):                  # shapes disagree
        multistep_accounting(M_PATHS, M_TRUTH, [0.5, 0.5, 0.0], [[0.5, 0.5], [0.5, 0.5]], {(): [0.5, 0.5]})


# ------------------------------------------------------------- zero world-model mass (review F002)
def test_an_unreachable_branch_is_left_out_of_the_residuals():
    # re-exposure at step 1: P_hat puts all its mass on o_1 = 0, so the prefix (1,) is unreachable.
    # The corrected beliefs below (1,) are arbitrary, and both residuals are still exactly zero.
    paths = [(0, 0), (0, 1), (1, 0), (1, 1)]
    truth = np.array([[0.4, 0.1], [0.1, 0.4], [0.0, 0.0], [0.0, 0.0]])
    probs = np.array([0.5, 0.5, 0.0, 0.0])
    leaf = np.array([[0.8, 0.2], [0.2, 0.8], [0.3, 0.7], [0.6, 0.4]])
    prefix = {(): [0.5, 0.5], (0,): [0.5, 0.5], (1,): [0.9, 0.1]}
    with np.errstate(all="raise"):
        found = multistep_accounting(paths, truth, probs, leaf, prefix)
    assert found.martingale_residual == pytest.approx(0.0, abs=1e-15)
    assert found.tower_residual == pytest.approx(0.0, abs=1e-15)
    assert found.residual == pytest.approx(0.0, abs=1e-15) and found.excess_log_loss == pytest.approx(0.0, abs=1e-15)


def test_a_class_the_belief_never_covers_gives_an_infinite_prior_term_without_nan():
    leaf = np.array([[1.0, 0.0], [1.0, 0.0]])                     # m_0 = (1, 0) while b_star = (1/2, 1/2)
    with np.errstate(all="raise"):
        found = multistep_accounting(M_PATHS, M_TRUTH, [0.5, 0.5], leaf, {(): [1.0, 0.0]})
    assert found.prior_kl == math.inf and found.rhs == math.inf and found.excess_log_loss == math.inf
    assert math.isfinite(found.likelihood_kl)
