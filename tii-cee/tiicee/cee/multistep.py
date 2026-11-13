"""Multi-step accounting by enumeration (SPEC 5.12; PROOF section 7).

Accounting statements, not a guarantee of improvement. Nothing here is floored: the inputs are an
enumerated problem, not reader output at a decision point.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import kl


@dataclass(frozen=True)
class MultiStepAccounting:
    """Statement (iii) of SPEC 5.12 for one belief process, and the residuals of (i) and (ii).

    excess_log_loss = prior_kl + likelihood_kl - world_model_kl, exactly, for any belief process.

    m0                    implied class marginal, sum over paths of P_hat(path) x bel_T(path)
    excess_log_loss       E_O[KL(P_star(Y | O) || bel_T)]
    prior_kl              KL(b_star || m_0): the accumulated-drift term; m_0 = c_0 for the corrected agent
    likelihood_kl         E_Y[KL(P_star(O | Y) || M(O | Y))]: error of the likelihood the agent implicitly uses
    world_model_kl        KL(P_star(O) || P_hat(O)); it enters the identity with a minus sign
    rhs                   prior_kl + likelihood_kl - world_model_kl
    residual              excess_log_loss - rhs
    martingale_residual   max over prefixes h and classes of |E_{P_hat}[bel(h + o) | h] - bel(h)|
    tower_residual        max over prefixes h and classes of |M(y | h) - bel(h)|; at the root, |m_0 - bel_0|
    """

    m0: np.ndarray
    excess_log_loss: float
    prior_kl: float
    likelihood_kl: float
    world_model_kl: float
    rhs: float
    residual: float
    martingale_residual: float
    tower_residual: float


def multistep_accounting(paths, truth, path_probs, leaf_beliefs, prefix_beliefs):
    """The three terms of SPEC 5.12 (iii), the implied class marginal, and the martingale and tower residuals.

    paths            the outcome paths of the enumerated tree, as tuples, none a prefix of another
    truth            P_star(path, y), one row per path
    path_probs       P_hat(path) = product over steps of P_hat(o_t | h_{t-1}), one per path
    leaf_beliefs     the belief at the end of each path, one row per path
    prefix_beliefs   mapping from every proper prefix of a path (the root is ()) to the belief there

    The joint the belief process implies is M(path, y) = P_hat(path) x bel(y; path). For the
    corrected agent both residuals are zero and m0 = c_0 at every horizon (SPEC 5.12 (i), (ii));
    for a raw reader they are not. To account at an earlier horizon, pass the truncated tree with
    the truth marginalized over the later outcomes.

    Zero world-model mass is legitimate (exact re-exposure makes P_hat a point mass). A prefix with
    P_hat(h) = 0 is unreachable under the world model, so the martingale and tower statements, which
    condition on h, say nothing there and it is left out of both residuals. A labeled class with
    m_0(y) = 0 makes the prior term +inf and has no conditional M(O | y); it is left out of the
    likelihood term, so the right side is +inf whatever value that class would add.
    """
    paths = [tuple(p) for p in paths]
    truth, probs, leaf = np.asarray(truth, dtype=float), np.asarray(path_probs, dtype=float), np.asarray(leaf_beliefs, dtype=float)
    if truth.ndim != 2 or truth.shape != leaf.shape or probs.shape != (len(paths),) or truth.shape[0] != len(paths):
        raise ValueError("truth and leaf_beliefs need one row per path, and path_probs one entry per path")
    members = {}                                    # proper prefix -> indices of the paths below it
    for i, p in enumerate(paths):
        for t in range(len(p)):
            members.setdefault(p[:t], []).append(i)
    beliefs = {tuple(h): np.asarray(v, dtype=float) for h, v in prefix_beliefs.items()}
    if set(members) != set(beliefs) or set(members) & set(paths) or len(set(paths)) != len(paths):
        raise ValueError("paths must be distinct with none a prefix of another, and prefix_beliefs must cover exactly their proper prefixes")

    joint = probs[:, None] * leaf                   # M(path, y)
    m0 = joint.sum(0)
    p_star, b_star = truth.sum(1), truth.sum(0)
    seen, labeled = p_star > 0, (b_star > 0) & (m0 > 0)
    excess = float(p_star[seen] @ kl(truth[seen] / p_star[seen, None], leaf[seen]))
    likelihood = float(b_star[labeled] @ kl((truth[:, labeled] / b_star[labeled]).T, (joint[:, labeled] / m0[labeled]).T))
    prior_kl, world_model_kl = float(kl(b_star, m0)), float(kl(p_star, probs))
    rhs = prior_kl + likelihood - world_model_kl

    beliefs.update({p: leaf[i] for i, p in enumerate(paths)})
    martingale = tower = 0.0
    for h, idx in members.items():
        idx = np.asarray(idx)
        reach = probs[idx].sum()
        if reach <= 0:                              # unreachable under P_hat: nothing to state
            continue
        tower = max(tower, float(np.abs(joint[idx].sum(0) / joint[idx].sum() - beliefs[h]).max()))
        mass = {}                                   # next outcome -> P_hat mass of the paths through it
        for i in idx:
            child = paths[i][: len(h) + 1]
            mass[child] = mass.get(child, 0.0) + probs[i]
        expected = sum(m * beliefs[child] for child, m in mass.items() if m > 0) / reach
        martingale = max(martingale, float(np.abs(expected - beliefs[h]).max()))
    return MultiStepAccounting(m0, excess, prior_kl, likelihood, world_model_kl, rhs, excess - rhs, martingale, tower)
