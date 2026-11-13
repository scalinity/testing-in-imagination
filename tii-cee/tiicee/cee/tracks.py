"""Sequential tracking: the stateful and the stateless track (SPEC 5.9).

The real result is named either by its position among the outcomes the world model proposed
(real_index) or by the reader's belief after it (real_posterior). SPEC 5.9 needs only that belief,
and a sampled world model will usually not have proposed the result that arrives. The raw reader's
vectors b_t, q_o and q_real are floored; the tracked belief c_t is not, because flooring it would
break the martingale E_{P_hat}[c_{t+1} | h_t] = c_t that the stateful track has by construction
(SPEC 5.12 (i)). Start a track from c_0 = floor_norm(R(h_0)).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .core import MAX_ITER, TOL, Projection, ipf, reader_pair, reader_vector
from .scaling import RTOL, ScaleResult, scale_positive, tempered


@dataclass(frozen=True)
class Step:
    """One step of a track (SPEC 5.9).

    belief        c_{t+1}, the belief carried forward after the real result
    projection    the projection record of the step (at tau_B when a scale stage ran)
    scale         the CEE-Scale record, or None when no scale stage was asked for
    fallback      True when the projection did not converge and the raw update was used (SPEC 5.3); with a
                  scale stage this is the projection at tau_B, and the search's own convergence is
                  scale.search_converged
    """

    belief: np.ndarray
    projection: Projection
    scale: Optional[ScaleResult]
    fallback: bool


def _real_vector(real_index, real_posterior, n_classes):
    """The reader's floored belief after the real result, or None when the result is named by index."""
    if (real_index is None) == (real_posterior is None):
        raise ValueError("give exactly one of real_index and real_posterior")
    if real_posterior is None:
        return None
    q_real = reader_vector(real_posterior, "real_posterior")
    if q_real.shape != (n_classes,):
        raise ValueError("real_posterior must be one belief over the reader's classes")
    return q_real


def _step(target, weights, proposals, real_index, real_proposal, scale, tol, max_iter):
    """Project `proposals` onto mean `target`, after an optional scale stage, and apply the result to the real proposal.

    The real proposal is proposals[real_index], or real_proposal when the real result was not
    among the imagined ones. The scale stage tempers it by the same tau_B before u is applied.
    """
    if real_proposal is None:
        if not 0 <= real_index < weights.size:
            raise IndexError("real_index must be the position of the real outcome among the outcomes")
        real_proposal = proposals[real_index]
    if scale is None:
        scaled, projection, moved = None, ipf(target, weights, proposals, tol, max_iter), real_proposal
    else:
        budget, variant = scale
        scaled = scale_positive(target, weights, proposals, budget, variant, RTOL, tol, max_iter)
        projection, moved = scaled.projection, tempered(target, real_proposal, scaled.tau_b)
    if not projection.converged:
        return Step(real_proposal, projection, scaled, True)
    belief = projection.u * moved
    return Step(belief / belief.sum(), projection, scaled, False)


def _propose(c, q, b):
    """Stateful proposals: c_t x (q / b_t), normalized."""
    m = c * q / b
    return m / m.sum(-1, keepdims=True)


def stateful_step(c_t, b_t, weights, posteriors, real_index=None, scale=None, tol=TOL, max_iter=MAX_ITER, *, real_posterior=None):
    """Stateful track (SPEC 5.9): c_{t+1} proportional to u_t x c_t x rho_real.

    With increments rho_o = q_o / b_t, the proposals m_o proportional to c_t x rho_o are projected
    onto mean c_t to give u_t. Name the real result by exactly one of real_index (its position
    among the outcomes) or real_posterior (the reader's belief after it, q_real, so that
    rho_real = q_real / b_t). scale = (budget, variant) tempers the increments to the budget before
    projection, and the real increment by the same tau_B before u_t is applied (SPEC 5.11). If the
    projection does not converge the uncorrected, untempered c_t x rho_real is carried forward and
    fallback is True.
    """
    b, w, q = reader_pair(b_t, weights, posteriors)
    q_real = _real_vector(real_index, real_posterior, b.size)
    c = np.asarray(c_t, dtype=float)
    if c.shape != b.shape or not np.all(np.isfinite(c)) or np.any(c <= 0) or abs(c.sum() - 1.0) > 1e-6:
        raise ValueError("c_t must be a strictly positive belief over the reader's classes")
    real_proposal = None if q_real is None else _propose(c, q_real, b)
    return _step(c, w, _propose(c, q, b), real_index, real_proposal, scale, tol, max_iter)


def stateless_step(b_t, weights, posteriors, real_index=None, scale=None, tol=TOL, max_iter=MAX_ITER, *, real_posterior=None):
    """Stateless track (SPEC 5.9): c_{t+1} proportional to u_t x q_real, with u_t computed against b_t.

    Name the real result by exactly one of real_index (its position among the outcomes) or
    real_posterior (the reader's belief after it, q_real). scale = (budget, variant) tempers the
    reader's posteriors to the budget before projection, and q_real by the same tau_B before u_t is
    applied (SPEC 5.11). If the projection does not converge the raw posterior q_real is carried
    forward and fallback is True.
    """
    b, w, q = reader_pair(b_t, weights, posteriors)
    return _step(b, w, q, real_index, _real_vector(real_index, real_posterior, b.size), scale, tol, max_iter)
