"""Sequential tracking: the stateful and the stateless track (SPEC 5.9).

Both take the index of the real outcome among the outcomes the world model proposed. The raw
reader's vectors b_t and q_o are floored; the tracked belief c_t is not, because flooring it would
break the martingale E_{P_hat}[c_{t+1} | h_t] = c_t that the stateful track has by construction
(SPEC 5.12 (i)). Start a track from c_0 = floor_norm(R(h_0)).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .core import MAX_ITER, TOL, Projection, ipf, reader_pair
from .scaling import RTOL, ScaleResult, scale_positive


@dataclass(frozen=True)
class Step:
    """One step of a track (SPEC 5.9).

    belief        c_{t+1}, the belief carried forward after the real outcome
    projection    the projection record of the step (at tau_B when a scale stage ran)
    scale         the CEE-Scale record, or None when no scale stage was asked for
    fallback      True when the projection did not converge and the raw update was used (SPEC 5.3)
    """

    belief: np.ndarray
    projection: Projection
    scale: Optional[ScaleResult]
    fallback: bool


def _step(target, weights, proposals, real_index, scale, tol, max_iter):
    """Project `proposals` onto mean `target`, after an optional scale stage, and read off the real outcome."""
    if not 0 <= real_index < weights.size:
        raise IndexError("real_index must be the position of the real outcome among the outcomes")
    if scale is None:
        scaled, projection = None, ipf(target, weights, proposals, tol, max_iter)
    else:
        budget, variant = scale
        scaled = scale_positive(target, weights, proposals, budget, variant, RTOL, tol, max_iter)
        projection = scaled.projection
    if projection.converged:
        return Step(projection.corrected[real_index], projection, scaled, False)
    return Step(proposals[real_index], projection, scaled, True)


def stateful_step(c_t, b_t, weights, posteriors, real_index, scale=None, tol=TOL, max_iter=MAX_ITER):
    """Stateful track (SPEC 5.9): c_{t+1} proportional to u_t x c_t x rho_real.

    With increments rho_o = q_o / b_t, the proposals m_o proportional to c_t x rho_o are projected
    onto mean c_t to give u_t. scale = (budget, variant) tempers the increments to the budget
    before projection (SPEC 5.11). If the projection does not converge the uncorrected proposal
    c_t x rho_real is carried forward and fallback is True.
    """
    b, w, q = reader_pair(b_t, weights, posteriors)
    c = np.asarray(c_t, dtype=float)
    if c.shape != b.shape or not np.all(np.isfinite(c)) or np.any(c <= 0) or abs(c.sum() - 1.0) > 1e-6:
        raise ValueError("c_t must be a strictly positive belief over the reader's classes")
    proposals = c * q / b
    return _step(c, w, proposals / proposals.sum(1, keepdims=True), real_index, scale, tol, max_iter)


def stateless_step(b_t, weights, posteriors, real_index, scale=None, tol=TOL, max_iter=MAX_ITER):
    """Stateless track (SPEC 5.9): c_{t+1} proportional to u_t x q_real, with u_t computed against b_t.

    scale = (budget, variant) tempers the reader's posteriors to the budget before projection
    (SPEC 5.11). If the projection does not converge the raw posterior q_real is carried forward
    and fallback is True.
    """
    b, w, q = reader_pair(b_t, weights, posteriors)
    return _step(b, w, q, real_index, scale, tol, max_iter)
